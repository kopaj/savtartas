import cv2
import numpy as np
import math
from geometry_msgs.msg import Twist, PointStamped, Point
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32, Header
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from rclpy.node import Node
import rclpy
from collections import deque
from filterpy.kalman import KalmanFilter

# --- ACKERMANN ODOMETRIA ÉS PURE PURSUIT LOGIKA ---
class AckermannLocalizer:
    def __init__(self, wheelbase=0.33, x=0.0, y=0.0, theta=0.0):
        self.L = wheelbase  # Tengelytáv (méter)
        self.x = x          # Globális X
        self.y = y          # Globális Y
        self.theta = theta  # Irányszög (radián)

    def update_odometry(self, velocity, steering_angle, dt):
        """ Pozíció frissítése a sebesség és kormányzási szög alapján (Open-Loop) """
        angular_velocity = (velocity / self.L) * math.tan(steering_angle)
        
        self.x += velocity * math.cos(self.theta) * dt
        self.y += velocity * math.sin(self.theta) * dt
        self.theta += angular_velocity * dt
        
        # Szög normalizálása -PI és PI közé
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

    def get_global_lookahead_point(self, lookahead_distance, lateral_error):
        """
        Kiszámolja a 3D célpontot a globális térben.
        lateral_error: pozitív = balra, negatív = jobbra (méterben)
        """
        # Lokális koordináták (Robot vázához rögzítve: X=Előre, Y=Balra)
        local_x = lookahead_distance
        local_y = lateral_error 
        
        # Transzformáció globális koordinátákba
        global_target_x = self.x + (local_x * math.cos(self.theta) - local_y * math.sin(self.theta))
        global_target_y = self.y + (local_x * math.sin(self.theta) + local_y * math.cos(self.theta))
        
        return (global_target_x, global_target_y)

    def pure_pursuit_control(self, target_x, target_y):
        """ Kiszámolja a szükséges kormányzási szöget """
        dx = target_x - self.x
        dy = target_y - self.y
        Ld = math.sqrt(dx**2 + dy**2) # Tényleges távolság a pontig
        
        if Ld == 0: return 0.0
        
        # A célpont szöge a robot jelenlegi irányához képest
        angle_to_target = math.atan2(dy, dx)
        alpha = angle_to_target - self.theta
        
        # Pure Pursuit képlet
        steering_angle = math.atan((2 * self.L * math.sin(alpha)) / Ld)
        return steering_angle

# --- FŐ ROS NODE ---

class LaneDetect(Node):
    def __init__(self):
        super().__init__('lane_detect')
        # parameters
        self.declare_parameter('raw_image', False)
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('debug', True)
        self.declare_parameter('brightness', -10)
        self.declare_parameter('multiplier_bottom', 1.0)
        self.declare_parameter('multiplier_top', 0.45)
        self.declare_parameter('divisor', 3.0)
        self.declare_parameter('saturation', 10)
        self.declare_parameter('cam_align', 0)
        self.declare_parameter('islane', True)
        
        # Pure Pursuit / Odometry Parameters
        self.declare_parameter('wheelbase', 0.33)        # Robot tengelytávja (m)
        self.declare_parameter('lookahead_dist', 1.0)    # Milyen messze nézzen előre (m)
        self.declare_parameter('meters_per_pixel', 0.002) # Pixel -> Méter konverziós ráta (KALIBRÁLNI KELL!)
        
        # Get parameter values
        self.brightness = self.get_parameter('brightness').value
        self.multiplier_bottom = self.get_parameter('multiplier_bottom').value
        self.multiplier_top = self.get_parameter('multiplier_top').value
        self.divisor = self.get_parameter('divisor').value
        self.saturation = self.get_parameter('saturation').value
        self.cam_align = self.get_parameter('cam_align').value
        self.islane = self.get_parameter('islane').value
        
        # Odometry params
        self.wheelbase = self.get_parameter('wheelbase').value
        self.lookahead_dist = self.get_parameter('lookahead_dist').value
        self.meters_per_pixel = self.get_parameter('meters_per_pixel').value

        img_topic = self.get_parameter('image_topic').value
        if self.get_parameter('raw_image').value:
            self.sub1 = self.create_subscription(Image, img_topic, self.raw_listener, 10)
        else:
            self.sub2 = self.create_subscription(CompressedImage, '/image_raw/compressed', self.compr_listener, 10)
            
        self.pub1 = self.create_publisher(Image, '/lane_img', 10)
        self.pub2 = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Debug publisher a célpontnak
        self.target_pub = self.create_publisher(PointStamped, '/pure_pursuit_target', 10)
        
        self.bridge = CvBridge()
        self.debug = self.get_parameter('debug').value
        self.center_pub = self.create_publisher(Float32, '/lane_center', 10)

        # Kalman Filter setup (változatlan)
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([0, 0])
        self.kf.F = np.array([[1, 1], [0, 1]])
        self.kf.H = np.array([[1, 0]])
        self.kf.P *= 1000
        self.kf.R = 10
        self.kf.Q = np.array([[0.01, 0], [0, 0.5]])

        self.center_history = deque(maxlen=3)
        self.width = 640
        self.height = 480
        
        # --- ODOMETRIA INICIALIZÁLÁS ---
        self.localizer = AckermannLocalizer(wheelbase=self.wheelbase)
        self.last_time = self.get_clock().now()
        self.current_steering_angle = 0.0
        self.current_velocity = 0.0

    def raw_listener(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.get_logger().info(f'First raw img arrived, shape: {cv_image.shape}', once=True)
        lane_image = self.detect_lanes(cv_image)
        ros_image = self.bridge.cv2_to_imgmsg(lane_image, 'bgr8')
        self.pub1.publish(ros_image)
    
    def compr_listener(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.get_logger().info(f'First compressed img arrived, shape: {cv_image.shape}', once=True)
        lane_image = self.detect_lanes(cv_image)
        ros_image = self.bridge.cv2_to_imgmsg(lane_image, 'bgr8')
        self.pub1.publish(ros_image)

    def lane_img(self, image):
        # ... (változatlan a te kódodhoz képest) ...
        imageBrightness = cv2.convertScaleAbs(image, alpha=1, beta=self.brightness)
        hsv = cv2.cvtColor(imageBrightness, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, self.saturation, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        lower_yellow = np.array([20, 200, 100], dtype=np.uint8)
        upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(mask_white, mask_yellow)
        filtered_image = cv2.bitwise_and(image, image, mask=mask)
        gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 75, 150)
        self.height, self.width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, int(self.height * self.multiplier_bottom)),
            (self.width, int(self.height * self.multiplier_bottom)),
            (self.width, int(self.height * self.multiplier_top)),
            (0, int(self.height * self.multiplier_top))
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        return cropped_edges

    def tube_img(self, image):
        # ... (változatlan) ...
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=10, beta=20)
        gray = cv2.GaussianBlur(gray, (15, 19), 0)
        edges = cv2.Canny(gray, 0, 50)
        self.height, self.width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, int(self.height * self.multiplier_bottom)),
            (self.width, int(self.height * self.multiplier_bottom)),
            (self.width, int(self.height * self.multiplier_top)),
            (0, int(self.height * self.multiplier_top))
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        kernel = np.ones((3, 3), np.uint8)
        final_edges = cv2.morphologyEx(cropped_edges, cv2.MORPH_CLOSE, kernel)
        return final_edges

    def shutdown_node(self):
        """
        Biztonságos leállítási rutin: Nullázza a sebességet és a kormányzást.
        """
        self.get_logger().warn('LaneDetect node shutting down. Stopping motors...')
        
        # Létrehoz egy nulla sebességű parancsot
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        
        # Üzenet küldése kétszer a hálózati biztonság érdekében
        try:
            # Első küldés
            self.pub2.publish(stop_cmd)
            # Várakozás a parancs feldolgozására
            time.sleep(0.1) 
            # Második küldés (garantálja, hogy megkapja a motorvezérlő node)
            self.pub2.publish(stop_cmd)
            self.get_logger().warn('Motors commanded to stop (Twist 0.0).')
        except Exception as e:
            self.get_logger().error(f"Hiba a Twist üzenet publikálásakor leállításkor: {e}")

    def detect_lanes(self, image): 
        # 1. ODOMETRIA RESET (Mivel bag file-t nézünk, minden képkocka "új kezdet")
        # Ha ezt nem teszed meg, a virtuális robot elmegy a világ végére, miközben a videó egy helyben áll!
        self.localizer.x = 0.0
        self.localizer.y = 0.0
        self.localizer.theta = 0.0
        
        # Időmérés (csak a dt miatt kell)
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # ... (Képfeldolgozás változatlan) ...
        if self.islane:
            lines = cv2.HoughLinesP(self.lane_img(image), 1, np.pi / 180, 50, maxLineGap=50)
        else:
            lines = cv2.HoughLinesP(self.tube_img(image), 1, np.pi / 180, 50, maxLineGap=50)

        line_image = np.zeros_like(image)
        left_x = []
        right_x = []

        # ... (Center history logika változatlan) ...
        if (len(self.center_history) == 0) or (self.center_history[-1] in range(-self.width, 2 * self.width)):
            check_center = self.width / 2
        else:
            deque_center = np.mean(self.center_history)
            self.kf.predict()
            self.kf.update(np.array([deque_center]))
            check_center = self.kf.x[0]

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 60:
                    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
                    if 0.1 < abs(slope) < 5.0:
                        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        if x1 < check_center and x2 < check_center:
                            if slope < 0.2:
                                x1 += self.width/(2*self.divisor)
                                x2 += self.width/(2*self.divisor)
                            left_x.extend([x1, x2])
                        elif x1 > check_center and x2 > check_center:
                            if slope < 0.2:
                                x1 -= self.width/(2*self.divisor)
                                x2 -= self.width/(2*self.divisor)
                            right_x.extend([x1, x2])

        center = self.width / 2
        if left_x and right_x:
            left_avg = np.mean(left_x)
            right_avg = np.mean(right_x)
            center = (left_avg + right_avg) / 2
        elif left_x:
            left_avg = np.mean(left_x)
            center = (left_avg + self.width) / 2
            center = center + ((self.width/self.divisor) * (5 * (1 - (self.divisor/10))) * (0.5 - (abs(center - left_avg)/self.width)))
        elif right_x:
            right_avg = np.mean(right_x)
            center = (right_avg - (self.width/self.divisor))/ 2
            center = center - ((self.width/self.divisor) * (5 * (1 - (self.divisor/10))) * (0.5 - (abs(center - right_avg)/self.width)))

        self.center_history.append(center)
        deque_center = np.mean(self.center_history)

        self.kf.predict()
        self.kf.update(np.array([deque_center]))
        smoothed_center = self.kf.x[0]

        # --- PURE PURSUIT SZÁMÍTÁS & DEBUG ---
        twist = Twist()

        if not left_x and not right_x:
            self.get_logger().warn("NINCS VONAL DETEKTÁLVA! (Ellenőrizd a brightness/saturation paramétereket)")
            twist.angular.z = 0.0
            twist.linear.x = 0.0
        else:
            # 1. Hiba számítás
            # smoothed_center: Ahol a sáv közepe van pixelben (0..640)
            # self.width / 2: A kép közepe (320)
            
            # HA A SÁV JOBBRA VAN (center > 320) -> hiba NEGATÍV -> Kormányzás JOBBRA
            # HA A SÁV BALRA VAN (center < 320) -> hiba POZITÍV -> Kormányzás BALRA
            pixel_error = (self.width / 2) - (smoothed_center + self.cam_align)
            
            lateral_error_m = pixel_error * self.meters_per_pixel
            
            # 2. Pure Pursuit
            target_point_3d = self.localizer.get_global_lookahead_point(self.lookahead_dist, lateral_error_m)
            new_steering_angle = self.localizer.pure_pursuit_control(target_point_3d[0], target_point_3d[1])
            
            # 3. Twist
            velocity = 0.2
            twist.linear.x = velocity
            twist.angular.z = (velocity / self.wheelbase) * math.tan(new_steering_angle)
            
            self.current_steering_angle = new_steering_angle

            # --- DIAGNOSZTIKA LOG (EZT FIGYELD A TERMINÁLBAN!) ---
            self.get_logger().info(
                f"Center:{smoothed_center:.0f} | "
                f"ErrPx:{pixel_error:.0f} | "
                f"ErrM:{lateral_error_m:.3f}m | "
                f"Steer:{math.degrees(new_steering_angle):.1f}° | "
                f"TgtX:{target_point_3d[0]:.2f} TgtY:{target_point_3d[1]:.2f}"
            )

            # --- VIZUALIZÁCIÓ ---
            # Most már biztosan tudjuk, hol a célpont relatíve a robothoz (Y tengelyen)
            # target_point_3d[1] a laterális távolság (Left+, Right-)
            
            # Visszaváltás pixelbe a megjelenítéshez
            viz_lateral_offset_px = target_point_3d[1] / self.meters_per_pixel
            # Mivel a Robot Y+ BALRA van, de a Kép X+ JOBBRA, ezért kivonjuk
            target_pixel_x = int((self.width / 2) - self.cam_align - viz_lateral_offset_px) 
            target_pixel_y = int(self.height * self.multiplier_top) # Kicsit a horizont alatt

            # CLAMP LOGIKA (Sárga, ha kimegy, Zöld ha bent van)
            draw_x = max(10, min(target_pixel_x, self.width - 10))
            color = (0, 255, 0) # Zöld
            if draw_x != target_pixel_x: color = (0, 255, 255) # Sárga (figyelmeztetés)

            # Rajzolás
            cv2.circle(line_image, (draw_x, target_pixel_y), 10, color, -1)
            cv2.line(line_image, (int(self.width/2) - self.cam_align, self.height), (draw_x, target_pixel_y), color, 2)
            cv2.putText(line_image, f"Tgt:{target_pixel_x}", (draw_x-20, target_pixel_y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.pub2.publish(twist)

        # Kék jelzések (Sáv közepe)
        cv2.line(line_image, (int((smoothed_center)), self.height), (int((smoothed_center)), int(self.height * (self.multiplier_top + 0.1))), (255, 0, 0), 2)
        cv2.circle(line_image, (int((smoothed_center)), int(self.height * self.multiplier_top)), 5, (255, 0, 0), -1)

        # Szöveg
        font = cv2.FONT_HERSHEY_SIMPLEX
        steer_deg = math.degrees(self.current_steering_angle)
        text = "Straight"
        if steer_deg > 1.0: text = f"Left {abs(steer_deg):.1f}"
        elif steer_deg < -1.0: text = f"Right {abs(steer_deg):.1f}"
        cv2.putText(line_image, f'{text}', (10, 30), font, 1, (60, 40, 200), 2, cv2.LINE_AA)
        
        if self.debug:
            combined_image = cv2.addWeighted(image, 0.3, line_image, 1, 1)
        else: 
            combined_image = line_image

        center_msg = Float32()
        center_msg.data = smoothed_center
        self.center_pub.publish(center_msg)

        return combined_image
"""
def main(args=None):
    rclpy.init(args=args)
    lane_detect = LaneDetect()
    rclpy.spin(lane_detect)
    lane_detect.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
"""
def main(args=None):
    # ROS 2 inicializálása
    rclpy.init(args=args)
    node = LaneDetect()
    
    try:
        # A Node futtatása, itt vár a Ctrl+C-re
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        # Ctrl+C esetén a program ide ugrik
        node.get_logger().info('Keyboard Interrupt received. Initiating shutdown.')
        
    finally:
        # Ez a blokk mindig lefut, függetlenül attól, hogy a spin() hogyan fejeződött be
        try:
            # 1. Meghívja a biztonságos leállítást (nulla Twist)
            node.shutdown_node()
        except Exception as e:
            node.get_logger().error(f"Hiba a shutdown_node-ban: {e}")
            
        # 2. A Node és a ROS 2 környezet leállítása
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()