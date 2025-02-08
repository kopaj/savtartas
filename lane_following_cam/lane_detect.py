import cv2
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from rclpy.node import Node
import rclpy
from collections import deque

class LaneDetect(Node):
    def __init__(self):
        super().__init__('lane_detect')
        # parameters
        self.declare_parameter('raw_image', False)
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('debug', True)
        img_topic = self.get_parameter('image_topic').value
        if self.get_parameter('raw_image').value:
            self.sub1 = self.create_subscription(Image, img_topic, self.raw_listener, 10)
            self.sub1  # prevent unused variable warning
            self.get_logger().info(f'lane_detect subscribed to raw image topic: {img_topic}')
        else:
            self.sub2 = self.create_subscription(CompressedImage, '/image_raw/compressed', self.compr_listener, 10)
            self.sub2  # prevent unused variable warning
            self.get_logger().info(f'lane_detect subscribed to compressed image topic: {img_topic}')
        self.pub1 = self.create_publisher(Image, '/lane_img', 10)
        self.pub2 = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.debug = self.get_parameter('debug').value
        # New publisher for lane center
        self.center_pub = self.create_publisher(Float32, '/lane_center', 10)
        # For center smoothing
        self.center_history = deque(maxlen=5)  # Stores the previous five measured centers
        
    def raw_listener(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # print info of the image, only once not every time
        self.get_logger().info(f'First raw img arrived, shape: {cv_image.shape}', once=True)
        # Detect lanes
        lane_image = self.detect_lanes(cv_image)
        # Convert OpenCV image to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(lane_image, 'bgr8')
        # Publish the image
        self.pub1.publish(ros_image)
    
    def compr_listener(self, msg):
        # Convert ROS CompressedImage message to OpenCV image
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # print info of the image, only once not every time
        self.get_logger().info(f'First compressed img arrived, shape: {cv_image.shape}', once=True)
        # Detect lanes
        lane_image = self.detect_lanes(cv_image)
        # Convert OpenCV image to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(lane_image, 'bgr8')
        # Publish the image
        self.pub1.publish(ros_image)


    def detect_lanes(self, image): 
        # Adjust brightness until only lanes are visible
        imageBrighnessLow = cv2.convertScaleAbs(image, alpha=1, beta=-40)       

        hsv = cv2.cvtColor(imageBrighnessLow, cv2.COLOR_BGR2HSV)

        # Recognise lanes based on color (white, yellow)
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 25, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mask = cv2.bitwise_or(mask_white, mask_yellow)
        filtered_image = cv2.bitwise_and(image, image, mask=mask)

        # Convert to grayscale
        gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)
        # Defining ROI
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (1, height),
            (width, height),
            (width, int(height * 0.45)),         #big_track_munchen_only_camera_a.mcap: 0.45    f1tenth: 0.6
            (1, int(height * 0.45))              
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, maxLineGap=50)
        line_image = np.zeros_like(image)
        left_x = []
        right_x = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 80:  # Only accepting lines that measure over 80 px in length
                    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
                    if 0.1 < abs(slope) < 5.0:  # Filter based on slope
                        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        if x1 < width / 2 and x2 < width / 2:
                            left_x.extend([x1, x2])
                        elif x1 > width / 2 and x2 > width / 2:
                            right_x.extend([x1, x2])

        # Calculating the center
        
        center = width / 2  # Default center

        if left_x and right_x:
            left_avg = np.mean(left_x)
            right_avg = np.mean(right_x)
            center = (left_avg + right_avg) / 2
        elif left_x:
            left_avg = np.mean(left_x)
            center = (left_avg + (width * 1.6)) / (2 + (1-abs(slope)))
        elif right_x:
            right_avg = np.mean(right_x)
            center = right_avg / (2 + (1-abs(slope)))

        # Apply exponential smoothing and averaging using the deque
        self.center_history.append(center)
        smoothed_center = np.mean(self.center_history)
        
        # Twist logic

        twist = Twist()

        if not left_x and not right_x:
            # If there are no lines detected, slow the robot
            twist.angular.z = 0.0
            twist.linear.x = 0.05
            self.pub2.publish(twist)
        else:
            twist.linear.x = 0.2
            #   velocity (m/s)
            twist.angular.z = -0.005 * (smoothed_center - (width/2))
            #   angular velocity(rad/s)       turn left -> positive value, turn right -> negative value
            self.pub2.publish(twist)

         # Displaying center of lane using smoothed center
        cv2.line(line_image, (int(smoothed_center), height), (int(smoothed_center), int(height * 0.6)), (255, 0, 0), 2)
        cv2.circle(line_image, (int(smoothed_center), int(height / 2)), 10, (255, 0, 0), -1)

        # Display the twist.angular.z value on the image and direction (left or right or straight)

        font = cv2.FONT_HERSHEY_SIMPLEX
        if twist.angular.z > 0.01:
            text = 'Left'
        elif twist.angular.z < -0.01:
            text = 'Right'
        else:
            text = 'Straight'
        cv2.putText(line_image, f'{text} {abs(twist.angular.z):.2f}', (10, 30), font, 1, (60, 40, 200), 2, cv2.LINE_AA)
        
        # Combine the original image with the line image
        if self.debug:
            combined_image = cv2.addWeighted(image, 0.25, line_image, 1, 1)
        else: 
            combined_image = line_image

        # Publish the center as a Float32 message
        center_msg = Float32()
        center_msg.data = smoothed_center
        self.center_pub.publish(center_msg)

        return combined_image

def main(args=None):
    rclpy.init(args=args)
    lane_detect = LaneDetect()
    rclpy.spin(lane_detect)
    lane_detect.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()