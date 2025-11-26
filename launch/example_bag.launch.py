from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('brightness', default_value='-10'),
        DeclareLaunchArgument('multiplier_bottom', default_value='1.0'),
        DeclareLaunchArgument('multiplier_top', default_value='0.45'),
        DeclareLaunchArgument('divisor', default_value='9.0'),
        DeclareLaunchArgument('saturation', default_value='10'),
        DeclareLaunchArgument('cam_align', default_value='0'),
        DeclareLaunchArgument('islane', default_value='True'),
        DeclareLaunchArgument('wheelbase', default_value='0.33', 
                              description='Robot tengelytavolsaga meterben'),
        
        DeclareLaunchArgument('lookahead_dist', default_value='0.8', 
                              description='Eloretekintesi tavolsag meterben'),
        
        DeclareLaunchArgument('meters_per_pixel', default_value='0.002', 
                              description='Pixel -> Meter kalibracios ertek'),
        
        Node(
            package='lane_following_cam',
            executable='lane_detect',
            output='screen',
            parameters=[{
                'raw_image': True, # True for raw image, False for compressed image
                'image_topic': '/camera/color/image_raw',
                'brightness': LaunchConfiguration('brightness'),
                'multiplier_bottom': LaunchConfiguration('multiplier_bottom'),
                'multiplier_top': LaunchConfiguration('multiplier_top'),
                'divisor': LaunchConfiguration('divisor'),
                'saturation': LaunchConfiguration('saturation'),
                'cam_align': LaunchConfiguration('cam_align'),
                'islane': LaunchConfiguration('islane'),
                'wheelbase': LaunchConfiguration('wheelbase'),
                'lookahead_dist': LaunchConfiguration('lookahead_dist'),
                'meters_per_pixel': LaunchConfiguration('meters_per_pixel')
            }],
        ),
    ])