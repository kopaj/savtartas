# `lane_following_cam` package
Lane following based on camera as a ROS 2 python package.  

[![Static Badge](https://img.shields.io/badge/ROS_2-Humble-34aec5)](https://docs.ros.org/en/humble/)
[![Static Badge](https://img.shields.io/badge/ROS_2-Jazzy-34aec5)](https://docs.ros.org/en/jazzy/)

## Packages and build

It is assumed that the workspace is `~/ros2_ws/`.

### Clone the packages
``` r
cd ~/ros2_ws/src
```
``` r
git clone https://github.com/kopaj/savtartas
```

### Build ROS 2 packages
``` r
cd ~/ros2_ws
```
``` r
colcon build --packages-select lane_following_cam --symlink-install
```

<details>
<summary> Don't forget to source before ROS commands.</summary>

``` bash
source ~/ros2_ws/install/setup.bash
```
</details>

## Run the package

Start the lane detection node with **compressed** or **raw** image, e.g.:

``` r
ros2 run lane_following_cam lane_detect --ros-args -p image_topic:=/image_raw/compressed -p raw_image:=false
```

``` r
ros2 run lane_following_cam lane_detect --ros-args -p image_topic:=/image_raw -p raw_image:=true
```

``` r
ros2 run lane_following_cam lane_detect --ros-args -p image_topic:=/camera/color/image_raw -p raw_image:=true
```

There are launch files as well: 

``` r
ros2 launch lane_following_cam example_bag.launch.py
```

``` r
ros2 launch lane_following_cam robot_raw1.launch.py
```

``` r
ros2 launch lane_following_cam robot_compressed1.launch.py
```


### Start the camera

``` r 
~/ros2_ws/src/savtartas/shell/start_drivers.sh
```

## Use ROS 2 bag (mcap)

Link: [drive.google.com/drive/folders/181pcgmwZ5YjaCReB-aZJtW1JtxnJUaru?usp=sharing](https://drive.google.com/drive/folders/181pcgmwZ5YjaCReB-aZJtW1JtxnJUaru?usp=sharing)

```r
ros2 bag play runde_vdi_lausitz_1.mcap --loop
```

This bag file contains `/camera/color/image_raw` topic, so easiest way to use is:

``` r
ros2 launch lane_following_cam example_bag.launch.py brightness:=125 saturation:=10 multiplier_bottom:=0.8 multiplier_top:=0.65 divisor:=7.5 cam_align:=-50
```
