cmake_minimum_required(VERSION 3.0.2)
project(human_robot_collaborative_learning)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  cartesian_state_msgs
  geometry_msgs
  roscpp
  rospy
  std_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
    Score.msg
)

add_service_files(
  FILES
    Reset.srv
)

generate_messages(
  DEPENDENCIES std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_generation roscpp rospy
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable (robot_control_sign src/robot_control_sign.cpp src/acc_control_sign.cpp)
target_link_libraries(robot_control_sign ${catkin_LIBRARIES})

add_executable (check_acc_scaling src/robot_control_sign.cpp src/check_acc_scaling.cpp)
target_link_libraries(check_acc_scaling ${catkin_LIBRARIES})

add_executable (baseline src/baseline.cpp)
target_link_libraries(baseline ${catkin_LIBRARIES})
