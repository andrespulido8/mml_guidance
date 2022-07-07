/*
 *
 *  Author: Andres Pulido
 *  Date: June, 2022
 *
 */

#include "drone_guidance.hpp"

drone_guidance::drone_guidance(ros::NodeHandle &node) {
    // Command will be sent in NED frame 
    
    // Maximum linear velocity possibly applied to the robot
    double VMax =  1;
    // Minimum linear velocity applied to the robot
    double VMin = 0.1;
    // Maximum angular velocity possibly applied
    double PhiMax = 0.8;
    
    turtle_min_pose[0] = 0;
    turtle_min_pose[1] = 0;
    turtle_min_pose[2] = 0;

    ROS_INFO("Initializing drone guidance class");
    queueSize = 5;
    
    // register the first simulation time
    prev_time = ros::Time(0);

    turtle_pose_sub = node.subscribe("/robot0/odom", queueSize,
        &drone_guidance::turtle_odom_cb, this);

    ROS_INFO("Subscribed to turtle_pose_sub");
    des_state_pub = node.advertise<reef_msgs::DesiredState>(
        "desired_state", queueSize);

}
drone_guidance::~drone_guidance() {
    // TODO Auto-generated destructor stub
}

void drone_guidance::desired_msg() {
    // Construct desired state message
    quad_des_state.pose.x =  turtle_min_pose[0];
    quad_des_state.pose.y = -turtle_min_pose[1];  // -1 to transform to NED frame
    quad_des_state.pose.z = -1.5;
    quad_des_state.pose.yaw = turtle_min_pose[2];
//    quad_des_state.pose.yaw = turtle_min_pose[2];
    quad_des_state.position_valid = true;
    quad_des_state.velocity_valid = false;
    des_state_pub.publish(quad_des_state);

}

void drone_guidance::turtle_odom_cb(const nav_msgs::Odometry &msg) {

    // this function register the current pose of the robot thanks to gazebo
    //ROS_INFO("Received odometry message! now computing PID");
    turtle_min_pose[0] = msg.pose.pose.position.x;
    turtle_min_pose[1] = msg.pose.pose.position.y;
    turtle_min_pose[2] = tf::getYaw(msg.pose.pose.orientation);
    dt = (msg.header.stamp - prev_time).toSec();
    prev_time = msg.header.stamp;
    desired_msg();
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "drone_guidance_node");

  ros::NodeHandle node;
  drone_guidance drone_guidance(node);

  ros::spin();

  return 0;
}
