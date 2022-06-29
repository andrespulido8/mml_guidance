/*
 * drone_guidance.hpp
 *
 * Author: Andres Pulido 
 */

#include <ros/ros.h>

//#include <geometry_msgs/Pose.h>
//#include <geometry_msgs/TransformStamped.h>
//#include <geometry_msgs/Twist.h>
//#include <geometry_msgs/Vector3.h>
#include <math.h>
#include <nav_msgs/Odometry.h>
//#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
//#include <turtlesim/Pose.h>
#include <reef_msgs/DesiredState.h>
#include <array>

//TODO: change comments

/*!
@class drone_guidance 
@brief It is a ros c++ which compute a linear and an angular
velocity applied to a robot though a ros publisher. To do so, this class
subscribes through ros to the robot pose and to the robot goal pose
*/
class drone_guidance {
  public:
    /*!
    @brief Constructor of a drone_guidance instance 
    @param[in] _node  is the rosnode taken as argument
    @return Newly created drone_guidance instance
    */
    drone_guidance(ros::NodeHandle &node);
    
    /*!
    #@brief Callback function of the robot pose when the drone_guidance class
    subscribes to a geometry_msgs::TransformStamped msgs
    #@param[in] _msg  is the pose of the robot retrieved through ros and is in a
    geometry_msgs::TransformStamped format
    */
    //void poseTransFormCallback(const geometry_msgs::TransformStamped &_msg);
    
    /*!
    @brief Callback function of the turtlebot pose when the node subscribes to a nav_msgs::Odometry msgs
    @param[in] _msg  is the pose of the robot retrieved through ros and is in a
    //TransformStamped msg format
    */
    void turtle_odom_cb(const nav_msgs::Odometry &_msg);

    /*!    
    @brief Callback function of the robot goal pose when the drone_guidance 
    class subscribes to a geometry_msgs::Pose &msg
    @param[in] _msg  is the goal pose of the robot retrieved through ros and is
    in a geometry_msgs::Pose msg format
     */
    void desired_msg();
    
    /*!
    @brief Destructor of the drone_guidance instance 
    @param[in] _node  is the rosnode taken as argument
    */
    virtual ~drone_guidance();

  private:
    ros::Subscriber turtle_pose_sub;
    ros::Publisher des_state_pub;
    int queueSize;
    double VMax;
    double VMin;
    double PhiMax;
    reef_msgs::DesiredState quad_des_state;
    std::array<double, 3> turtle_min_pose;
    ros::Time prev_time;
    double dt = 0;
};

