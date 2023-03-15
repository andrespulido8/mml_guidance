#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry


class MarkovChain:
    def __init__(self):
        self.n_states = 5

        self.tolerance_radius = 0.2

        # self.visited_states = np.zeros(self.n_states)
        self.init_time = np.array(rospy.get_time())
        self.prev_goal = 0
        self.prev_mult = 0

        # ROS stuff
        rospy.loginfo("Initializing markov_goal_pose node")
        self.pose_sub = rospy.Subscriber("agent_pose", PoseStamped, self.odom_cb)
        self.pose_sub = rospy.Subscriber("/odom", Odometry, self.odom_cb)
        self.pose_pub = rospy.Publisher("goal_pose", PoseStamped, queue_size=2)
        self.goal_pose_square()
        self.position = [0.,0.]
        self.ps = PoseStamped()
        self.p  = self.ps.pose
        self.create_pose_msg(self.goal_list[0])
	
    def goal_pose_square(self):
        """Generates an square of sides 2*k"""
        self.goal_list = []

        z = 0  # turtlebot on the ground
        qx = qy = 0  # no roll or pitch
        k = 1.25  # Multiplier  TODO: change this to make square bigger or smaller
        x_offset = -1.25  # TODO: change this to not crash to the net
        y_offset = 0.2
        self.goal_list.append(
            {
                "curr_goal": 0,
                "x": x_offset + 0 * k,
                "y": y_offset + 0 * k,
                "z": z,
                "qx": qx,
                "qy": qy,
                "qz": 0,
                "qw": 1,
            }
        )
        self.goal_list.append(
            {
                "curr_goal": 1,
                "x": x_offset + 0 * k,
                "y": y_offset + -1 * k,
                "z": z,
                "qx": qx,
                "qy": qy,
                "qz": 0.707,
                "qw": -0.707,
            }
        )  # 90 degrees orientation
        self.goal_list.append(
            {
                "curr_goal": 2,
                "x": x_offset + 2 * k,
                "y": y_offset + -1 * k,
                "z": z,
                "qx": qx,
                "qy": qy,
                "qz": 0,
                "qw": 1,
            }
        )
        self.goal_list.append(
            {
                "curr_goal": 3,
                "x": x_offset + 2 * k,
                "y": y_offset + 1 * k,
                "z": z,
                "qx": qx,
                "qy": qy,
                "qz": 0.707,
                "qw": 0.707,
            }
        )
        self.goal_list.append(
            {
                "curr_goal": 4,
                "x": x_offset + 0 * k,
                "y": y_offset + 1 * k,
                "z": z,
                "qx": qx,
                "qy": qy,
                "qz": 1,
                "qw": 0,
            }
        )  # 180 degrees orientation

        self.trans_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )

    def odom_cb(self, msg):
        self.position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
    def posestamped_cb(self, msg):
        self.position = np.array([msg.pose.position.x, msg.pose.position.y])

    # self.position = np.array([msg.pose.position.x, msg.pose.position.y])

    def pub_goal_pose(self):
        """Publishes a goal pose after the goal is reached within tolerance_radius"""
        #rospy.logwarn(self.goal_list[self.prev_goal]['x','y'])
        d = np.linalg.norm(self.position - np.array([self.goal_list[self.prev_goal]['x'], self.goal_list[self.prev_goal]['y']]))
        #rospy.logwarn("G: %d\tD: %.4f"%(self.prev_goal, d))
        if d < self.tolerance_radius:
        #if np.linalg.norm(self.position - self.goal_list[self.prev_goal]['x':'y']) < self.tolerance_radius:
            self.prev_goal = np.random.choice(len(self.goal_list), p=self.trans_matrix[self.prev_goal])

            goal_pose = self.goal_list[self.prev_goal]
            self.create_pose_msg(goal_pose)
            rospy.logwarn("New goal pose: x={}, y={} with index {}".format(
                    goal_pose["x"], goal_pose["y"], self.prev_goal
                )
            )

    def create_pose_msg(self, goal_pose):
        self.p.position.x = goal_pose["x"]
        self.p.position.y = goal_pose["y"]
        self.p.position.z = goal_pose["z"]
        self.p.orientation.x = goal_pose["qx"]
        self.p.orientation.y = goal_pose["qy"]
        self.p.orientation.z = goal_pose["qz"]
        self.p.orientation.w = goal_pose["qw"]


if __name__ == "__main__":
    rospy.init_node("goal_pose_node", anonymous=True)
    rate = rospy.Rate(10)  # Hz
    square_chain = MarkovChain()
    while not rospy.is_shutdown():
            square_chain.pub_goal_pose()
            square_chain.pose_pub.publish(square_chain.ps)
            rate.sleep()
            

