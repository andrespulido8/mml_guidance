#!/usr/bin/env python
from fileinput import filename

import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped


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
        pose_pub = rospy.Publisher("goal_pose", PoseStamped, queue_size=2)
        self.goal_pose_square()
        rate = rospy.Rate(10)  # Hz
        # self.p = Pose()
        self.p = PoseStamped().pose

        while not rospy.is_shutdown():
            self.pub_goal_pose()
            pose_pub.publish(self.p)
            rate.sleep()

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
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )

    def odom_cb(self, msg):
        self.position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

    # self.position = np.array([msg.pose.position.x, msg.pose.position.y])

    def pub_goal_pose(self):
        """Gets time and publishes a goal pose every time_step seconds or after the goal is reached within tolerance_radius"""
        # if np.linalg.norm(self.position - self.goal_list[self.prev_goal]['x':'y']) < self.tolerance_radius:
        #    self.prev_goal = self.prev_goal + 1
        #    self.create_pose_msg(self.goal_list[self.prev_goal])
        time_step = 10  # amount of seconds until next goal pose TODO: change if desired
        now = rospy.get_time() - self.init_time
        mult = np.floor(now / time_step)
        curr_goal = self.prev_goal
        # change goal pose if time is greater than time_step
        change = True if mult > self.prev_mult else False

        if now > 0 and now < time_step:
            curr_goal = 0  # Start at the first state
        elif change:
            curr_goal = np.random.choice(
                np.arange(self.n_states), p=self.trans_matrix[curr_goal, :]
            )

        goal_pose = self.goal_list[curr_goal]
        if curr_goal is not self.prev_goal:
            rospy.logwarn(
                "New goal pose: x={}, y={} with index {}".format(
                    goal_pose["x"], goal_pose["y"], curr_goal
                )
            )
        # Publish the goal pose
        self.create_pose_msg(goal_pose)
        # Restart previous values
        self.prev_goal = curr_goal
        self.prev_mult = mult

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
    square_chain = MarkovChain()
