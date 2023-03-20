#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry

class MarkovChain:
    def __init__(self):
        self.is_sim = rospy.get_param("/is_sim", False)
        self.n_states = 5

        self.tolerance_radius = 0.2

        # self.visited_states = np.zeros(self.n_states)
        self.init_time = np.array(rospy.get_time())
        self.prev_goal = 0
        self.prev_mult = 0

        # ROS stuff
        rospy.loginfo(
            f"Initializing markov_goal_pose node with parameter is_sim: {self.is_sim}"
        )
        self.goal_pose_square()
        self.position = [0.,0.]
        if self.is_sim:
            self.pose_sub = rospy.Subscriber("odom", Odometry, self.odom_cb)
            self.pose_pub = rospy.Publisher("goal_pose", Pose, queue_size=2)
            self.p_msg = Pose()
        else:
            self.pose_sub = rospy.Subscriber("agent_pose", PoseStamped, self.odom_cb)
            self.pose_pub = rospy.Publisher("goal_pose", PoseStamped, queue_size=2)
            self.p_msg  = PoseStamped()
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
        if self.is_sim:
            self.position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        else:
            self.position = np.array([msg.pose.position.x, msg.pose.position.y])

    def compute_goal_pose(self):
        """Computes a new goal pose after the goal is reached within tolerance_radius"""
	    #rospy.logwarn(self.goal_list[self.prev_goal]['x','y'])
        d = np.linalg.norm(self.position - np.array([self.goal_list[self.prev_goal]['x'], self.goal_list[self.prev_goal]['y']]))
        if d < self.tolerance_radius:
            self.prev_goal = np.random.choice(len(self.goal_list), p=self.trans_matrix[self.prev_goal])

            goal_pose = self.goal_list[self.prev_goal]
            rospy.logwarn("New goal pose: x={}, y={} with index {}".format(
                    goal_pose["x"], goal_pose["y"], self.prev_goal
                )
            )
        else:
            goal_pose = self.goal_list[self.prev_goal]
        self.create_pose_msg(goal_pose)

    def create_pose_msg(self, goal_pose):
        """Creates a message and publishes it"""
        if self.is_sim:
            self.p_msg.position.x = goal_pose["x"]
            self.p_msg.position.y = goal_pose["y"]
            self.p_msg.position.z = goal_pose["z"]
            self.p_msg.orientation.x = goal_pose["qx"]
            self.p_msg.orientation.y = goal_pose["qy"]
            self.p_msg.orientation.z = goal_pose["qz"]
            self.p_msg.orientation.w = goal_pose["qw"]
        else:
            self.p_msg.pose.position.x = goal_pose["x"]
            self.p_msg.pose.position.y = goal_pose["y"]
            self.p_msg.pose.position.z = goal_pose["z"]
            self.p_msg.pose.orientation.x = goal_pose["qx"]
            self.p_msg.pose.orientation.y = goal_pose["qy"]
            self.p_msg.pose.orientation.z = goal_pose["qz"]
            self.p_msg.pose.orientation.w = goal_pose["qw"]
        self.pose_pub.publish(self.p_msg)


if __name__ == "__main__":
    rospy.init_node("goal_pose_node", anonymous=True)
    rate = rospy.Rate(10)  # Hz
    square_chain = MarkovChain()
    while not rospy.is_shutdown():
            square_chain.compute_goal_pose()
            rate.sleep()

