#!/usr/bin/env python
from fileinput import filename
import numpy as np
import rospy
from geometry_msgs.msg import Pose


class MarkovChain():
    def __init__(self):
        self.n_states = 5

        #self.visited_states = np.zeros(self.n_states) 
        self.init_time = np.array(rospy.get_time())
        self.prev_goal = 0 
        self.prev_mult = 0 
        
        # ROS stuff
        rospy.loginfo("Initializing markov_goal_pose node") 
        pose_pub = rospy.Publisher('goal_pose', Pose, queue_size=1)
        self.goal_pose_square()
        rate = rospy.Rate(10)  # Hz

        while not rospy.is_shutdown():
            p = self.pub_goal_pose()
            pose_pub.publish(p)
            rate.sleep()

    def goal_pose_square(self):
        """ Generates an square of sides 2*k"""
        self.goal_list = []
        z = 0  # turtlebot on the ground
        qx = qy = 0  # no roll or pitch
        k = 1  # Multiplier
        self.goal_list.append({'curr_goal':0, 'x': 0*k,  'y': 0*k,  'z': z, 'qx': qx, 'qy': qy, 'qz': 0,     'qw': 1})  
        self.goal_list.append({'curr_goal':1, 'x': 0  ,  'y': -1*k, 'z': z, 'qx': qx, 'qy': qy, 'qz': 0.707, 'qw': -0.707})  # 90 degress orientation
        self.goal_list.append({'curr_goal':2, 'x': 2*k,  'y': -1*k, 'z': z, 'qx': qx, 'qy': qy, 'qz': 0,     'qw': 1})
        self.goal_list.append({'curr_goal':3, 'x': 2*k,  'y': 1*k,  'z': z, 'qx': qx, 'qy': qy, 'qz': 0.707, 'qw': 0.707})
        self.goal_list.append({'curr_goal':4, 'x': 0*k,  'y': 1*k,  'z': z, 'qx': qx, 'qy': qy, 'qz': 1,     'qw': 0})  # 180 degress orientation
        self.trans_matrix = np.array([[0. , 0. , 1. , 0. , 0. ],
                                      [0. , 0. , 0.5, 0.5, 0. ],
                                      [0. , 0. , 0. , 1. , 0. ],
                                      [0. , 0. , 0. , 0. , 1. ],
                                      [0. , 1. , 0. , 0. , 0. ]])

    def pub_goal_pose(self):
        """ Gets time and publishes a goal pose every 10 seconds """
        time_step = 5 
        now = rospy.get_time() - self.init_time
        mult = np.floor(now/time_step)
        curr_goal = self.prev_goal 
        change = True if mult > self.prev_mult else False

        if now > 0 and now < time_step:  
            curr_goal = 0  # Start at the first state
        elif change:
            curr_goal = np.random.choice(np.arange(self.n_states),p=self.trans_matrix[curr_goal,:])

        goal_pose = self.goal_list[curr_goal]
        if curr_goal is not self.prev_goal:
            rospy.logwarn("New goal pose: x={}, y={} with index {}".format(goal_pose['x'], goal_pose['y'], curr_goal))

        p = Pose()
        p.position.x = goal_pose['x']
        p.position.y = goal_pose['y']
        p.position.z = goal_pose['z']
        p.orientation.x = goal_pose['qx']
        p.orientation.y = goal_pose['qy']
        p.orientation.z = goal_pose['qz']
        p.orientation.w = goal_pose['qw']

        self.prev_goal = curr_goal
        self.prev_mult = mult

        return p

if __name__ == '__main__':
    rospy.init_node('goal_pose_node', anonymous=True)
    square_chain = MarkovChain()
