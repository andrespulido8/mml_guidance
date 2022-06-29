#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import Pose


def publisher():
    rospy.init_node('goal_pose_node', anonymous=True)
    pub = rospy.Publisher('goal_pose', Pose, queue_size=1)
    goal_list = goal_pose_rectangle()

    rate = rospy.Rate(10)  # Hz
    global init_time 
    init_time = np.array(rospy.get_time())
    ii_last = None
    rospy.loginfo("Initializing markov_goal_pose node") 
    while not rospy.is_shutdown():
        p, ii_last = pub_goal_pose(goal_list, ii_last)
        pub.publish(p)
        rate.sleep()
    
def goal_pose_rectangle():
    goal_list = []
    z = 0
    qx = qy = 0
    k = 1  # Multiplier
    goal_list.append({'x': 0, 'y': -1*k, 'z': z, 'qx': qx, 'qy': qy, 'qz': 0.707, 'qw': -0.707})  # 90 degress orientation
    goal_list.append({'x': -1*k, 'y': -1*k, 'z': z, 'qx': qx, 'qy': qy, 'qz': 0, 'qw': 1})
    goal_list.append({'x': 1*k, 'y': 1*k, 'z': z, 'qx': qx, 'qy': qy, 'qz': 0.707, 'qw': 0.707})
    goal_list.append({'x': 0*k, 'y': 1*k, 'z': z, 'qx': qx, 'qy': qy, 'qz': 1, 'qw': 0})  # 180 degress orientation
    goal_list.append({'x': 0*k, 'y': 0*k, 'z': z, 'qx': qx, 'qy': qy, 'qz': 0, 'qw': 1})  
    return goal_list

def pub_goal_pose(goal_list, ii_last):
    time_step = 5 
    now = np.array(rospy.get_time()) - init_time
    if now > 0 and now < time_step*1:
        ii = 0
    elif now > time_step*1 and now < time_step*2:
        ii = 1 
    elif now > time_step*2 and now < time_step*3:
        ii = 2 
    elif now > time_step*3 and now < time_step*4:
        ii = 3
    else:
        ii = 4
    
    goal_pose = goal_list[ii]
    if ii is not ii_last:
        rospy.loginfo("New goal pose: {} with index {}".format(goal_pose, ii))

    p = Pose()
    p.position.x = goal_pose['x']
    p.position.y = goal_pose['y']
    p.position.z = goal_pose['z']
    p.orientation.x = goal_pose['qx']
    p.orientation.y = goal_pose['qy']
    p.orientation.z = goal_pose['qz']
    p.orientation.w = goal_pose['qw']
    
    ii_last = ii

    return p, ii_last

if __name__ == '__main__':
    publisher()
