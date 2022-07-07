#!/usr/bin/env python
from fileinput import filename
import numpy as np
import rospy
from geometry_msgs.msg import Pose
from reef_msgs.msg import DesiredState
from nav_msgs.msg import Odometry

class guidance():
    def __init__(self):
        ## Initialization of variables
        self.turtle_position = np.array([0, 0])
        self.turtle_pose = np.array([0, 0, 0])
        self.linear_velocity = np.array([0, 0])
        self.angular_velocity = np.array([0])

        ## PARTICLE FILTER  ##
        # boundary of the lab [[x_min, y_min], [x_max, y_,max]] 
        AVL_dims = np.array([[-2, -2], [2, 2]])  
        # number of particles
        self.N  = 10 
        # uniform distribution of particles (x, y, theta)
        self.particles = np.random.uniform([AVL_dims[0,0],AVL_dims[0,1],-np.pi],
                                [AVL_dims[1,0],AVL_dims[1,1],np.pi], (self.N, 3))  
        self.weights = np.ones(self.N) / self.N
        self.weights_sum = np.sum(self.weights)
        self.weights_cumsum = np.cumsum(self.weights)
        #self.weights_cumsum /= self.weights_cumsum[-1]
        #self.particles_cumsum = np.cumsum(self.particles, axis=0)
        #self.particles_cumsum /= self.particles_cumsum[-1, :]

        ## ROS stuff
        rospy.loginfo("Initializing markov_goal_pose node") 
        self.pose_pub = rospy.Publisher('desired_state', DesiredState, queue_size=1)
        self.turtle_odom_sub = rospy.Subscriber(
            '/robot0/odom', Odometry, self.turtle_odom_cb, queue_size=3)
        self.quad_odom_sub = rospy.Subscriber(
            '/multirotor/truth/NWU', Odometry, self.quad_odom_cb, queue_size=3)
        self.ds = DesiredState()

    def particle_filter(self):
        self.predict()
        self.update()
        self.resample()
        self.pub_desired_state()
        self.pose_pub.publish(self.ds)

    def predict(self):
        """The predict step in the Bayes algorithm uses the process model to update the belief in the system state.
        In our case, the process model is the motion of the turtlebot in 2D. In MML the predict step is a forward pass
        on the NN.
        Input: State of the particles
        Output: Predicted state of the particles
        """
        self.particles_pred_position = self.particles[:,:2] + self.linear_velocity*0.1
        self.particles_pred_angle = self.particles[:,2] + self.angular_velocity*0.1
        # TODO: add noise to the predicted state
        self.pred_pose = np.array([self.particles_pred_position[:,0], self.particles_pred_position[:,1], self.particles_pred_angle])

    def update(self):
        """The update step in the Bayes algorithm uses the measurement model to update the belief in the system state.
        In our case, the measurement model is the position of the turtlebot in 2D. In MML the update step is the camera model.
        Input: Likelihood of the particles from measurement model and prior belief of the particles
        Output: Updated state of the particles
        """
        self.weights = self.get_weight(self.particles, self.turtle_pose)
        self.weights_sum = np.sum(self.weights)
        self.weights_cumsum = np.cumsum(self.weights)
        self.weights_cumsum /= self.weights_sum
        # TODO: finish update step
    
    def get_weight(self, particles, pose):
        """Particles that closely match the measurements are weighted higher than
        particles which don't match the measurements very well.
        """
        pos_weight = np.linalg.norm((particles[:,:2] - pose[:,:2]), axis=0)
        angle_weight = np.abs(particles[:,2] - pose[:,2])
        pos_weight = self.normalize(pos_weight)
        angle_weight = self.normalize(angle_weight)
        weight = pos_weight + angle_weight
        # TODO: finish get_weight
        return weight

    def normalize(x):
        """Normalize a vector to have L2 norm equal to 1"""
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    
    def resample(self):
        """The resample step in the Bayes algorithm uses the resampling algorithm to update the belief in the system state.
        In our case, the resampling algorithm is the systematic resampling.
        Inputs: Updated state of the particles
        Outputs: Updated state of the particles
        """
        self.particles = self.particles_cumsum[np.random.choice(self.N, self.N, p=self.weights_cumsum)]
        self.weights_sum = np.sum(self.weights)
        self.weights_cumsum = np.cumsum(self.weights)
        self.weights_cumsum[-1] = 1.0  # avoid round-off errors
        indexes = np.searchsorted(self.weights_cumsum, np.random.rand(self.N))
        #self.weights_cumsum /= self.weights_sum
        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]

    def turtle_odom_cb(self, msg):
        self.turtle_position = np.array([msg.pose.pose.position.x, 
                                         msg.pose.pose.position.y])
        turtle_orientation = np.array([msg.pose.pose.orientation.x,
                                        msg.pose.pose.orientation.y,
                                        msg.pose.pose.orientation.z,
                                        msg.pose.pose.orientation.w])
        self.linear_velocity = np.array([msg.twist.twist.linear.x,
                                            msg.twist.twist.linear.y])
        self.angular_velocity = np.array([msg.twist.twist.angular.z])

        self.turtle_pose = np.array([self.turtle_position, turtle_orientation, self.linear_velocity, self.angular_velocity])
        _, _, theta_z = self.euler_from_quaternion(turtle_orientation)
        self.turtle_pose = np.array([self.turtle_position[0], self.turtle_position[1], theta_z]) 

        self.particle_filter()
                                    
    @staticmethod
    def euler_from_quaternion(q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = q[0]; y = q[1]; z = q[2]; w = q[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        return np.array([roll_x, pitch_y, yaw_z])  # in radians

    def quad_odom_cb(self, msg):
        self.quad_position = np.array([msg.pose.pose.position.x, 
                                         msg.pose.pose.position.y])
        self.pose_pub.publish(self.ds)
    
    def pub_desired_state(self):
        self.ds.pose.x = self.turtle_position[0]
        self.ds.pose.y = -self.turtle_position[1]
        self.ds.pose.z = -1.5
        self.ds.pose.yaw = 0
        self.ds.position_valid = True
        self.ds.velocity_valid = False

if __name__ == '__main__':
    try:
        rospy.init_node('guidance', anonymous=True)
        square_chain = guidance()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

