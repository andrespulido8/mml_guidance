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
        self.noise_covariance = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])

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
        if self.neff(self.weights) < self.N/2:
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
        #self.particles_pred_position = self.particles[:,:2] + self.linear_velocity*0.1
        #self.particles_pred_angle = self.particles[:,2] + self.angular_velocity*0.1
        self.particles[:,0] = self.particles[:,0] + self.linear_velocity[0]*0.1
        self.particles[:,1] = self.particles[:,1] + self.linear_velocity[1]*0.1
        self.particles[:,2] = self.particles[:,2] + self.angular_velocity[0]*0.1
        self.particles = self.add_noise(self.particles, self.noise_covariance)
    
    def add_noise(self, mean, covariance):
        """Add noise to the particles from a gaussian distribution with covariance matrix""" 
        size = mean.shape[0]
        noise = np.random.multivariate_normal(np.zeros(3), covariance, size)
        return mean + noise

    def update(self):
        """The update step in the Bayes algorithm uses the measurement model to update the belief in the system state.
        In our case, the measurement model is the position of the turtlebot in 2D. In MML the update step is the camera model.
        Input: Likelihood of the particles from measurement model and prior belief of the particles
        Output: Updated state of the particles
        """
        self.weights = self.get_weight(self.particles, self.turtle_pose, self.weights)
        self.weights_sum = np.sum(self.weights)
        self.weights = self.weights / self.weights_sum
    
    def get_weight(self, particles, actual_pose, weight):
        """Particles that are closer to the noisy measurements are weighted higher than
        particles which don't match the measurements very well.
        """
        noisy_pose = self.add_noise(np.repeat([actual_pose],self.N, axis=0), self.noise_covariance)
        pos_weight = np.linalg.norm(particles[:,:2] - noisy_pose[:,:2], axis=1)
        angle_weight = np.abs(particles[:,2] - noisy_pose[:,2])
        pos_weight = self.normalize(pos_weight)
        angle_weight = self.normalize(angle_weight)
        weight *= pos_weight + angle_weight
        return weight

    @staticmethod
    def normalize(x):
        """Normalize a vector to have L2 norm equal to 1"""
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    
    def resample(self):
        """The resample step in the Bayes algorithm uses the resampling algorithm to update the belief in the system state.
        In our case, the resampling algorithm is the multinomial resampling, where the particles are copied randomly with
        probability proportional to the weights.
        Inputs: Updated state of the particles
        Outputs: Updated state of the particles
        """
        indexes = np.random.choice(self.N, self.N, p=self.weights)
        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
    
    def neff(self, weights):
        """Compute the effective number of particles"""
        return 1. / np.sum(np.square(weights))

    def turtle_odom_cb(self, msg):
        self.turtle_position = np.array([msg.pose.pose.position.x, 
                                         msg.pose.pose.position.y])
        turtle_orientation = np.array([msg.pose.pose.orientation.x,
                                        msg.pose.pose.orientation.y,
                                        msg.pose.pose.orientation.z,
                                        msg.pose.pose.orientation.w])
        # covariance of the pose of the turtlebot [x, y, theta]
        self.noise_covariance = np.diag([msg.pose.covariance[0], msg.pose.covariance[7], msg.pose.covariance[14]]) 
        self.linear_velocity = np.array([msg.twist.twist.linear.x,
                                            msg.twist.twist.linear.y])
        self.angular_velocity = np.array([msg.twist.twist.angular.z])

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

