#!/usr/bin/env python
import numpy as np
import rospy
#from geometry_msgs.msg import Pose
from reef_msgs.msg import DesiredState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from mag_pf_pkg.msg import ParticleMean
from mag_pf_pkg.msg import Particle
from geometry_msgs.msg import PointStamped
import scipy.stats as stats


class Guidance():
    def __init__(self):
        # Initialization of variables
        self.turtle_pose = np.array([0, 0, 0])
        self.linear_velocity = np.array([0, 0])
        self.angular_velocity = np.array([0])
        deg2rad = np.pi/180

        ## PARTICLE FILTER  ##
        # boundary of the lab [[x_min, y_min], [x_max, y_,max]]
        AVL_dims = np.array([[-0.5, -1.5], [2.5, 1.5]])  # road network outline
        # number of particles
        self.N = 500
        self.measurement_update_time = 2.
        # uniform distribution of particles (x, y, theta)
        self.particles = np.random.uniform([AVL_dims[0,0],AVL_dims[0,1],-np.pi],
                               [AVL_dims[1,0],AVL_dims[1,1],np.pi], (self.N, 3))
        self.weights = np.ones(self.N) / self.N
        self.measurement_covariance = np.array(
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, deg2rad*5]])
        self.noise_inv = np.linalg.inv(self.measurement_covariance)
        # Process noise: q11, q22 is meters of error per meter, q33 is radians of error per revolution
        self.proces_covariance = np.array(
            [[0.02, 0, 0], [0, 0.02, 0], [0, 0, deg2rad*5]])  
        self.var = self.measurement_covariance
        #self.particles = np.random.multivariate_normal(
        #    np.array([1.3, -1.26, 0]), 2*self.measurement_covariance, self.N)
        rospy.loginfo("Number of particles: %d", self.particles.shape[0])

        self.initial_time = rospy.get_time()
        self.time_reset = 0
        self.last_time = 0

        # ROS stuff
        rospy.loginfo("Initializing markov_goal_pose node")
        self.pose_pub = rospy.Publisher(
            'desired_state', DesiredState, queue_size=1)
        self.turtle_odom_sub = rospy.Subscriber(
            '/robot0/odom', Odometry, self.turtle_odom_cb, queue_size=1)
        self.quad_odom_sub = rospy.Subscriber(
            '/multirotor/truth/NWU', Odometry, self.quad_odom_cb, queue_size=1)

        # Particle filter ROS stuff
        self.particle_pub = rospy.Publisher(
            'xyTh_estimate', ParticleMean, queue_size=1)
        self.err_estimate_pub = rospy.Publisher(
            'err_estimate', PointStamped, queue_size=1)
        self.entropy_pub = rospy.Publisher(
            'entropy', Float32, queue_size=1)

    def particle_filter(self):
        self.predict()
        if rospy.get_time() - self.time_reset - self.initial_time > self.measurement_update_time:
            # update particles every measurement_update_time seconds
            self.time_reset = rospy.get_time()
            rospy.loginfo("Updating weight of particles")
            #rospy.loginfo("Neff: %f < %f", self.neff(self.weights), self.N/2)
            self.update()

        if self.neff(self.weights) < self.N/2:
            rospy.logwarn("Resampling particles. Neff: %f < %f",
                          self.neff(self.weights), self.N/2)
            self.resample()
        self.estimate()

        self.H = self.entropy_particle(self.particles, self.weights, self.noisy_turtle_pose)

        self.pub_pf()
        self.pub_desired_state()

    def predict(self):
        """Uses the process model to propagate the belief in the system state.
        In our case, the process model is the motion of the turtlebot in 2D with added gaussian noise. 
        In MML the predict step is a forward pass on the NN.
        Input: State of the particles
        Output: Predicted (propagated) state of the particles
        """
        t = rospy.get_time()
        dt = t - self.last_time - self.initial_time

        delta_theta = self.angular_velocity[0]*dt
        self.particles[:, 2] = self.particles[:, 2] + delta_theta + \
            (delta_theta/(2*np.pi)) * \
            self.add_noise(np.zeros(self.N), self.proces_covariance[2, 2], size=self.N)
        self.yaw_mean = np.mean(self.particles[:, 2])
        for ii in range(self.N):
            if np.abs(self.particles[ii, 2]) > np.pi:
                # Wraps angle
                self.particles[ii, 2] = self.particles[ii, 2] - \
                    np.sign(self.particles[ii, 2]) * 2 * np.pi
        delta_distance = self.linear_velocity[0]*dt + self.linear_velocity[0]*dt*self.add_noise(
            0, self.proces_covariance[0, 0], size=self.N)
        self.particles[:, :2] = self.particles[:, :2] + np.array([delta_distance*np.cos(self.particles[:, 2]),
                                                                    delta_distance*np.sin(self.particles[:, 2])]).T
 
        self.last_time = t - self.initial_time

    @staticmethod
    def add_noise(mean, covariance, size=1):
        """Add noise to the mean from a gaussian distribution with covariance matrix"""
        if type(mean) is np.ndarray and type(covariance) is np.ndarray:
            if mean.ndim > 1:
                size = mean.shape[0]
                noise = np.random.multivariate_normal(np.zeros(mean.shape[1]), covariance, size)
            else:
                size = mean.shape[0]
                #print('shape of mean: ', mean.shape)
                noise = np.random.multivariate_normal(np.zeros(size), covariance)
        else:
            noise = np.random.normal(0, covariance, size)

        return mean + noise

    def update(self):
        """Uses the measurement model to update the belief in the system state.
        In our case, the measurement model is the position of the turtlebot in 2D. 
        In MML the update step is the camera model.
        Input: Likelihood of the particles from measurement model and prior belief of the particles
        Output: Updated (posterior) state of the particles
        """
        self.weights = self.get_weight(
            self.particles, self.noisy_turtle_pose, self.weights)
        self.weights = self.weights / \
            np.sum(self.weights) if np.sum(self.weights) > 0 else self.weights

    def get_weight(self, particles, y_act, weight):
        """Particles that are closer to the noisy measurements are weighted higher than
        particles which don't match the measurements very well.
        """
        for ii in range(self.N):
            # The factor sqrt(det((2*pi)*measurement_cov)) is not included in the
            # likelihood, but it does not matter since it can be factored
            # and then cancelled out during the normalization.
            like = -0.5 * \
                (particles[ii, :] -
                 y_act)@self.noise_inv@(particles[ii, :] - y_act)
            weight[ii] = weight[ii]*np.exp(like)

            # another way to implement the above line
            #weight[ii] *= stats.multivariate_normal.pdf(x=particles[ii,:], mean=y_act, cov=self.measurement_covariance)
        return weight

    def resample(self):
        """Uses the resampling algorithm to update the belief in the system state. In our case, the
        resampling algorithm is the multinomial resampling, where the particles are copied randomly with
        probability proportional to the weights plus some roughening from Crassidis and Junkins.
        Inputs: Updated state of the particles
        Outputs: Resampled updated state of the particles
        """
        self.weights = self.weights / \
            np.sum(self.weights) if np.sum(self.weights) > 0 else self.weights
        indexes = np.random.choice(a=self.N, size=self.N, p=self.weights)
        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
        # Roughening. See Bootstrap Filter from Crassidis and Junkins.
        G = 0.2
        E = np.array([0, 0, 0])
        for ii in range(self.turtle_pose.shape[0]):
            E[ii] = np.max(self.particles[ii, :]) - \
                np.min(self.particles[ii, :])
        cov = (G*E*self.N**(-1/3))**2
        P_sigmas = np.diag(cov)

        for ii in range(self.N):
            self.particles[ii, :] = self.add_noise(
                self.particles[ii, :], P_sigmas)

    def estimate(self):
        """returns mean and variance of the weighted particles"""
        if np.sum(self.weights) > 0.0:
            self.weighted_mean = np.average(self.particles, weights=self.weights, axis=0)
            self.var  = np.average((self.particles - self.weighted_mean)**2, weights=self.weights, axis=0)

    def neff(self, weights):
        """Compute the effective number of particles"""
        return 1. / np.sum(np.square(weights))

    def entropy_particle(self, particles, weights, y_act):
        """Compute the entropy of the particle distribution"""
        process_part_like = np.zeros(self.N)
        # likelihoof of measurement p(zt|xt)
        like_meas = stats.multivariate_normal.pdf(x=particles, mean=y_act, cov=self.measurement_covariance)
        #print('like_meas: ', like_meas)
        #print('lik_meas shape: ', like_meas.shape)
        # likelihood of particle p(xt|xt-1)
        for ii in range(self.N):
            # maybe kinematics with gaussian
            like_particle = stats.multivariate_normal.pdf(x=particles, mean=self.particles[ii,:], cov=self.proces_covariance)
            #print('like_particle: ', like_particle)
            #print('lik_particle shape: ', like_particle.shape)
            process_part_like[ii] = np.sum(like_particle)
            # process part like is repeating every loop
        #print('process_part_like: ', process_part_like)
        #print('sum of process_part_like: ', np.sum(process_part_like))
        entropy = np.log(np.sum(like_meas*weights)) - np.sum(np.log(like_meas*process_part_like*np.sum(weights)*self.N)*weights)
        #self.entropy_particle_pub.publish(entropy)
        #TODO: finish entropy
        return entropy

    def turtle_odom_cb(self, msg):
        turtle_position = np.array([msg.pose.pose.position.x,
                                    msg.pose.pose.position.y])
        turtle_orientation = np.array([msg.pose.pose.orientation.x,
                                       msg.pose.pose.orientation.y,
                                       msg.pose.pose.orientation.z,
                                       msg.pose.pose.orientation.w])
        # Gazebo covariance of the pose of the turtlebot [x, y, theta]
        #self.covariance = np.diag([msg.pose.covariance[0], msg.pose.covariance[7], msg.pose.covariance[14]])
        self.linear_velocity = np.array([msg.twist.twist.linear.x,
                                         msg.twist.twist.linear.y])
        self.angular_velocity = np.array([msg.twist.twist.angular.z])

        _, _, theta_z = self.euler_from_quaternion(turtle_orientation)
        self.turtle_pose = np.array(
            [turtle_position[0], turtle_position[1], theta_z])
        self.noisy_turtle_pose = self.add_noise(
            self.turtle_pose, self.measurement_covariance)

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

    def pub_desired_state(self):
        ds = DesiredState()
        ds.pose.x = self.turtle_pose[0]
        ds.pose.y = -self.turtle_pose[1]
        ds.pose.z = -1.5
        ds.pose.yaw = 0
        ds.position_valid = True
        ds.velocity_valid = False
        self.pose_pub.publish(ds)

    def pub_pf(self):
        mean_msg = ParticleMean()
        mean_msg.mean.x = self.particles[:, 0].mean()
        mean_msg.mean.y = self.particles[:, 1].mean()
        mean_msg.mean.yaw = self.yaw_mean 
        for ii in range(self.N):
            particle_msg = Particle()
            particle_msg.x = self.particles[ii, 0]
            particle_msg.y = self.particles[ii, 1]
            particle_msg.yaw = self.particles[ii, 2]
            particle_msg.weight = self.weights[ii]
            mean_msg.all_particle.append(particle_msg)
        mean_msg.cov = np.diag(self.var).flatten('C')
        #self.mean_msg.cov = self.full_cov
        err_msg = PointStamped()
        err_msg.point.x = self.particles[:, 0].mean(
        ) - self.turtle_pose[0]
        err_msg.point.y = self.particles[:, 1].mean(
        ) - self.turtle_pose[1]
        err_msg.point.z = self.particles[:, 2].mean(
        ) - self.turtle_pose[2]

        entropy_msg = Float32()
        entropy_msg.data = self.H
        self.entropy_pub.publish(entropy_msg)

        self.particle_pub.publish(mean_msg)
        self.err_estimate_pub.publish(err_msg)


if __name__ == '__main__':
    try:
        rospy.init_node('guidance', anonymous=True)
        square_chain = Guidance()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
