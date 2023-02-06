#!/usr/bin/env python3
import csv
import os

import numpy as np
import rospkg
import rospy
import scipy.stats as stats
from geometry_msgs.msg import PointStamped, PoseStamped
from mag_pf_pkg.msg import Particle, ParticleMean
from nav_msgs.msg import Odometry

# from geometry_msgs.msg import Pose
from reef_msgs.msg import DesiredState
from rosflight_msgs.msg import RCRaw
from std_msgs.msg import Bool, Float32, Float32MultiArray


class Guidance:
    def __init__(self):
        self.init_finished = False
        self.is_sim = rospy.get_param("/is_sim", False)
        self.is_info_guidance = True  # False is for tracking turtlebot
        # Initialization of variables
        self.turtle_pose = np.array([0, 0, 0])
        self.quad_position = np.array([0, 0])
        self.noisy_turtle_pose = np.array([0, 0, 0])
        self.goal_position = np.array([0, 0])
        self.linear_velocity = np.array([0, 0])
        self.angular_velocity = np.array([0])
        deg2rad = lambda deg: np.pi * deg / 180

        ## PARTICLE FILTER  ##
        self.is_viz = rospy.get_param("/is_viz", False)  # true to visualize plots
        # boundary of the lab [[x_min, y_min], [x_max, y_,max]]
        self.AVL_dims = np.array([[-1.2, -2], [2.8, 1.8]])  # road network outline
        # number of particles
        self.N = 500
        # Number of future measurements per sampled particle to consider in EER
        self.N_m = 1
        # Number of sampled particles
        self.N_s = 100
        # Time steps to propagate in the future for EER
        self.k = 2
        # initiate entropy
        self.Hp_t = 0  # partial entropy
        self.IG_range = np.array([0, 0, 0])
        self.t_EER = 0
        self.measurement_update_time = 0.5
        # uniform distribution of particles (x, y, theta)
        self.particles = np.random.uniform(
            [self.AVL_dims[0, 0], self.AVL_dims[0, 1], -np.pi],
            [self.AVL_dims[1, 0], self.AVL_dims[1, 1], np.pi],
            (self.N, 3),
        )
        self.prev_particles = np.copy(self.particles)
        self.weights = np.ones(self.N) / self.N
        self.prev_weights = np.ones(self.N) / self.N
        self.measurement_covariance = np.array(
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, deg2rad(5)]]
        )
        self.noise_inv = np.linalg.inv(self.measurement_covariance)
        # Process noise: q11, q22 is meters of error per meter, q33 is radians of error per revolution
        self.proces_covariance = np.array(
            [[0.04, 0, 0], [0, 0.04, 0], [0, 0, deg2rad(5)]]
        )
        self.var = np.diag(self.measurement_covariance)  # variance of particles
        self.H_gauss = 0  # just used for comparison
        self.weighted_mean = np.array([0, 0, 0])
        self.actions = np.array([[0.1, 0], [0, 0.1], [-0.1, 0], [0, -0.1]])
        self.position_following = False
        # Use multivariate normal if you know the initial condition
        # self.particles = np.random.multivariate_normal(
        #    np.array([1.3, -1.26, 0]), 2*self.measurement_covariance, self.N)

        # Measurement Model
        self.height = 1.5
        camera_angle = np.array(
            [deg2rad(69), deg2rad(42)]
        )  # camera angle in radians (horizontal, vertical)
        self.FOV_dims = np.tan(camera_angle) * self.height
        self.FOV = self.construct_FOV(self.quad_position)
        self.in_FOV = 0
        self.initial_time = rospy.get_time()
        self.time_reset = 0
        self.last_time = 0

        # ROS stuff
        rospy.loginfo(
            f"Initializing guidance node with parameter is_sim: {self.is_sim}"
        )
        if self.is_info_guidance:
            rospy.loginfo("Quadcopter in mode: Information Gain Guidance")
        else:
            rospy.loginfo("Quadcopter in mode: Position Target Tracking")
        self.pose_pub = rospy.Publisher("desired_state", DesiredState, queue_size=1)

        if self.is_sim:
            self.turtle_odom_sub = rospy.Subscriber(
                "/robot0/odom", Odometry, self.turtle_odom_cb, queue_size=1
            )
            self.quad_odom_sub = rospy.Subscriber(
                "/pose_stamped", PoseStamped, self.quad_odom_cb, queue_size=1
            )
        else:
            self.turtle_odom_sub = rospy.Subscriber(
                "/turtle_pose_stamped", PoseStamped, self.turtle_odom_cb, queue_size=1
            )
            self.rc_sub = rospy.Subscriber("rc_raw", RCRaw, self.rc_cb, queue_size=1)

        if self.is_viz:
            # Particle filter ROS stuff
            self.particle_pub = rospy.Publisher(
                "xyTh_estimate", ParticleMean, queue_size=1
            )
            self.err_estimate_pub = rospy.Publisher(
                "err_estimate", PointStamped, queue_size=1
            )
            self.entropy_pub = rospy.Publisher("entropy", Float32, queue_size=1)
            self.n_eff_pub = rospy.Publisher(
                "n_eff_particles", Float32MultiArray, queue_size=1
            )
            self.update_pub = rospy.Publisher("is_update", Bool, queue_size=1)
            self.fov_pub = rospy.Publisher("fov_coord", Float32MultiArray, queue_size=1)
            self.des_fov_pub = rospy.Publisher(
                "des_fov_coord", Float32MultiArray, queue_size=1
            )

        rospy.loginfo(
            "Number of particles for the Bayes Filter: %d", self.particles.shape[0]
        )
        rospy.sleep(1)
        self.init_finished = True

    def particle_filter(self):
        """Main function of the particle filter
        where the predict, update, resample, estimate
        and publish PF values for visualization is done
        """
        t = rospy.get_time()

        # print('check')
        # Prediction step
        self.particles, self.last_time = self.predict(
            self.particles, self.prev_particles, self.weights, self.last_time
        )

        self.update_msg = Bool()
        updt_time = t - self.time_reset - self.initial_time
        # print("reset time: ", reset_time)
        if updt_time > self.measurement_update_time:
            # update particles every measurement_update_time seconds
            self.time_reset = t - self.initial_time
            # rospy.loginfo("Updating weight of particles")
            # if self.is_in_FOV(self.noisy_turtle_pose, self.FOV):
            if True:
                self.prev_weights = np.copy(self.weights)
                self.weights = self.update(
                    self.weights, self.particles, self.noisy_turtle_pose
                )

            self.update_msg.data = True
        else:
            self.update_msg.data = False  # no update

        # print('check1')
        self.Neff = self.neff(self.weights)
        if self.Neff < self.N / 2 and self.update_msg.data:
            if self.Neff < self.N / 100:
                # particles are basically lost, reinitialize
                rospy.logwarn("Uniformly resampling particles")
                self.particles = np.random.uniform(
                    [self.AVL_dims[0, 0], self.AVL_dims[0, 1], -np.pi],
                    [self.AVL_dims[1, 0], self.AVL_dims[1, 1], np.pi],
                    (self.N, 3),
                )
                self.prev_particles = np.copy(self.particles)
                self.weights = np.ones(self.N) / self.N
            else:
                # some are good but some are bad, resample
                # rospy.logwarn(
                #    "Resampling particles. Neff: %f < %f",
                #    self.Neff,
                #    self.N / 2,
                # )
                self.resample()

        if self.is_viz:
            self.estimate()
            self.pub_pf()

    def current_entropy(self, candidates_index):
        # print("check2")
        now = rospy.get_time() - self.initial_time
        # Entropy of current distribution
        if self.is_in_FOV(self.noisy_turtle_pose, self.FOV):
            self.in_FOV = 1
            H = self.entropy_particle(
                self.prev_particles[candidates_index],
                np.copy(self.prev_weights[candidates_index]),
                self.particles[candidates_index],
                np.copy(self.weights[candidates_index]),
                self.noisy_turtle_pose,
            )
        else:
            self.in_FOV = 0
            H = self.entropy_particle(
                self.prev_particles[candidates_index],
                np.copy(
                    self.weights[candidates_index]
                ),  # current weights are the (t-1) weights because no update
                self.particles[candidates_index],
            )

        entropy_time = rospy.get_time() - self.initial_time
        # print("Entropy time: ", entropy_time - now)
        return H

    def information_driven_guidance(
        self,
    ):
        """Compute the current entropy and future entropy using particles
        to then compute the expected entropy reduction (EER) over predicted
        measurements. The next action is the one that minimizes the EER.
        Output:
        goal_position: the position of the measurement that resulted in the
        maximum EER
        """
        now = rospy.get_time() - self.initial_time
        ## Guidance
        future_weight = np.zeros((self.N, self.N_s))
        Hp_k = np.zeros(self.N_s)  # partial entropy
        Ip = np.zeros(self.N_s)  # partial information gain

        prev_future_parts = np.copy(self.prev_particles)
        future_parts = np.copy(self.particles)
        last_future_time = np.copy(self.last_time)
        for k in range(self.k):
            future_parts, last_future_time = self.predict(
                future_parts, prev_future_parts, self.weights, last_future_time + 0.1
            )
        # Future measurements
        prob = np.nan_to_num(self.weights, copy=True, nan=0)
        prob = prob / np.sum(prob)
        candidates_index = np.random.choice(a=self.N, size=self.N_s, p=prob)
        # Future possible measurements
        # TODO: implement N_m sampled measurements
        z_hat = self.add_noise(
            future_parts[candidates_index], self.measurement_covariance
        )
        self.Hp_t = self.current_entropy(candidates_index)
        # TODO: implement N_m sampled measurements (double loop)
        for jj in range(self.N_s):
            k_fov = self.construct_FOV(z_hat[jj])
            # currently next if statement will always be true
            # N_m implementation will change this by
            # checking for measrement outside of fov
            if self.is_in_FOV(z_hat[jj], k_fov):
                # TODO: read Jane's math to see exactly how EER is computed
                # the expectation can be either over all possible measurements
                # or over only the measurements from each sampled particle (1 in this case)
                future_weight[:, jj] = self.update(
                    self.weights, future_parts, z_hat[jj]
                )
            # H (x_{t+k} | \hat{z}_{t+k})
            # Partial entropy and full entropy
            Hp_k[jj] = self.entropy_particle(
                self.particles[candidates_index],
                np.copy(self.weights[candidates_index]),
                future_parts[candidates_index],
                future_weight[:, jj][candidates_index],
                z_hat[jj],
            )
            # Information Gain
            Ip[jj] = self.Hp_t - Hp_k[jj]

        # EER = I.mean() # implemented when N_m is implemented
        self.t_EER = rospy.get_time() - self.initial_time - now

        action_index = np.argmax(Ip)
        self.IG_range = np.array([np.min(Ip), np.mean(Ip), np.max(Ip)])

        # print("possible actions: ", z_hat[:, :2])
        # print("information gain: ", I)
        # print("Chosen action:", z_hat[action_index, :2])
        return z_hat[action_index][:2]

    def predict(self, particles, prev_particles, weights, last_time):
        """Uses the process model to propagate the belief in the system state.
        In our case, the process model is the motion of the turtlebot in 2D with added gaussian noise.
        In MML the predict step is a forward pass on the NN.
        Input: State of the particles
        Output: Predicted (propagated) state of the particles
        """
        t = rospy.get_time()
        dt = t - last_time - self.initial_time

        prev_particles = np.copy(particles)
        delta_theta = self.angular_velocity[0] * dt
        particles[:, 2] = (
            prev_particles[:, 2]
            + delta_theta
            + (delta_theta / (2 * np.pi))
            * self.add_noise(
                np.zeros(self.N), self.proces_covariance[2, 2], size=self.N
            )
        )

        for ii in range(self.N):
            if np.abs(particles[ii, 2]) > np.pi:
                # Wraps angle
                particles[ii, 2] = (
                    particles[ii, 2] - np.sign(particles[ii, 2]) * 2 * np.pi
                )

        # Component mean in the complex plane to prevent wrong average
        # source: https://www.rosettacode.org/wiki/Averages/Mean_angle#C.2B.2B
        self.yaw_mean = np.arctan2(
            np.sum(weights * np.sin(particles[:, 2])),
            np.sum(weights * np.cos(particles[:, 2])),
        )
        delta_distance = self.linear_velocity[0] * dt + self.linear_velocity[
            0
        ] * dt * self.add_noise(0, self.proces_covariance[0, 0], size=self.N)
        particles[:, :2] = (
            prev_particles[:, :2]
            + np.array(
                [
                    delta_distance * np.cos(particles[:, 2]),
                    delta_distance * np.sin(particles[:, 2]),
                ]
            ).T
        )

        last_time = t - self.initial_time

        return particles, last_time

    @staticmethod
    def add_noise(mean, covariance, size=1):
        """Add noise to the mean from a gaussian distribution with covariance matrix"""
        if type(mean) is np.ndarray and type(covariance) is np.ndarray:
            if mean.ndim > 1:
                size = mean.shape[0]
                noise = np.random.multivariate_normal(
                    np.zeros(mean.shape[1]), covariance, size
                )
            else:
                size = mean.shape[0]
                # print('shape of mean: ', mean.shape)
                noise = np.random.multivariate_normal(np.zeros(size), covariance)
        else:
            noise = np.random.normal(0, covariance, size)

        return mean + noise

    def update(self, weights, particles, noisy_turtle_pose):
        """Updates the belief in the system state.
        In our case, the measurement model is the position and orientation of the
        turtlebot with added noise from a gaussian distribution.
        In MML the update step is the camera model.
        Input: Likelihood of the particles from measurement model and prior belief of the particles
        Output: Updated (posterior) weight of the particles
        """
        weights = self.get_weight(particles, noisy_turtle_pose, weights)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        return weights

    def is_in_FOV(self, sparticle, fov) -> bool:
        """Check if the particles are in the FOV of the camera.
        Input: Particle, FOV
        Output: Boolean, True if the particle is in the FOV, False otherwise
        """
        return np.all(
            [
                sparticle[0] > fov[0],
                sparticle[0] < fov[1],
                sparticle[1] > fov[2],
                sparticle[1] < fov[3],
            ]
        )

    def construct_FOV(self, fov_center=np.array([0, 0])) -> np.ndarray:
        """Construct the FOV of the camera given the center
        of the FOV and the dimensions of the FOV
        Input: Center of the FOV
        Output: FOV dimensions in the form (x_min, x_max, y_min, y_max)
        """
        fov = np.array(
            [
                fov_center[0] - self.FOV_dims[0] / 2,
                fov_center[0] + self.FOV_dims[0] / 2,
                fov_center[1] - self.FOV_dims[1] / 2,
                fov_center[1] + self.FOV_dims[1] / 2,
            ]
        )
        return fov

    def get_weight(self, particles, y_act, weight):
        """Particles that are closer to the noisy measurements are weighted higher than
        particles which don't match the measurements very well.
        """
        # Update the weights of each particle. There are two methods to compute this:

        # Method 1: For loop
        # for ii in range(self.N):
        #    # The factor sqrt(det((2*pi)*measurement_cov)) is not included in the
        #    # likelihood, but it does not matter since it can be factored
        #    # and then cancelled out during the normalization.
        #    like = (
        #        -0.5
        #        * (particles[ii, :] - y_act)
        #        * self.noise_inv
        #        * (particles[ii, :] - y_act)
        #        @ self.noise_inv
        #        @ (particles[ii, :] - y_act)
        #    )
        #    weight[ii] = weight[ii] * np.exp(like)

        # Method 2: Vectorized using scipy.stats
        weight = weight * stats.multivariate_normal.pdf(
            x=particles, mean=y_act, cov=self.measurement_covariance
        )
        return weight

    def resample(self):
        """Uses the resampling algorithm to update the belief in the system state. In our case, the
        resampling algorithm is the multinomial resampling, where the particles are copied randomly with
        probability proportional to the weights plus some roughening from Crassidis and Junkins.
        Inputs: Updated state of the particles
        Outputs: Resampled updated state of the particles
        """
        self.weights = (
            self.weights / np.sum(self.weights)
            if np.sum(self.weights) > 0
            else self.weights
        )
        indexes = np.random.choice(a=self.N, size=self.N, p=self.weights)
        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
        # Roughening. See Bootstrap Filter from Crassidis and Junkins.
        G = 0.2
        E = np.array([0, 0, 0])
        for ii in range(self.turtle_pose.shape[0]):
            E[ii] = np.max(self.particles[ii, :]) - np.min(self.particles[ii, :])
        cov = (G * E * self.N ** (-1 / 3)) ** 2
        P_sigmas = np.diag(cov)

        for ii in range(self.N):
            self.particles[ii, :] = self.add_noise(self.particles[ii, :], P_sigmas)

    def estimate(self):
        """returns mean and variance of the weighted particles"""
        if np.sum(self.weights) > 0.0:
            self.weighted_mean = np.append(
                np.average(self.particles[:, :2], weights=self.weights, axis=0),
                self.yaw_mean,
            )
            angle_diff = self.particles[:, 2] - self.weighted_mean[2]
            angle_diff_sq = ((angle_diff + np.pi) % (2 * np.pi) - np.pi) ** 2
            yaw_var = np.arctan2(
                np.sum(self.weights * np.sin(angle_diff_sq)),
                np.sum(self.weights * np.cos(angle_diff_sq)),
            )
            self.var = np.append(
                np.average(
                    (self.particles[:, :2] - self.weighted_mean[:2]) ** 2,
                    weights=self.weights,
                    axis=0,
                ),
                yaw_var,
            )
            # source: Differential Entropy in Wikipedia - https://en.wikipedia.org/wiki/Differential_entropy
            self.H_gauss = (
                np.log((2 * np.pi * np.e) ** (3) * np.linalg.det(np.diag(self.var))) / 2
            )

    def neff(self, weights):
        """Compute the number of effective particles
        Source: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
        """
        return 1.0 / np.sum(np.square(weights))

    def entropy_particle(
        self,
        prev_particles,
        prev_wgts,
        particles,
        wgts=np.array([]),
        y_meas=np.array([]),
    ):
        """Compute the entropy of the particle distribution based on the equation in the
        paper: Y. Boers, H. Driessen, A. Bagchi, and P. Mandal, 'Particle filter based entropy'
        There are two computations, one for the case where the measurement is inside the fov
        and there is an update step before, and one where the measurement is outside the fov.
        Output: entropy: numpy.int64
        """
        if wgts.size > 0:
            # likelihoof of measurement p(zt|xt)
            # (how likely is each of the particles in the gaussian of the measurement)
            # TODO: change the N to be the general case (Default)
            like_meas = stats.multivariate_normal.pdf(
                x=particles, mean=y_meas, cov=self.measurement_covariance
            )

            # likelihood of particle p(xt|xt-1)
            process_part_like = np.zeros(prev_particles.shape[0])
            for ii in range(prev_particles.shape[0]):
                # maybe kinematics with gaussian
                # maybe get weight wrt to previous state (distance)
                like_particle = stats.multivariate_normal.pdf(
                    x=prev_particles, mean=particles[ii, :], cov=self.proces_covariance
                )
                # TODO: investigate if I need to multiply this by prev_wgts
                process_part_like[ii] = np.sum(like_particle)

            # Numerical stability
            cutoff = 1e-4
            like_meas[like_meas < cutoff] = np.nan
            prev_wgts[prev_wgts < cutoff] = np.nan
            # remove the nans from the likelihoods
            # like_meas = like_meas[~np.isnan(like_meas)]
            process_part_like[process_part_like < cutoff] = np.nan
            product = like_meas * prev_wgts
            notnans = product[~np.isnan(product)]
            notnans[notnans < cutoff * 0.01] = np.nan
            product[~np.isnan(product)] = notnans
            first_term = np.log(np.nansum(product))
            first_term = first_term if np.isfinite(first_term) else 0.0
            # second_term = np.nansum(np.log(prev_wgts)*weights)
            # third_term = np.nansum(weights*np.log(like_meas))
            # fourth_term = np.nansum(weights*np.log(process_part_like))

            entropy = (
                first_term
                - np.nansum(np.log(prev_wgts) * wgts)
                - np.nansum(wgts * np.log(like_meas))
                - np.nansum(wgts * np.log(process_part_like))
            )

            if np.abs(entropy) > 30:
                print("first term: ", np.log(np.nansum(like_meas * prev_wgts)))
                print("second term: ", np.nansum(np.log(prev_wgts) * wgts))
                print("third term: ", np.nansum(wgts * np.log(like_meas)))
                print("fourth term: ", np.nansum(wgts * np.log(process_part_like)))
                # print('like_meas min: ', like_meas.min())
                # print('like_meas max: ', like_meas.max())
                # print('like_meas mean: ', like_meas.mean())
                # print('like_meas std: ', like_meas.std())

                # print if first term is -inf
                if np.isinf(np.log(np.nansum(like_meas * prev_wgts))):
                    rospy.logwarn(
                        "first term of entropy is -inf. Likelihood is very small"
                    )
        else:
            # likelihood of particle p(xt|xt-1)
            process_part_like = np.zeros(prev_particles.shape[0])
            for ii in range(prev_particles.shape[0]):
                # maybe kinematics with gaussian
                # maybe get weight wrt to previous state (distance)
                like_particle = stats.multivariate_normal.pdf(
                    x=prev_particles, mean=particles[ii, :], cov=self.proces_covariance
                )
                process_part_like[ii] = np.sum(like_particle * prev_wgts)

            # Numerical stability
            cutoff = 1e-4
            process_part_like[process_part_like < cutoff] = np.nan
            prev_wgts[prev_wgts < cutoff] = np.nan

            entropy = -np.nansum(prev_wgts * np.log(process_part_like))

        return np.clip(entropy, -20, 1000)

    @staticmethod
    def euler_from_quaternion(q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]
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

    def turtle_odom_cb(self, msg):
        if self.init_finished:
            if self.is_sim:
                turtle_position = np.array(
                    [msg.pose.pose.position.x, msg.pose.pose.position.y]
                )
                turtle_orientation = np.array(
                    [
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w,
                    ]
                )
                # Gazebo covariance of the pose of the turtlebot [x, y, theta]
                # self.covariance = np.diag([msg.pose.covariance[0], msg.pose.covariance[7], msg.pose.covariance[14]])
                self.linear_velocity = np.array(
                    [msg.twist.twist.linear.x, msg.twist.twist.linear.y]
                )
                self.angular_velocity = np.array([msg.twist.twist.angular.z])

                _, _, theta_z = self.euler_from_quaternion(turtle_orientation)
                self.turtle_pose = np.array(
                    [turtle_position[0], turtle_position[1], theta_z]
                )
                self.noisy_turtle_pose = self.add_noise(
                    self.turtle_pose, self.measurement_covariance
                )
            else:
                self.turtle_pose = np.array([msg.pose.position.x, msg.pose.position.y])
                self.pub_desired_state()

    def rc_cb(self, msg):
        if msg.values[6] > 500:
            self.position_following = True
        else:
            self.position_following = False

    def quad_odom_cb(self, msg):
        if self.init_finished:
            self.quad_position = np.array([msg.pose.position.x, msg.pose.position.y])
            # self.quad_yaw = self.euler_from_quaternion(msg.pose.orientation)[2]  # TODO: unwrap message before function
            # self.quad_position[1] = -self.quad_position[1] if not self.is_info_guidance else self.quad_position[1]
            self.FOV = self.construct_FOV(self.quad_position)

    def pub_desired_state(self, is_velocity=False, xvel=0, yvel=0):
        if self.init_finished:
            ds = DesiredState()
            if self.position_following or self.is_sim:
                if is_velocity:
                    ds.velocity.x = xvel
                    ds.velocity.y = yvel
                    ds.position_valid = False
                    ds.velocity_valid = True
                else:
                    if self.is_info_guidance:
                        ds.pose.x = self.goal_position[0]
                        ds.pose.y = -self.goal_position[1]
                    else:
                        ds.pose.x = self.weighted_mean[0]
                        ds.pose.y = -self.weighted_mean[1]
                    ds.pose.yaw = 1.571  # 90 degrees
                    ds.position_valid = True
                    ds.velocity_valid = False
            else:
                ds.pose.x = 0
                ds.pose.y = 0
                ds.pose.yaw = 1.571
                ds.position_valid = True
                ds.velocity_valid = False
            ds.pose.z = -self.height
            self.pose_pub.publish(ds)
            if self.is_sim:
                # Entropy pub
                entropy_msg = Float32()
                entropy_msg.data = self.Hp_t
                self.entropy_pub.publish(entropy_msg)

    def pub_pf(self):
        mean_msg = ParticleMean()
        mean_msg.mean.x = self.weighted_mean[0]
        mean_msg.mean.y = self.weighted_mean[1]
        mean_msg.mean.yaw = self.weighted_mean[2]
        for ii in range(self.N):
            particle_msg = Particle()
            particle_msg.x = self.particles[ii, 0]
            particle_msg.y = self.particles[ii, 1]
            particle_msg.yaw = self.particles[ii, 2]
            particle_msg.weight = self.weights[ii]
            mean_msg.all_particle.append(particle_msg)
        mean_msg.cov = np.diag(self.var).flatten("C")
        err_msg = PointStamped()
        err_msg.point.x = self.weighted_mean[0] - self.turtle_pose[0]
        err_msg.point.y = self.weighted_mean[1] - self.turtle_pose[1]
        err_msg.point.z = self.weighted_mean[2] - self.turtle_pose[2]
        self.FOV_err = self.quad_position - self.turtle_pose[:2]
        # Particle pub
        self.particle_pub.publish(mean_msg)
        self.err_estimate_pub.publish(err_msg)
        # TODO: change publisher to service
        self.update_pub.publish(self.update_msg)
        # FOV pub
        fov_msg = Float32MultiArray()
        fov_matrix = np.array(
            [
                [self.FOV[0], -self.FOV[2]],
                [self.FOV[0], -self.FOV[3]],
                [self.FOV[1], -self.FOV[3]],
                [self.FOV[1], -self.FOV[2]],
                [self.FOV[0], -self.FOV[2]],
            ]
        )
        fov_msg.data = fov_matrix.flatten("C")
        self.fov_pub.publish(fov_msg)
        self.des_fov = self.construct_FOV(self.goal_position)
        des_fov_matrix = np.array(
            [
                [self.des_fov[0], self.des_fov[2]],
                [self.des_fov[0], self.des_fov[3]],
                [self.des_fov[1], self.des_fov[3]],
                [self.des_fov[1], self.des_fov[2]],
                [self.des_fov[0], self.des_fov[2]],
            ]
        )
        fov_msg.data = des_fov_matrix.flatten("C")
        self.des_fov_pub.publish(fov_msg)
        # Number of effective particles pub
        neff_msg = Float32MultiArray()
        neff_msg.data = np.array([self.Neff])
        self.n_eff_pub.publish(neff_msg)

        pkgDir = rospkg.RosPack().get_path("mml_guidance")
        with open(pkgDir + "/data/errors.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    err_msg.point.x,
                    err_msg.point.y,
                    err_msg.point.z,
                    self.FOV_err[0],
                    self.FOV_err[1],
                    self.Hp_t,
                    self.IG_range[0],
                    self.IG_range[1],
                    self.IG_range[2],
                    self.in_FOV,
                    self.t_EER,
                ]
            )

    def shutdown(self, event):
        # Stop the node when shutdown is called
        rospy.logfatal("Timer expired. Stopping the node...")
        rospy.sleep(0.1)
        rospy.signal_shutdown("Timer signal shutdown")
        # os.system("rosnode kill other_node")


if __name__ == "__main__":
    try:
        time_to_shutdown = 50
        rospy.init_node("guidance", anonymous=True)
        guidance = Guidance()
        rospy.Timer(rospy.Duration(time_to_shutdown), guidance.shutdown, oneshot=True)
        rospy.on_shutdown(guidance.shutdown)
        working_directory = os.getcwd()
        # print("Working directory: ", working_directory)
        # empty the errors file without writing on it
        pkgDir = rospkg.RosPack().get_path("mml_guidance")

        with open(pkgDir + "/data/errors.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            # write the first row as the header with the variable names
            writer.writerow(
                [
                    "X error [m]",
                    "Y error [m]",
                    "Yaw error [rad]",
                    "FOV X error [m]",
                    "FOV Y error [m]",
                    "Partial Entropy",
                    "min Info Gain",
                    "Avg Info Gain",
                    "Max Info gain",
                    "Percent in FOV",
                    "EER time [s]",
                ]
            )
            rate = rospy.Rate(10)  # 10 Hz
            while not rospy.is_shutdown():
                # now = rospy.get_time() - guidance.initial_time
                guidance.particle_filter()
                # print("particle filter time: ", rospy.get_time() - guidance.initial_time - now)
                if guidance.update_msg.data:
                    guidance.goal_position = guidance.information_driven_guidance()

                guidance.pub_desired_state()
                rate.sleep()
    except rospy.ROSInterruptException:
        pass
