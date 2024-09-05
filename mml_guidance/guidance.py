#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
import time
import scipy.stats as stats
import os
from geometry_msgs.msg import PointStamped, PoseStamped
from .lawnmower import LawnmowerPath
from mml_guidance_msgs.msg import Particle, ParticleMean, ParticleArray
from nav_msgs.msg import Odometry
from .ParticleFilter import ParticleFilter
from .kf import KalmanFilter

# from geometry_msgs.msg import Pose
#from rosflight_msgs.msg import RCRaw
from std_msgs.msg import Bool, Float32, Float32MultiArray


class Guidance(Node):
    def __init__(self):
        super().__init__("guidance")
        self.init_finished = False
        self.is_sim = (
            self.declare_parameter("is_sim", False).get_parameter_value().bool_value
        )
        self.is_viz = (
            self.declare_parameter("is_viz", False).get_parameter_value().bool_value
        )  # true to visualize plots

        self.guidance_mode = (
            "Estimator"  # 'Information', 'Particles', 'Lawnmower', or 'Estimator'
        )
        self.prediction_method = "Transformer"  # 'KF', 'NN', 'Velocity' or 'Unicycle'

        # Initialization of robot variables
        self.quad_position = np.array([0.0, 0.0])
        self.actual_turtle_pose = np.array([0.0, 0.0, 0.0])
        self.noisy_turtle_pose = np.array([0.0, 0.0, 0.0])
        self.goal_position = np.array([0.0, 0.0])
        self.linear_velocity = np.array([0.0, 0.0])
        self.angular_velocity = np.array([0.0])
        deg2rad = lambda deg: np.pi * deg / 180
        self.initial_time = self.get_clock().now().seconds_nanoseconds()[0]

        ## PARTICLE FILTER  ##
        self.N = 500  # Number of particles
        self.filter = ParticleFilter(self.N, self.prediction_method, self.is_sim)
        # Camera Model
        self.height = 1.2  # height of the quadcopter in meters
        camera_angle = 3*np.array(
            [deg2rad(35), deg2rad(35)]
        )  # camera angle in radians (horizontal, vertical)
        self.FOV_dims = np.tan(camera_angle) * self.height
        self.FOV = self.construct_FOV(self.quad_position)

        ## INFO-DRIVEN GUIDANCE ##
        # Number of future measurements per sampled particle to consider in EER
        # self.N_m = 1  # not implemented yet
        self.N_s = 25  # Number of sampled particles
        self.declare_parameter("num_sampled_particles", self.N_s)
        self.K = 4  # Time steps to propagate in the future for EER
        self.declare_parameter("predict_window", self.K)
        self.Hp_t = 1.0  # partial entropy
        self.prev_Hp = np.ones((5, 1))
        self.EER_range = np.array([0, 0, 0])
        self.t_EER = 0.0
        self.eer_particle = 0  # initialize future particle to follow
        self.sampled_index = np.arange(self.N)
        self.sampled_particles = self.filter.particles[:, : self.N_s, :]
        self.sampled_weights = np.ones(self.N_s) / self.N_s
        self.position_following = True

        # Occlusions
        occ_widths = [1, 1]
        occ_centers = [-1.25, -0.6, 0.35, 0.2]  # flattened [x1, y1, x2, y2] to send as parameters
        self.declare_parameter("occlusion_widths", occ_widths)
        self.declare_parameter("occlusion_centers", occ_centers)
        occ_centers = [occ_centers[:2], occ_centers[2:]]

        self.occlusions = [
            Occlusions(occ_centers[i], occ_widths[i]) for i in range(len(occ_widths))
        ]

        if self.guidance_mode == "Lawnmower":
            # Lawnmower Method
            lawnmower = LawnmowerPath([0, 0], [3.5, 3.5], 1.5, is_sim=self.is_sim)
            self.path = lawnmower.trajectory()
            self.lawnmower_idx = 0
            self.increment = 1
        if self.prediction_method == "KF":
            # Kalman Filter
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            self.kf = KalmanFilter(
                R=self.filter.measurement_covariance,
                Q=self.filter.process_covariance,
                H=H,
            )

        # ROS stuff
        self.get_logger().info(
            f"Initializing guidance node with parameter is_sim: {self.is_sim}"
        )
        self.get_logger().info(f"...and parameter is_viz: {self.is_viz}")
        self.get_logger().info(f"Quadcopter in guidance mode: {self.guidance_mode}")
        self.get_logger().info(
            f"... and in prediction method: {self.prediction_method}"
        )

        self.pose_pub = self.create_publisher(
            msg_type=PoseStamped, topic="/mavros/setpoint_position/local", qos_profile=1
        )
        self.err_tracking_pub = self.create_publisher(PointStamped, "err_tracking", 1)

        if self.is_sim:
            self.turtle_odom_sub = self.create_subscription(
                Odometry, "/robot0/odom", self.turtle_odom_cb, 1
            )
            self.quad_odom_sub = self.create_subscription(
                PoseStamped, "/pose_stamped", self.quad_odom_cb, 1
            )
        else:
            self.quad_odom_sub = self.create_subscription(
                PoseStamped, "/quad_pose_stamped", self.quad_odom_cb, 1
            )
            self.turtle_odom_sub = self.create_subscription(
                PoseStamped, "/turtle_pose_stamped", self.turtle_odom_cb, 1
            )
            self.turtle_odom_sub = self.create_subscription(
                Odometry, "/odom", self.turtle_hardware_odom_cb, 1
            )
            #self.rc_sub = self.create_subscription(RCRaw, "rc_raw", self.rc_cb, 1)

        if self.is_viz:
            self.particle_pub = self.create_publisher(ParticleMean, "xyTh_estimate", 1)
            self.particle_pred_pub = self.create_publisher(
                ParticleArray, "xyTh_predictions", 1
            )
            self.sampled_index_pub = self.create_publisher(
                Float32MultiArray, "sampled_index", 1
            )
            self.err_estimation_pub = self.create_publisher(
                PointStamped, "err_estimation", 1
            )
            self.meas_pub = self.create_publisher(PointStamped, "noisy_measurement", 1)
            self.entropy_pub = self.create_publisher(Float32, "entropy", 1)
            self.info_gain_pub = self.create_publisher(Float32, "info_gain", 1)
            self.eer_time_pub = self.create_publisher(Float32, "eer_time", 1)
            self.det_cov_pub = self.create_publisher(
                Float32, "xyTh_estimate_cov_det", 1
            )
            self.n_eff_pub = self.create_publisher(Float32, "n_eff_particles", 1)
            self.update_pub = self.create_publisher(Bool, "is_update", 1)
            self.occ_pub = self.create_publisher(Bool, "is_occlusion", 1)
            self.fov_pub = self.create_publisher(Float32MultiArray, "fov_coord", 1)
            self.des_fov_pub = self.create_publisher(
                Float32MultiArray, "des_fov_coord", 1
            )

        self.get_logger().info("Number of particles for the Bayes Filter: %d" % self.N)
        self.init_finished = True

    def current_entropy(self, event=None) -> None:
        """Compute the entropy of the current distribution of particles
        Equation changes if there is a measurement available. See paper for reference
        We sample the particle distribution to reduce computation time.
        Output: Hp_t (float) - entropy of the current distribution"""
        t = self.get_clock().now().seconds_nanoseconds()[0] - self.initial_time

        prev_Hp = np.copy(self.Hp_t)

        self.sampled_index = np.random.choice(a=self.N, size=self.N_s)
        self.sampled_particles = np.copy(
            self.filter.particles[:, self.sampled_index, :]
        )
        self.sampled_weights = np.copy(self.filter.weights[self.sampled_index])
        # Normalize weights
        self.sampled_weights = (
            self.sampled_weights / np.sum(self.sampled_weights)
            if np.sum(self.sampled_weights) > 0
            else self.sampled_weights
        )
        sampled_prev_weights = np.copy(self.filter.prev_weights[self.sampled_index])
        if self.filter.is_update:
            Hp_t = self.entropy_particle(
                self.sampled_particles[-2],
                np.copy(sampled_prev_weights),
                self.sampled_particles[-1],
                np.copy(self.sampled_weights),
                self.noisy_turtle_pose,
            )
        else:
            Hp_t = self.entropy_particle(
                self.sampled_particles[-2],
                np.copy(
                    self.sampled_weights
                ),  # current weights are the (t-1) weights because no update
                self.sampled_particles[-1],
            )

        Hp_t = (
            prev_Hp if not np.isfinite(Hp_t) else Hp_t
        )  # if really bad value, keep the previous one
        if not np.array_equal(self.prev_Hp, np.ones((5, 1))):
            self.Hp_t = self.reject_spikes(
                self.prev_Hp,
                Hp_t,
                threshold_factor=3,
            )

        # update entropy history
        self.prev_Hp = np.roll(self.prev_Hp, -1, axis=0)
        self.prev_Hp[-1, :2] = Hp_t

        entropy_time = (
            self.get_clock().now().seconds_nanoseconds()[0] - t - self.initial_time
        )
        # print("Entropy time: ", entropy_time)

    def information_driven_guidance(self, event=None):
        """Compute the current entropy and future entropy using particles
        to then compute the expected entropy reduction (EER) over predicted
        measurements. The next action is the  position of the particle
        that maximizes the EER.
        Output:
        eer_particle: the index of the particle that maximizes the EER which
        we propagate to choose the goal position
        """
        now = self.get_clock().now().seconds_nanoseconds()[0] - self.initial_time
        # Initialize variables
        future_weight = np.zeros((self.N_s, self.N_s))
        Hp_k = np.zeros(self.N_s)  # partial entropy
        EER = np.zeros(self.N_s)  # Expected Entropy Reduction
        future_parts = np.copy(self.sampled_particles)
        last_future_time = np.copy(self.filter.last_time)

        pred_msg = ParticleArray()
        for k in range(self.K):  # propagate k steps in the future
            if (
                self.prediction_method == "NN"
                or self.prediction_method == "Transformer"
            ):
                future_parts = self.filter.predict_mml(np.copy(future_parts))
            elif self.prediction_method == "Unicycle":
                future_parts, last_future_time = self.filter.predict(
                    future_parts,
                    last_future_time + 0.3,
                    angular_velocity=self.angular_velocity,
                    linear_velocity=self.linear_velocity,
                )
            elif self.prediction_method == "Velocity":
                future_parts, last_future_time = self.filter.predict(
                    future_parts,
                    last_future_time + 0.3,
                )

            if self.is_viz:
                assert future_parts.shape == (
                    self.filter.N_th,
                    self.N_s,
                    self.filter.Nx,
                )
                mean_msg = ParticleMean()
                for ii in range(self.N_s):
                    particle_msg = Particle()
                    particle_msg.x = future_parts[-1, ii, 0]
                    particle_msg.y = future_parts[-1, ii, 1]
                    particle_msg.weight = self.filter.weights[ii]
                    mean_msg.all_particle.append(particle_msg)
                pred_msg.particle_array.append(mean_msg)

        if self.is_viz:
            if len(pred_msg.particle_array) > self.K:
                pred_msg.particle_array.pop(0)  # eliminate the oldest particle
            self.particle_pred_pub.publish(pred_msg)  # publish the history
            # publish the sampled index array
            self.sampled_index_pub.publish(data=self.sampled_index)

        # Future possible measurements
        # TODO: implement N_m sampled measurements (assumption: N_m = 1)
        z_hat = self.filter.add_noise(
            future_parts[-1, :, : self.filter.measurement_covariance.shape[0]],
            self.filter.measurement_covariance,
        )
        # p(z_{k+K} | x_{k+K})
        likelihood = self.filter.likelihood(future_parts[-1], z_hat)
        # TODO: implement N_m sampled measurements (double loop)
        for jj in range(self.N_s):
            k_fov = self.construct_FOV(z_hat[jj])
            # checking for measurement outside of fov or in occlusion
            if self.is_in_FOV(z_hat[jj], k_fov) and not self.in_occlusion(z_hat[jj]):
                future_weight[:, jj] = self.filter.update(
                    self.sampled_weights, future_parts, z_hat[jj]
                )

                # H (x_{k+K} | \hat{z}_{k+K})
                Hp_k[jj] = self.entropy_particle(
                    future_parts[-2],
                    np.copy(self.sampled_weights),
                    future_parts[-1],
                    future_weight[:, jj],
                    z_hat[jj],
                )
            else:
                Hp_k[jj] = self.entropy_particle(
                    future_parts[-2],
                    np.copy(self.sampled_weights),
                    future_parts[-1],
                )

            # Information Gain
            EER[jj] = self.Hp_t - Hp_k[jj] * likelihood[jj]

        # EER = I.mean() # implemented when N_m is implemented
        action_index = np.argmax(EER)
        self.EER_range = np.array([np.min(EER), np.mean(EER), np.max(EER)])
        # print("EER: ", EER)

        self.t_EER = (
            self.get_clock().now().seconds_nanoseconds()[0] - self.initial_time - now
        )
        # print("EER time: ", self.t_EER)

        self.eer_particle = self.sampled_index[action_index]

    def entropy_particle(
        self,
        prev_particles,
        prev_wgts,
        particles,
        wgts=np.array([]),
        z_meas=np.array([]),
    ) -> np.int64:
        """Compute the entropy of the particle distribution based on the equation in the
        paper: Y. Boers, H. Driessen, A. Bagchi, and P. Mandal, 'Particle filter based entropy'
        There are two computations, one for the case where the measurement is inside the fov
        and there is an update step before, and one where the measurement is outside the fov.
        Output:
            entropy (numpy.int64)
        """
        if wgts.size > 0:
            # likelihod of measurement p(zt|xt)
            # (how likely is each of the particles in the gaussian of the measurement)
            like_meas = stats.multivariate_normal.pdf(
                x=particles[:, :2],
                mean=z_meas[:2],
                cov=self.filter.measurement_covariance[:2, :2],
            )

            # likelihood of particle p(xt|xt-1)
            part_len, _ = particles.shape
            process_part_like = np.zeros(part_len)

            for ii in range(part_len):
                like_particle = stats.multivariate_normal.pdf(
                    x=prev_particles[:, :2],
                    mean=particles[ii, :2],
                    cov=self.filter.process_covariance[:2, :2],
                )
                process_part_like[ii] = np.sum(like_particle * prev_wgts[ii])

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
            # first_term = first_term if np.isfinite(first_term) else 0.0
            # second_term = np.nansum(np.log(prev_wgts)*weights)
            # third_term = np.nansum(weights*np.log(like_meas))
            # fourth_term = np.nansum(weights*np.log(process_part_like))

            entropy = (
                first_term
                - np.nansum(np.log(prev_wgts) * wgts)
                - np.nansum(wgts * np.log(like_meas))
                - np.nansum(wgts * np.log(process_part_like))
            )

            if np.abs(entropy) > 30:  # debugging
                print("\nEntropy term went bad :(")
                print("first term: ", np.log(np.nansum(like_meas * prev_wgts)))
                print("second term: ", np.nansum(np.log(prev_wgts) * wgts))
                print("third term: ", np.nansum(wgts * np.log(like_meas)))
                print("fourth term: ", np.nansum(wgts * np.log(process_part_like)))
                # print('like_meas min: ', like_meas.min())
                # print('like_meas max: ', like_meas.max())
                # print('like_meas mean: ', like_meas.mean())
                # print('like_meas std: ', like_meas.std())

                if np.isinf(np.log(np.nansum(like_meas * prev_wgts))):
                    self.get_logger().warn(
                        "first term of entropy is -inf. Likelihood is very small"
                    )
        else:
            # likelihood of particle p(xt|xt-1)
            part_len2, _ = prev_particles.shape
            process_part_like = np.zeros(part_len2)
            for ii in range(part_len2):
                like_particle = stats.multivariate_normal.pdf(
                    x=prev_particles[:, :2],
                    mean=particles[ii, :2],
                    cov=self.filter.process_covariance[:2, :2],
                )
                process_part_like[ii] = np.sum(like_particle * prev_wgts)

            # Numerical stability
            cutoff = 1e-4
            process_part_like[process_part_like < cutoff] = np.nan
            prev_wgts[prev_wgts < cutoff] = np.nan

            entropy = -np.nansum(prev_wgts * np.log(process_part_like))

        # return np.clip(entropy, -100, 1000)
        return entropy

    @staticmethod
    def reject_spikes(prev_values, current_value, threshold_factor=3):
        """
        Rejects spikes in time-series data of previous values.

        Parameters:
            prev_values (numpy.ndarray): An array containing previous data points in the time series.

            current_value (float or int): The current data point in the time series.
            threshold_factor (float, optional): A factor to scale the standard deviation
                to determine the threshold for rejecting spikes. Defaults to 3.

        Returns:
            float or int: The filtered value. If the current value is considered a spike,
                it is replaced by the median of the values in the moving window and current_value;
                otherwise, the current value is returned unchanged.

        Raises:
            None
        """
        # join array of values prev_values and current_value
        values = np.append(prev_values, current_value)
        # print("values: ", values)
        median = np.median(values)
        std = np.std(values)
        threshold = median + threshold_factor * std
        if abs(current_value - median) > threshold:
            return median
        else:
            return current_value

    #@staticmethod
    def is_in_FOV(self, sparticles, fov):
        """Check if the particles are in the FOV of the camera.
        Input: Array of particles, FOV
        Output: Array of booleans indicating if each particle is in the FOV
        """
        if sparticles.ndim == 1:
            return np.all(
                [
                    sparticles[0] > fov[0],
                    sparticles[0] < fov[1],
                    sparticles[1] > fov[2],
                    sparticles[1] < fov[3],
                ]
            )
        else:
            return np.logical_and.reduce(
                [
                    sparticles[:, 0] > fov[0],
                    sparticles[:, 0] < fov[1],
                    sparticles[:, 1] > fov[2],
                    sparticles[:, 1] < fov[3],
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

    def lawnmower(self) -> np.ndarray:
        """Return the position of the measurement if there is one,
        else return the next position in the lawnmower path.
        If the rate of pub_desired_state changes, the increment
        variable needs to change
        """
        if self.filter.is_update:
            return self.noisy_turtle_pose[:2]
        else:
            if self.lawnmower_idx <= 0:
                self.increment = 1
            if self.lawnmower_idx >= self.path.shape[0] - 1:
                self.increment = -1
            self.lawnmower_idx += self.increment
            return self.path[int(np.floor(self.lawnmower_idx)), :2]

    def in_occlusion(self, pos):
        """Return true if the position measurement is in occlusion zones for a single
        position but if it is an array of positions, return an array of booleans for
        particles inside occlusion
        Inputs: pos: position to check - numpy.array of shape (2,)
        """
        if pos.ndim == 1:
            in_occ = False
            for occlusion in self.occlusions:
                if np.all(
                    [
                        pos[0] > occlusion.positions[0] - occlusion.widths / 2,
                        pos[0] < occlusion.positions[0] + occlusion.widths / 2,
                        pos[1] > occlusion.positions[1] - occlusion.widths / 2,
                        pos[1] < occlusion.positions[1] + occlusion.widths / 2,
                    ]
                ):
                    in_occ = True
            return in_occ
        else:
            # array of false values of size pos
            prev_check = np.zeros(pos.shape[0], dtype=bool)
            for occlusion in self.occlusions:
                prev_check = np.logical_or(
                    prev_check,
                    np.logical_and.reduce(
                        [
                            pos[:, 0] > occlusion.positions[0] - occlusion.widths / 2,
                            pos[:, 0] < occlusion.positions[0] + occlusion.widths / 2,
                            pos[:, 1] > occlusion.positions[1] - occlusion.widths / 2,
                            pos[:, 1] < occlusion.positions[1] + occlusion.widths / 2,
                        ]
                    ),
                )
            return prev_check

    def get_goal_position(self, event=None):
        """Get the goal position based on the guidance mode.
        Output: goal_position (numpy.array of shape (2,))"""
        if self.guidance_mode == "Information":
            future_part = np.copy(
                self.filter.particles[:, self.eer_particle, :]
            ).reshape((self.filter.N_th, 1, self.filter.Nx))
            last_future_time = np.copy(self.filter.last_time)
            for k in range(self.K):
                if (
                    self.prediction_method == "NN"
                    or self.prediction_method == "Transformer"
                ):
                    future_part = self.filter.predict_mml(future_part)
                if self.prediction_method == "Unicycle":
                    future_part, last_future_time = self.filter.predict(
                        future_part,
                        last_future_time + 0.3,
                        angular_velocity=self.angular_velocity,
                        linear_velocity=self.linear_velocity,
                    )
                elif self.prediction_method == "Velocity":
                    future_part, last_future_time = self.filter.predict(
                        future_part,
                        last_future_time + 0.3,
                    )
            self.goal_position = future_part[-1][0]
        elif self.guidance_mode == "Particles":
            self.goal_position = self.filter.weighted_mean
        elif self.guidance_mode == "Lawnmower":
            mower_position = self.lawnmower()
            self.goal_position = mower_position
        elif self.guidance_mode == "Estimator":
            self.goal_position = self.actual_turtle_pose

    @staticmethod
    def euler_from_quaternion(q):
        """Convert a quaternion into euler angles (roll, pitch, yaw)
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
        """Callback for the turtlebot odometry. Here we get the position and orientation
        of the target"""
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
                self.actual_turtle_pose = np.array(
                    [turtle_position[0], turtle_position[1], theta_z]
                )
                self.noisy_turtle_pose[2] = self.actual_turtle_pose[2]
                self.noisy_turtle_pose[:2] = self.filter.add_noise(
                    self.actual_turtle_pose[:2],
                    self.filter.measurement_covariance[:2, :2],
                )
            else:
                turtle_orientation = np.array(
                   [
                       msg.pose.orientation.x,
                       msg.pose.orientation.y,
                       msg.pose.orientation.z,
                       msg.pose.orientation.w,
                   ]
                )
                _, _, theta_z = self.euler_from_quaternion(turtle_orientation)
                self.actual_turtle_pose = np.array(
                   [msg.pose.position.x, msg.pose.position.y, theta_z]
                )

                # in hardware we assume the pose is already noisy
                self.noisy_turtle_pose = np.copy(self.actual_turtle_pose)

            # Only update if the turtle is not in occlusion and in the FOV
            if self.in_occlusion(self.actual_turtle_pose[:2]):
                self.filter.is_occlusion = True
                self.filter.is_update = False  # no update
            else:
                self.filter.is_occlusion = False
                self.get_logger().info(f"in fov: {self.is_in_FOV(self.actual_turtle_pose[:2], self.FOV)}")
                self.get_logger().info(f"fov: {self.FOV}")
                self.get_logger().info(f"tp: {self.actual_turtle_pose[:2]}")
                if self.is_in_FOV(self.actual_turtle_pose[:2], self.FOV):
                    # we get measurements if the turtle is in the FOV and not in occlusion
                    self.filter.is_update = True
                else:
                    self.filter.is_update = False
            #self.get_logger().info(f"is occ: {self.filter.is_occlusion}")
            #self.get_logger().info(f"update: {self.filter.is_update}")

    def turtle_hardware_odom_cb(self, msg):
        """Callback where we get the linear and angular velocities
        from the physical turtlebot wheel encoders"""
        if self.init_finished:
            self.linear_velocity = np.array(
                [msg.twist.twist.linear.x, msg.twist.twist.linear.y]
            )
            self.angular_velocity = np.array([msg.twist.twist.angular.z])

    def rc_cb(self, msg):
        """Reads the autonomous mode switch from the RC controller"""
        if msg.values[6] > 500:
            self.position_following = True
            # print("rc message > 500, AUTONOMOUS MODE")
        else:
            self.position_following = False
            # print("no rc message > 500, MANUAL MODE: turn it on with the rc controller")

    def quad_odom_cb(self, msg):
        if self.init_finished:
            if self.is_sim:
                self.quad_position = np.array(
                    [msg.pose.position.x, -msg.pose.position.y]
                )
            else:
                self.quad_position = np.array(
                    [msg.pose.position.x, msg.pose.position.y]
                )  # -y to transform NWU to NED
            self.FOV = self.construct_FOV(self.quad_position)

    def pub_desired_state(self, event=None):
        """Publishes messages related to desired state"""
        if self.init_finished:
            ds = PoseStamped()
            # run the quad if sim or the remote controller
            # sends signal of autonomous control
            if self.position_following or self.is_sim:
                ds.pose.position.x = self.goal_position[0]
                ds.pose.position.y = self.goal_position[1]
                #ds.pose.orientation.z = 1.571  # 90 degrees
            else:
                ds.pose.position.x = 0
                ds.pose.position.y = 0
                #ds.pose.orientation.z = 1.571
            ds.pose.position.z = self.height
            # Given boundary of the lab flight space [[x_min, y_min], [x_max, y_,max]]
            # clip the x and y position to the space self.filter.AVL_dims
            ds.pose.position.x = np.clip(
                ds.pose.position.x, self.filter.AVL_dims[0][0], self.filter.AVL_dims[1][0]
            )
            ds.pose.position.y = np.clip(
                ds.pose.position.y, self.filter.AVL_dims[1][1], self.filter.AVL_dims[0][1]
            )  
            self.pose_pub.publish(ds)
            # tracking err pub
            self.FOV_err = self.quad_position - self.actual_turtle_pose[:2]
            err_tracking_msg = PointStamped()
            err_tracking_msg.point.x = self.FOV_err[0]
            err_tracking_msg.point.y = self.FOV_err[1]
            self.err_tracking_pub.publish(err_tracking_msg)
            if self.is_viz:
                # FOV pub
                fov_msg = Float32MultiArray()
                fov_matrix = np.array(
                    [
                        [self.FOV[0], self.FOV[2]],
                        [self.FOV[0], self.FOV[3]],
                        [self.FOV[1], self.FOV[3]],
                        [self.FOV[1], self.FOV[2]],
                        [self.FOV[0], self.FOV[2]],
                    ]
                )
                fov_msg.data = fov_matrix.flatten("C").tolist()
                self.fov_pub.publish(fov_msg)
                # Desired FOV
                self.des_fov = self.construct_FOV(
                    np.array([ds.pose.position.x, -ds.pose.position.y])
                )  # from turtle frame to quad frame
                des_fov_matrix = np.array(
                    [
                        [self.des_fov[0], self.des_fov[2]],
                        [self.des_fov[0], self.des_fov[3]],
                        [self.des_fov[1], self.des_fov[3]],
                        [self.des_fov[1], self.des_fov[2]],
                        [self.des_fov[0], self.des_fov[2]],
                    ]
                )
                fov_msg.data = des_fov_matrix.flatten("C").tolist()
                self.des_fov_pub.publish(fov_msg)
                # Is update pub
                update_msg = Bool()
                update_msg.data = self.filter.is_update
                self.update_pub.publish(update_msg)
                # Measurement pub
                if self.filter.is_update:
                    meas_msg = PointStamped()
                    meas_msg.point.x = self.noisy_turtle_pose[0]
                    meas_msg.point.y = self.noisy_turtle_pose[1]
                    self.meas_pub.publish(meas_msg)
                # Is occlusion pub
                occ_msg = Bool()
                occ_msg.data = self.filter.is_occlusion
                self.occ_pub.publish(occ_msg)

    def pub_pf(self, event=None):
        """Publishes the particles and the mean of the particle filter"""
        if self.init_finished:
            mean_msg = ParticleMean()
            mean_msg.mean.x = self.filter.weighted_mean[0]
            mean_msg.mean.y = self.filter.weighted_mean[1]
            mean_msg.mean.yaw = np.linalg.norm(self.filter.weighted_mean[2:4])
            for ii in range(self.N):
                particle_msg = Particle()
                particle_msg.x = self.filter.particles[-1, ii, 0]
                particle_msg.y = self.filter.particles[-1, ii, 1]
                particle_msg.yaw = np.linalg.norm(self.filter.particles[-1, ii, 2:4])
                particle_msg.weight = self.filter.weights[ii]
                mean_msg.all_particle.append(particle_msg)
            mean_msg.cov = np.diag(self.filter.var).flatten("C").tolist()
            self.particle_pub.publish(mean_msg)
            # Error pub
            err_msg = PointStamped()
            err_msg.point.x = self.filter.weighted_mean[0] - self.actual_turtle_pose[0]
            err_msg.point.y = self.filter.weighted_mean[1] - self.actual_turtle_pose[1]
            self.err_estimation_pub.publish(err_msg)
            # Number of effective particles pub
            neff_msg = Float32()
            neff_msg.data = self.filter.neff
            self.n_eff_pub.publish(neff_msg)
            # Info gain pub
            info_gain_msg = Float32()
            info_gain_msg.data = float(self.EER_range[0])
            self.entropy_pub.publish(info_gain_msg)
            # EER time pub
            eer_time_msg = Float32()
            eer_time_msg.data = self.t_EER
            self.eer_time_pub.publish(eer_time_msg)
            # Entropy pub
            entropy_msg = Float32()
            entropy_msg.data = float(self.Hp_t)
            self.entropy_pub.publish(entropy_msg)
            if self.prediction_method == "KF":
                # Determinant of the covariance of the estimate
                det_msg = Float32()
                det_msg.data = np.linalg.det(self.kf.P[:2, :2])
                # print("det: ", det_msg.data)
                self.det_cov_pub.publish(det_msg)

    def guidance_pf(self, event=None):
        """Runs the particle (or Kalman) filter loop based on the estimation method"""

        # Select to resample all particles if there is a measurement or select
        # the particles in FOV and not in occlusion if there is no measurement (negative information)
        if not self.filter.is_update:
            self.filter.resample_index = np.where(
                np.logical_and(
                    self.is_in_FOV(self.filter.particles[-1], self.FOV),
                    ~self.in_occlusion(self.filter.particles[-1, :, :2]),
                )
            )[0]
            # set weights of samples close to zero
            self.filter.weights[self.filter.resample_index] = 1e-10
            # normalize weights
            self.filter.weights = self.filter.weights / np.sum(self.filter.weights)
        else:
            self.filter.resample_index = np.arange(self.N)

        # Run the particle filter loop
        if self.prediction_method == "Unicycle":
            self.filter.pf_loop(
                self.noisy_turtle_pose, self.angular_velocity, self.linear_velocity
            )
        elif (
            self.prediction_method == "Velocity"
            or self.prediction_method == "NN"
            or self.prediction_method == "Transformer"
        ):
            self.filter.pf_loop(self.noisy_turtle_pose)
        elif self.prediction_method == "KF":
            self.kf.X[:2] = (
                self.actual_turtle_pose[:2] if not self.init_finished else self.kf.X[:2]
            )
            self.kf.predict(dt=0.333)
            self.kf.update(
                self.noisy_turtle_pose[:2]
            ) if self.filter.is_update else None
            self.filter.weighted_mean = np.array([self.kf.X[0], self.kf.X[1]])

    def shutdown(self):
        # Stop the node when shutdown is called
        self.get_logger().fatal(
            "Timer expired or user terminated. Stopping the node..."
        )
        time.sleep(0.1)
        # Use ROS2 service to kill nodes
        os.system("ros2 lifecycle set /drone_guidance shutdown")
        os.system("ros2 lifecycle set /mml_pf_visualization shutdown")
        self.shutdown_timer.cancel()  # Cancel the timer after execution

    def schedule_shutdown(self, time_to_shutdown):
        # Create the timer and store the reference to cancel it later
        self.shutdown_timer = self.create_timer(time_to_shutdown, self.shutdown)

class Occlusions:
    def __init__(self, positions, widths):
        """All occlusions are defined as squares with
        some position (x and y) and some width. The attributes are
        the arrays of positions and widths of the occlusions."""
        self.positions = positions
        self.widths = widths


def main(args=None):
    rclpy.init(args=args)
    guidance = Guidance()

    time_to_shutdown = 1000  # in seconds
    guidance.schedule_shutdown(time_to_shutdown=time_to_shutdown)

    # Running functions at a certain rate
    guidance.create_timer(1.0 / 3.0, guidance.guidance_pf)
    guidance.create_timer(1.0 / 3.0, guidance.current_entropy)

    if guidance.guidance_mode == "Information":
        guidance.create_timer(1.0 / 2.5, guidance.information_driven_guidance)

    # Publish topics
    guidance.create_timer(1.0 / 3.0, guidance.get_goal_position)

    if guidance.is_viz:
        guidance.create_timer(1.0 / 3.0, guidance.pub_pf)

    guidance.create_timer(1.0 / 3.0, guidance.pub_desired_state)

    try:
        rclpy.spin(guidance)
    except KeyboardInterrupt:
        guidance.shutdown()
    finally:
        guidance.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
