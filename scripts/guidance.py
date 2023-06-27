#!/usr/bin/env python3
import csv
import os

import numpy as np
import rospkg
import rospy
import scipy.stats as stats
from geometry_msgs.msg import PointStamped, PoseStamped
from lawnmower import LawnmowerPath
from mag_pf_pkg.msg import Particle, ParticleMean
from nav_msgs.msg import Odometry
from ParticleFilter import ParticleFilter

# from geometry_msgs.msg import Pose
from reef_msgs.msg import DesiredState
from rosflight_msgs.msg import RCRaw
from std_msgs.msg import Bool, Float32, Float32MultiArray


class Guidance:
    def __init__(self):
        self.init_finished = False
        self.is_sim = rospy.get_param("/is_sim", False)
        self.is_viz = rospy.get_param("/is_viz", False)  # true to visualize plots

        self.guidance_mode = (
                "Information"  # 'Information', 'Particles', 'Lawnmower', or 'Estimator'
        )
        self.prediction_method = "NN"  # 'NN', 'Velocity' or 'Unicycle'

        # Initialization of variables
        self.quad_position = np.array([0.0, 0.0])
        self.actual_turtle_pose = np.array([0.0, 0.0, 0.0])
        self.noisy_turtle_pose = np.array([0.0, 0.0, 0.0])
        self.goal_position = np.array([0.0, 0.0])
        self.eer_goal = np.array([0.0, 0.0])
        self.linear_velocity = np.array([0.0, 0.0])
        self.angular_velocity = np.array([0.0])
        deg2rad = lambda deg: np.pi * deg / 180
        # Number of future measurements per sampled particle to consider in EER
        # self.N_m = 1  # not implemented yet
        # Number of sampled particles
        self.N_s = 25 
        # Time steps to propagate in the future for EER
        self.k = 5
        # initiate entropy
        self.Hp_t = 0  # partial entropy
        self.IG_range = np.array([0, 0, 0])
        self.t_EER = 0
        self.position_following = False
        # Measurement Model
        self.height = 1.5
        camera_angle = np.array(
            [deg2rad(69), deg2rad(42)]
        )  # camera angle in radians (horizontal, vertical)
        self.FOV_dims = np.tan(camera_angle) * self.height
        self.FOV = self.construct_FOV(self.quad_position)
        self.initial_time = rospy.get_time()
        self.idg_counter = 0
        ## PARTICLE FILTER  ##
        # number of particles
        self.N = 500 
        self.filter = ParticleFilter(self.N, self.prediction_method)

        # Occlusions
        occ_width = 0.75
        occ_center = [-1.25, -1.05]
        rospy.set_param("/occlusions", [occ_center, occ_width])
        self.occlusions = Occlusions([occ_center], [occ_width])

        # Lawnmower
        lawnmower = LawnmowerPath([0, 0], [3.5, 3.5], 1.5)
        self.path = lawnmower.trajectory()
        self.lawnmower_idx = 0
        self.increment = 1

        # ROS stuff
        rospy.loginfo(
            f"Initializing guidance node with parameter is_sim: {self.is_sim}"
        )
        rospy.loginfo(f"...and parameter is_viz: {self.is_viz}")
        rospy.loginfo(f"Quadcopter in guidance mode: {self.guidance_mode}")
        rospy.loginfo(f"... and in prediction method: {self.prediction_method}")
        self.pose_pub = rospy.Publisher("desired_state", DesiredState, queue_size=1)
        self.err_tracking_pub = rospy.Publisher(
            "err_tracking", PointStamped, queue_size=1
        )

        if self.is_sim:
            self.turtle_odom_sub = rospy.Subscriber(
                "/robot0/odom", Odometry, self.turtle_odom_cb, queue_size=1
            )
            self.quad_odom_sub = rospy.Subscriber(
                "/pose_stamped", PoseStamped, self.quad_odom_cb, queue_size=1
            )
        else:
            self.quad_odom_sub = rospy.Subscriber(
                "/quad_pose_stamped", PoseStamped, self.quad_odom_cb, queue_size=1
            )
            self.turtle_odom_sub = rospy.Subscriber(
                "/turtle_pose_stamped", PoseStamped, self.turtle_odom_cb, queue_size=1
            )
            self.turtle_odom_sub = rospy.Subscriber(
                "/odom", Odometry, self.turtle_hardware_odom_cb, queue_size=1
            )
            self.rc_sub = rospy.Subscriber("rc_raw", RCRaw, self.rc_cb, queue_size=1)

        if self.is_viz:
            self.particle_pub = rospy.Publisher(
                "xyTh_estimate", ParticleMean, queue_size=1
            )
            self.err_estimation_pub = rospy.Publisher(
                "err_estimation", PointStamped, queue_size=1
            )
            self.entropy_pub = rospy.Publisher("entropy", Float32, queue_size=1)
            self.info_gain_pub = rospy.Publisher("info_gain", Float32, queue_size=1)
            self.eer_time_pub = rospy.Publisher("eer_time", Float32, queue_size=1)
            self.n_eff_pub = rospy.Publisher("n_eff_particles", Float32, queue_size=1)
            self.update_pub = rospy.Publisher("is_update", Bool, queue_size=1)
            self.fov_pub = rospy.Publisher("fov_coord", Float32MultiArray, queue_size=1)
            self.des_fov_pub = rospy.Publisher(
                "des_fov_coord", Float32MultiArray, queue_size=1
            )

        rospy.loginfo("Number of particles for the Bayes Filter: %d", self.N)
        rospy.sleep(0.1)
        self.init_finished = True

    def current_entropy(self, event=None):
        now = rospy.get_time() - self.initial_time
        prob = np.nan_to_num(self.filter.weights, copy=True, nan=0)
        prob = prob / np.sum(prob)
        # try choosing next particles, if there is an error, return list from 0 to N_s
        try:
            self.sampled_index = np.random.choice(a=self.N, size=self.N_s, p=prob)
        except:
            print("Bad particle weights when sampling for entropy calc :(\n")
            # particles are basically lost, reinitialize
            self.filter.resample()
            self.sampled_index = np.arange(self.N_s)
        # Entropy of current distribution
        if self.filter.is_update:
            self.in_FOV = 1
            self.Hp_t = self.entropy_particle(
                self.filter.particles[-2, self.sampled_index, :],
                np.copy(self.filter.prev_weights[self.sampled_index]),
                self.filter.particles[-1, self.sampled_index, :],
                np.copy(self.filter.weights[self.sampled_index]),
                self.noisy_turtle_pose,
            )
        else:
            self.in_FOV = 0
            # rospy.logwarn("Current_entropy else statement:")
            self.Hp_t = self.entropy_particle(
                self.filter.particles[-2, self.sampled_index, :],
                np.copy(
                    self.filter.weights[self.sampled_index]
                ),  # current weights are the (t-1) weights because no update
                self.filter.particles[-1, self.sampled_index, :],
            )

        entropy_time = rospy.get_time() - self.initial_time
        # print("Entropy time: ", entropy_time - now)

    def information_driven_guidance(self, event=None):
        """Compute the current entropy and future entropy using particles
        to then compute the expected entropy reduction (EER) over predicted
        measurements. The next action is the one that minimizes the EER.
        Output:
        goal_position: the position of the measurement that resulted in the
        maximum EER
        """
        self.idg_counter += 1
        now = rospy.get_time() - self.initial_time
        # rospy.logwarn("Counter: %d - Elapsed: %f" %(self.idg_counter, now))
        ## Guidance
        future_weight = np.zeros((self.N, self.N_s))
        Hp_k = np.zeros(self.N_s)  # partial entropy
        Ip = np.zeros(self.N_s)  # partial information gain

        future_parts = np.copy(self.filter.particles)
        last_future_time = np.copy(self.filter.last_time)
        for k in range(self.k):
            # future_parts = self.filter.motion_model.predict(future_parts)
            if self.prediction_method == "NN":
                future_parts = self.filter.predict_mml(future_parts)
            elif self.prediction_method == "Unicycle":
                future_parts, last_future_time = self.filter.predict(
                    future_parts,
                    self.filter.weights,
                    last_future_time + 0.3,
                    angular_velocity=self.angular_velocity,
                    linear_velocity=self.linear_velocity,
                )
            elif self.prediction_method == "Unicycle":
                future_parts, last_future_time = self.filter.predict(
                    future_parts,
                    self.filter.weights,
                    last_future_time + 0.3,
                )
        # Future possible measurements
        # TODO: implement N_m sampled measurements
        z_hat = self.filter.add_noise(
            future_parts[-1, self.sampled_index, :2], self.filter.measurement_covariance
        )
        likelihood = self.filter.likelihood(
            z_hat, future_parts[-1, self.sampled_index, :]
        )
        # TODO: implement N_m sampled measurements (double loop)
        for jj in range(self.N_s):
            k_fov = self.construct_FOV(z_hat[jj])
            # checking for measurement outside of fov or in occlusion
            if self.is_in_FOV(z_hat[jj], k_fov) and not self.in_occlusion(z_hat[jj]):
                future_weight[:, jj] = self.filter.update(
                    self.filter.weights, future_parts, z_hat[jj]
                )

                # H (x_{t+k} | \hat{z}_{t+k})
                Hp_k[jj] = self.entropy_particle(
                    future_parts[-2, self.sampled_index, :],
                    np.copy(self.filter.weights[self.sampled_index]),
                    future_parts[-1, self.sampled_index, :],
                    future_weight[self.sampled_index, jj],
                    z_hat[jj],
                )
            else:
                Hp_k[jj] = self.entropy_particle(
                    future_parts[-2, self.sampled_index, :],
                    np.copy(self.filter.weights[self.sampled_index]),
                    future_parts[-1, self.sampled_index, :],
                )

            # Information Gain
            Ip[jj] = self.Hp_t - Hp_k[jj]

        # EER = I.mean() # implemented when N_m is implemented
        EER = likelihood * Ip
        action_index = np.argmax(EER)
        self.IG_range = np.array([np.min(Ip), np.mean(Ip), np.max(Ip)])

        self.t_EER = rospy.get_time() - self.initial_time - now
        #print("EER time: ", self.t_EER)

        # print("possible actions: ", z_hat[:, :2])
        # print("information gain: ", I)
        # print("Chosen action:", z_hat[action_index, :2])
        self.eer_goal = z_hat[action_index][:2]

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
            like_meas = stats.multivariate_normal.pdf(
                x=particles[:, :2],
                mean=y_meas[:2],
                cov=self.filter.measurement_covariance[:2, :2],
            )

            # likelihood of particle p(xt|xt-1)
            part_len, _ = particles.shape
            process_part_like = np.zeros(part_len)

            for ii in range(part_len):
                # maybe kinematics with gaussian
                # maybe get weight wrt to previous state (distance)
                like_particle = stats.multivariate_normal.pdf(
                    x=prev_particles[:, :2],
                    mean=particles[ii, :2],
                    cov=self.filter.process_covariance[:2, :2],
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
                print("\nEntropy term went bad :(")
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
            part_len2, _ = prev_particles.shape
            process_part_like = np.zeros(part_len2)
            for ii in range(part_len2):
                # maybe kinematics with gaussian
                # maybe get weight wrt to previous state (distance)
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

        return np.clip(entropy, -20, 1000)

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

    def lawnmower(self):
        """Return the position of the measurement if there is one,
        else return the next position in the lawnmower path.
        If the rate of get_goal_position changes, the increment 
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
        """Return true if the position measurement is in occlusion zones
        Inputs: pos: position to check - numpy.array of shape (2,)
        Output: true if it is in occlusion - bool
        """
        for rect in range(len(self.occlusions.positions)):
            if (
                pos[0]
                > self.occlusions.positions[rect][0] - self.occlusions.widths[rect] / 2
                and pos[0]
                < self.occlusions.positions[rect][0] + self.occlusions.widths[rect] / 2
            ):
                if (
                    pos[1]
                    > self.occlusions.positions[rect][1]
                    - self.occlusions.widths[rect] / 2
                    and pos[1]
                    < self.occlusions.positions[rect][1]
                    + self.occlusions.widths[rect] / 2
                ):
                    return True

        return False

    def get_goal_position(self, event=None):
        if self.guidance_mode == "Information":
            self.goal_position = self.eer_goal
        elif self.guidance_mode == "Particles":
            self.goal_position = self.filter.weighted_mean
        elif self.guidance_mode == "Lawnmower":
            mower_position = self.lawnmower()
            self.goal_position = mower_position
        elif self.guidance_mode == "Estimator":
            self.goal_position = self.actual_turtle_pose

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
                self.actual_turtle_pose = np.array(
                    [turtle_position[0], turtle_position[1], theta_z]
                )
                self.noisy_turtle_pose[2] = self.actual_turtle_pose[2]
                self.noisy_turtle_pose[:2] = self.filter.add_noise(
                    self.actual_turtle_pose[:2],
                    self.filter.measurement_covariance[:2, :2],
                )
            else:
                self.actual_turtle_pose = np.array(
                    [msg.pose.position.x, msg.pose.position.y]
                )
                # In the current network we do not use the orientation of the turtlebot
                # turtle_orientation = np.array(
                #    [
                #        msg.pose.orientation.x,
                #        msg.pose.orientation.y,
                #        msg.pose.orientation.z,
                #        msg.pose.orientation.w,
                #    ]
                # )
                # _, _, theta_z = self.euler_from_quaternion(turtle_orientation)
                # self.noisy_turtle_pose = np.array(
                #    [msg.pose.position.x, msg.pose.position.y, theta_z]
                # )
                # in hardware we assume the pose is already noisy
                self.noisy_turtle_pose = np.copy(self.actual_turtle_pose)

            if self.is_in_FOV(
                self.noisy_turtle_pose, self.FOV
            ) and not self.in_occlusion(self.noisy_turtle_pose):
                # if not self.in_occlusion(self.noisy_turtle_pose):
                self.filter.is_update = True
            else:
                self.filter.is_update = False  # no update

    def turtle_hardware_odom_cb(self, msg):
        """Here we get the linear and angular velocities
        from the wheel encoders"""
        if self.init_finished:
            self.linear_velocity = np.array(
                [msg.twist.twist.linear.x, msg.twist.twist.linear.y]
            )
            self.angular_velocity = np.array([msg.twist.twist.angular.z])

    def rc_cb(self, msg):
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
            # self.quad_position[1] = -self.quad_position[1] if not self.guidance_mode else self.quad_position[1]
            self.FOV = self.construct_FOV(self.quad_position)

    def pub_desired_state(self, event=None):
        if self.init_finished:
            is_velocity = False  # [not used] might be useful later
            ds = DesiredState()
            # run the quad if sim or the remote controller
            # sends signal of autonomous control
            if self.position_following or self.is_sim:
                if is_velocity:
                    xvel = yvel = 0
                    ds.velocity.x = xvel
                    ds.velocity.y = yvel
                    ds.position_valid = False
                    ds.velocity_valid = True
                else:
                    ds.pose.x = self.goal_position[0]
                    ds.pose.y = -self.goal_position[1]
                    # flip sign of y to transform from NWU to NED
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
            # Given boundary of the lab [[x_min, y_min], [x_max, y_,max]]
            # clip the x and y position to the  space self.filter.AVL_dims
            ds.pose.x = np.clip(
                ds.pose.x, self.filter.AVL_dims[0][0], self.filter.AVL_dims[1][0]
            )
            ds.pose.y = np.clip(
                ds.pose.y, self.filter.AVL_dims[0][1], self.filter.AVL_dims[1][1]
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
                fov_msg.data = fov_matrix.flatten("C")
                self.fov_pub.publish(fov_msg)
                # Desired FOV
                self.des_fov = self.construct_FOV(
                    np.array([ds.pose.x, -ds.pose.y])
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
                fov_msg.data = des_fov_matrix.flatten("C")
                self.des_fov_pub.publish(fov_msg)
                # Is update pub
                update_msg = Bool()
                update_msg.data = self.filter.is_update
                self.update_pub.publish(update_msg)

    def pub_pf(self, event=None):
        # Particle pub
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
        mean_msg.cov = np.diag(self.filter.var).flatten("C")
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
        info_gain_msg.data = self.IG_range[0]
        self.entropy_pub.publish(info_gain_msg)
        # EER time pub
        eer_time_msg = Float32()
        eer_time_msg.data = self.t_EER
        self.eer_time_pub.publish(eer_time_msg)
        # Entropy pub
        entropy_msg = Float32()
        entropy_msg.data = self.Hp_t
        self.entropy_pub.publish(entropy_msg)

    def write_csv(self, event=None):
        if self.is_sim:
            pkgDir = rospkg.RosPack().get_path("mml_guidance")
            with open(pkgDir + "/data/errors.csv", "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        self.filter.weighted_mean[0] - self.actual_turtle_pose[0],
                        self.filter.weighted_mean[1] - self.actual_turtle_pose[1],
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

    def guidance_pf(self, event=None):
        if self.prediction_method == "Unicycle":
            self.filter.pf_loop(
                self.noisy_turtle_pose, self.angular_velocity, self.linear_velocity
            )
        else:
            self.filter.pf_loop(self.noisy_turtle_pose)

    def shutdown(self, event=None):
        # Stop the node when shutdown is called
        rospy.logfatal("Timer expired. Stopping the node...")
        rospy.sleep(0.1)
        rospy.signal_shutdown("Timer signal shutdown")
        # os.system("rosnode kill other_node")


class Occlusions:
    def __init__(self, positions, widths):
        """All occlusions are defined as squares with
        some position (x and y) and some width. The attributes are
        the arrays of positions and widths of the occlusions."""
        self.positions = positions
        self.widths = widths


if __name__ == "__main__":
    rospy.init_node("guidance", anonymous=True)
    guidance = Guidance()

    time_to_shutdown = 2000
    rospy.Timer(rospy.Duration(time_to_shutdown), guidance.shutdown, oneshot=True)
    rospy.on_shutdown(guidance.shutdown)
    working_directory = os.getcwd()
    # print("Working directory: ", working_directory)
    # empty the errors file without writing on it
    pkgDir = rospkg.RosPack().get_path("mml_guidance")

    if guidance.is_sim:
        with open(pkgDir + "/data/errors.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            # write the first row as the header with the variable names
            writer.writerow(
                [
                    "X error [m]",
                    "Y error [m]",
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

    rospy.Timer(rospy.Duration(1.0 / 3.0), guidance.guidance_pf)
    rospy.Timer(rospy.Duration(1.0 / 3.0), guidance.current_entropy)
    if guidance.guidance_mode == "Information":
        rospy.Timer(rospy.Duration(1.0 / 1.7), guidance.information_driven_guidance)

    # Publish topics
    if guidance.is_viz:
        rospy.Timer(rospy.Duration(1.0 / 3.0), guidance.pub_pf)
    rospy.Timer(rospy.Duration(1.0 / 1.65), guidance.get_goal_position)
    rospy.Timer(rospy.Duration(1.0 / 3.0), guidance.pub_desired_state)
    rospy.Timer(rospy.Duration(1.0 / 3.0), guidance.write_csv)

    rospy.spin()
