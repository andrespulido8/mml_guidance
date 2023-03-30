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

        self.guidance_mode = "Particles"  # 'Information', 'Particles' or 'Lawnmower'

        # Initialization of variables
        self.quad_position = np.array([0, 0])
        self.actual_turtle_pose = np.array([0, 0, 0])
        self.noisy_turtle_pose = np.array([0, 0, 0])
        self.goal_position = np.array([0, 0])
        self.linear_velocity = np.array([0, 0])
        self.angular_velocity = np.array([0])
        deg2rad = lambda deg: np.pi * deg / 180
        # Number of future measurements per sampled particle to consider in EER
        self.N_m = 1
        # Number of sampled particles
        self.N_s = 50
        # Time steps to propagate in the future for EER
        self.k = 1
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
        self.N = 1000
        self.filter = ParticleFilter(self.N)

        # Occlusions
        occ_width = 0.5
        occ_center = [-1.25, -1.05]
        rospy.set_param("/occlusions", [occ_center, occ_width])
        self.occlusions = Occlusions([occ_center], [occ_width])

        # Lawnmower
        lawnmower = LawnmowerPath([0, 0], [2, 2], 1.5)
        self.path = lawnmower.trajectory()
        self.lawnmower_idx = 0
        self.increment = 1

        # ROS stuff
        rospy.loginfo(
            f"Initializing guidance node with parameter is_sim: {self.is_sim}"
        )
        rospy.loginfo(
            f"...and parameter is_viz: {self.is_viz}"
        )
        rospy.loginfo(f"Quadcopter in mode: {self.guidance_mode}")
        self.pose_pub = rospy.Publisher("desired_state", DesiredState, queue_size=1)

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
            # Particle filter ROS stuff
            self.particle_pub = rospy.Publisher(
                "xyTh_estimate", ParticleMean, queue_size=1
            )
            self.err_fov_pub = rospy.Publisher(
                "err_fov", PointStamped, queue_size=1
            )
            self.err_estimation_pub = rospy.Publisher(
                "err_estimation", PointStamped, queue_size=1
            )
            self.entropy_pub = rospy.Publisher("entropy", Float32, queue_size=1)
            self.info_gain_pub = rospy.Publisher("info_gain", Float32, queue_size=1)
            self.eer_time_pub = rospy.Publisher("eer_time", Float32, queue_size=1)
            self.n_eff_pub = rospy.Publisher(
                "n_eff_particles", Float32, queue_size=1
            )
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
            self.filter.resample()
            self.sampled_index = np.arange(self.N_s)
        # Entropy of current distribution
        if self.filter.is_update:
            self.in_FOV = 1
            self.Hp_t = self.entropy_particle(
                self.filter.prev_particles[:, self.sampled_index, :],
                np.copy(self.filter.prev_weights[self.sampled_index]),
                self.filter.particles[:, self.sampled_index, :],
                np.copy(self.filter.weights[self.sampled_index]),
                self.noisy_turtle_pose,
            )
        else:
            self.in_FOV = 0
            #rospy.logwarn("Current_entropy else statement:")
            #rospy.logwarn(self.filter.prev_particles.shape)
            self.Hp_t = self.entropy_particle(
                self.filter.prev_particles[:, self.sampled_index,:],
                np.copy(
                    self.filter.weights[self.sampled_index]
                ),  # current weights are the (t-1) weights because no update
                self.filter.particles[:, self.sampled_index, :],
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
        self.idg_counter+=1
        now = rospy.get_time() - self.initial_time
        #rospy.logwarn("Counter: %d - Elapsed: %f" %(self.idg_counter, now))
        ## Guidance
        future_weight = np.zeros((self.N, self.N_s))
        Hp_k = np.zeros(self.N_s)  # partial entropy
        Ip = np.zeros(self.N_s)  # partial information gain

        prev_future_parts = np.copy(self.filter.prev_particles)
        future_parts = np.copy(self.filter.particles)
        last_future_time = np.copy(self.filter.last_time)
        for k in range(self.k):
            #future_parts = self.filter.motion_model.predict(future_parts)
            future_parts, last_future_time = self.filter.predict(
                future_parts,
                self.filter.weights,
               last_future_time + 0.1,
                angular_velocity=self.angular_velocity,
                linear_velocity=self.linear_velocity,
            )
        # Future possible measurements
        # TODO: implement N_m sampled measurements
        z_hat = self.filter.add_noise(future_parts[-1,self.sampled_index,:], self.filter.measurement_covariance)
        # TODO: implement N_m sampled measurements (double loop)
        for jj in range(self.N_s):
            k_fov = self.construct_FOV(z_hat[jj])
            # checking for measurement outside of fov or in occlusion
            if self.is_in_FOV(z_hat[jj], k_fov) and not self.in_occlusion(z_hat[jj]):
                # TODO: read Jane's math to see exactly how EER is computed
                # the expectation can be either over all possible measurements
                # or over only the measurements from each sampled particle (1 in this case)
                future_weight[:, jj] = self.filter.update(
                    self.filter.weights, future_parts, z_hat[jj]
                )
                # Check to see if weight dimension vs particle dimension is an issue

                # H (x_{t+k} | \hat{z}_{t+k})
                # TODO: figure out how to prevent weights from changing inside this function
                #rospy.logerr(future_weight.shape)
                Hp_k[jj] = self.entropy_particle(
                    prev_future_parts[:,self.sampled_index, :],
                    np.copy(self.filter.weights[self.sampled_index]),
                    future_parts[:, self.sampled_index, :],
                    future_weight[self.sampled_index, jj],
                    z_hat[jj],
                )
            else:
                Hp_k[jj] = self.entropy_particle(
                    prev_future_parts[:, self.sampled_index, :],
                    np.copy(self.filter.weights[self.sampled_index]),
                    future_parts[:, self.sampled_index, :],
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
        self.goal_position = z_hat[action_index][:2]

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
                x=particles, mean=y_meas, cov=self.filter.measurement_covariance
            )

            # likelihood of particle p(xt|xt-1)
            _, part_len, _ = particles.shape
            process_part_like = np.zeros(part_len)
            
            for ii in range(part_len):
                # maybe kinematics with gaussian
                # maybe get weight wrt to previous state (distance)
                like_particle = stats.multivariate_normal.pdf(
                    x=prev_particles,
                    mean=particles[-1, ii, :],
                    cov=self.filter.process_covariance,
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
            _, part_len2, _ = prev_particles.shape
            process_part_like = np.zeros(part_len2)
            for ii in range(part_len2):
                # maybe kinematics with gaussian
                # maybe get weight wrt to previous state (distance)
                like_particle = stats.multivariate_normal.pdf(
                    x=prev_particles,
                    mean=particles[-1, ii, :],
                    cov=self.filter.process_covariance,
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
        else return the next position in the lawnmower path
        """
        if self.filter.is_update:
            return self.noisy_turtle_pose[:2]
        else:
            if self.lawnmower_idx == 0:
                self.increment = 1
            if self.lawnmower_idx >= self.path.shape[0] - 1:
                self.increment = -1
            self.lawnmower_idx += self.increment
            return self.path[self.lawnmower_idx, :2]

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
                self.noisy_turtle_pose = self.filter.add_noise(
                    self.actual_turtle_pose, self.filter.measurement_covariance
                )
            else:
                self.actual_turtle_pose = np.array([msg.pose.position.x, msg.pose.position.y])
                # In the current network we do not use the orientation of the turtlebot
                #turtle_orientation = np.array(
                #    [
                #        msg.pose.orientation.x,
                #        msg.pose.orientation.y,
                #        msg.pose.orientation.z,
                #        msg.pose.orientation.w,
                #    ]
                #)
                #_, _, theta_z = self.euler_from_quaternion(turtle_orientation)
                #self.noisy_turtle_pose = np.array(
                #    [msg.pose.position.x, msg.pose.position.y, theta_z]
                #)
                # in hardware we assume the pose is already noisy
                self.noisy_turtle_pose = np.copy(self.actual_turtle_pose)
                self.pub_desired_state()

            self.filter.turtle_pose = self.noisy_turtle_pose

            if self.is_in_FOV(
                self.noisy_turtle_pose, self.FOV
            ) and not self.in_occlusion(self.noisy_turtle_pose):
                # if not self.in_occlusion(self.noisy_turtle_pose):
                self.filter.is_update = True
            else:
                self.filter.is_update = False  # no update

    def turtle_hardware_odom_cb(self, msg):
        """ Here we get the linear and angular velocities 
        from the wheel encoders"""
        if self.init_finished:
            self.linear_velocity = np.array(
                [msg.twist.twist.linear.x, msg.twist.twist.linear.y]
            )
            self.angular_velocity = np.array([msg.twist.twist.angular.z])

    def rc_cb(self, msg):
        if msg.values[6] > 500:
            self.position_following = True
            #print("rc message > 500, AUTONOMOUS MODE")
        else:
            self.position_following = False
            #print("no rc message > 500, MANUAL MODE: turn it on with the rc controller")

    def quad_odom_cb(self, msg):
        if self.init_finished:
            if self.is_sim:
                self.quad_position = np.array([msg.pose.position.x, -msg.pose.position.y])
            else:
                self.quad_position = np.array([msg.pose.position.x, msg.pose.position.y])  # -y to transform NWU to NED 
            # self.quad_position[1] = -self.quad_position[1] if not self.guidance_mode else self.quad_position[1]
            self.FOV = self.construct_FOV(self.quad_position)

    def pub_desired_state(self, is_velocity=False, xvel=0, yvel=0, event=None):
        if self.init_finished:
            ds = DesiredState()
            # run the quad if sim or the remote controller
            # sends signal of autonomous control
            if self.position_following or self.is_sim:
                if is_velocity:
                    ds.velocity.x = xvel
                    ds.velocity.y = yvel
                    ds.position_valid = False
                    ds.velocity_valid = True
                else:
                    # flip sign of y to transform from NWU to NED
                    if self.guidance_mode == "Information":
                        ds.pose.x = self.goal_position[0]
                        ds.pose.y = -self.goal_position[1]
                    elif self.guidance_mode == "Particles":
                        ds.pose.x = self.noisy_turtle_pose[0]
                        ds.pose.y = -self.noisy_turtle_pose[1]
                    elif self.guidance_mode == "Lawnmower":
                        mower_position = self.lawnmower()
                        ds.pose.x = mower_position[0]
                        ds.pose.y = -mower_position[1]
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
            # FOV err pub
            self.FOV_err = self.quad_position - self.actual_turtle_pose[:2]
            err_fov_msg = PointStamped()
            err_fov_msg.point.x = self.FOV_err[0]
            err_fov_msg.point.y = self.FOV_err[1]
            self.err_fov_pub.publish(err_fov_msg)
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
                self.des_fov = self.construct_FOV(np.array([ds.pose.x, -ds.pose.y]))  # from turtle frame to quad frame
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
                # TODO: change publisher to service
                update_msg = Bool()
                update_msg.data = self.filter.is_update
                self.update_pub.publish(update_msg)

    def pub_pf(self, event=None):
        # Particle pub
        mean_msg = ParticleMean()
        mean_msg.mean.x = self.filter.weighted_mean[0]
        mean_msg.mean.y = self.filter.weighted_mean[1]
        mean_msg.mean.yaw = self.filter.weighted_mean[2]
        for ii in range(self.N):
            particle_msg = Particle()
            particle_msg.x = self.filter.particles[-1, ii, 0]
            particle_msg.y = self.filter.particles[-1, ii, 1]
            particle_msg.yaw = self.filter.particles[-1, ii, 2]
            particle_msg.weight = self.filter.weights[ii]
            mean_msg.all_particle.append(particle_msg)
        mean_msg.cov = np.diag(self.filter.var).flatten("C")
        self.particle_pub.publish(mean_msg)
        # Error pub
        err_msg = PointStamped()
        err_msg.point.x = self.filter.weighted_mean[0] - self.actual_turtle_pose[0]
        err_msg.point.y = self.filter.weighted_mean[1] - self.actual_turtle_pose[1]
        err_msg.point.z = self.filter.weighted_mean[2] - self.actual_turtle_pose[2]
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

        if self.is_sim:
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

    def guidance_pf(self, event=None):
        self.filter.pf_loop(self.noisy_turtle_pose, self.angular_velocity, self.linear_velocity)

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

    time_to_shutdown = 200
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

    rospy.Timer(rospy.Duration(1.0 / 20.0), guidance.guidance_pf) 
    rospy.Timer(rospy.Duration(1.0 / 20.0), guidance.current_entropy)
    if guidance.guidance_mode == "Information":
        rospy.Timer(rospy.Duration(1.0 / 2), guidance.information_driven_guidance)
    
    # Publish topics
    if guidance.is_viz:
        rospy.Timer(rospy.Duration(1.0 / 20.0), guidance.pub_pf)
    rospy.Timer(rospy.Duration(1.0 / 20.0), guidance.pub_desired_state)

    rospy.spin()
