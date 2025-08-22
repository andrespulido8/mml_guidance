#!/usr/bin/env python3

import numpy as np
import time
import scipy.stats as stats
from .lawnmower import LawnmowerPath
from .kf import KalmanFilter
from .ParticleFilter import ParticleFilter


class Guidance:
    def __init__(
        self,
        guidance_mode="Information",
        prediction_method="Transformer",
        occ_centers=[[]],
        occ_widths=[],
        filter: ParticleFilter = None,
        initial_time=None,
    ):
        print(
            "Initializing Guidance with mode:",
            guidance_mode,
            "and prediction method:",
            prediction_method,
        )
        self.init_finished = False
        self.guidance_mode = guidance_mode
        self.prediction_method = prediction_method
        self.filter = filter  # particle filter instance
        self.N = filter.N if filter is not None else 500

        # Initialization of robot variables
        self.quad_position = np.array([0.0, 0.0])
        self.actual_turtle_pose = np.array([0.0, 0.0, 0.0])
        self.noisy_turtle_pose = np.array([0.0, 0.0, 0.0])
        self.goal_position = np.array([0.0, 0.0])
        self.linear_velocity = np.array([0.0, 0.0])
        self.angular_velocity = np.array([0.0])
        deg2rad = lambda deg: np.pi * deg / 180
        self.initial_time = initial_time if initial_time is not None else 0

        # Camera Model
        self.height = 100  # initial height of the quadcopter in meters
        self.CAMERA_ANGLES = np.array(
            [deg2rad(45), deg2rad(45)]
        )  # camera angle in radians (horizontal, vertical)
        self.update_FOV_dims_and_measurement_cov()
        self.FOV = self.construct_FOV(self.quad_position)

        ## INFO-DRIVEN GUIDANCE ##
        self.N_s = 25  # Number of sampled particles
        self.K = 4  # Time steps to propagate in the future for EER
        self.Hp_t = 1.0  # partial entropy
        self.prev_Hp = np.ones((5, 1))
        self.EER_range = np.array([0, 0, 0])
        self.eer_particle = 0  # initialize future particle to follow
        self.sampled_index = np.arange(self.N_s)
        self.sampled_particles = self.filter.particles[:, : self.N_s, :]
        self.sampled_weights = np.ones(self.N_s) / self.N_s
        self.position_following = True
        self.avg_time = None
        self.max_time = 0.0
        self.min_time = 10.0
        self.iteration = 0

        # Occlusions
        if occ_centers is not None:
            occ_centers = [occ_centers[:2], occ_centers[2:]]
        self.occlusions = Occlusions(occ_centers, occ_widths)

        if self.guidance_mode == "Lawnmower":
            # Lawnmower Method
            lawnmower = LawnmowerPath(POINTS_PER_SLICE=8)
            bounds = [
                self.filter.APRILab_dims[0],
                np.array(
                    [self.filter.APRILab_dims[1][0], self.filter.APRILab_dims[0][1]]
                ),
                self.filter.APRILab_dims[1],
                np.array(
                    [self.filter.APRILab_dims[0][0], self.filter.APRILab_dims[1][1]]
                ),
            ]
            self.path, _ = lawnmower.generate_path(bounds, path_dist=0.4, angle=0)
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

        self.init_finished = True

    def current_entropy(self, event=None) -> None:
        """Compute the entropy of the current distribution of particles
        Equation changes if there is a measurement available. See paper for reference
        We sample the particle distribution to reduce computation time.
        Output: Hp_t (float) - entropy of the current distribution"""

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

    def propagate_particles(self, future_parts):
        if self.prediction_method in {"NN", "Transformer"}:
            future_parts = self.filter.predict_mml(
                np.copy(future_parts), np.ones(future_parts.shape[0]) * 0.33
            )
        elif self.prediction_method == "Unicycle":
            future_parts = self.filter.predict(
                future_parts,
                0.33,
                angular_velocity=self.angular_velocity,
                linear_velocity=self.linear_velocity,
            )
        elif self.prediction_method == "Velocity":
            future_parts = self.filter.predict(future_parts, 0.33)

    def information_driven_guidance(self, future_parts):
        """Compute the current entropy and future entropy using particles
        to then compute the expected entropy reduction (EER) over predicted
        measurements. The next action is the  position of the particle
        that maximizes the EER.
        Output:
        eer_particle: the index of the particle that maximizes the EER which
        we propagate to choose the goal position
        """
        # Initialize variables
        future_weight = np.zeros((self.N_s, self.N_s))
        Hp_k = np.zeros(self.N_s)  # partial entropy
        EER = np.zeros(self.N_s)  # Expected Entropy Reduction

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
            if self.is_in_FOV(z_hat[jj], k_fov) and not self.occlusions.in_occlusion(
                z_hat[jj]
            ):
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

        action_index = np.argmax(EER)
        self.EER_range = np.array([np.min(EER), np.mean(EER), np.max(EER)])
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

            # if np.abs(entropy) > 30:  # debugging
            # print("\nEntropy term went bad :(")
            # print("second term: ", np.nansum(np.log(prev_wgts) * wgts))
            # print("third term: ", np.nansum(wgts * np.log(like_meas)))
            # print("fourth term: ", np.nansum(wgts * np.log(process_part_like)))

            if np.isinf(first_term):
                print("first term of entropy is -inf. Likelihood is very small")
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
        median = np.median(values)
        std = np.std(values)
        threshold = median + threshold_factor * std
        if abs(current_value - median) > threshold:
            return median
        else:
            return current_value

    def is_in_FOV(self, sparticles, fov):
        """Check if the particles are in the FOV of the camera."""
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

    def update_FOV_dims_and_measurement_cov(self):
        """Update the FOV dimensions based on the camera angles and height of drone
        as well as the measurement covariance based on the height of the drone
        """
        self.FOV_dims = np.tan(self.CAMERA_ANGLES) * self.height
        self.filter.update_measurement_covariance(self.height)

    def lawnmower(self) -> np.ndarray:
        """Return the position of the measurement if there is one,
        else return the next position in the lawnmower path.
        If the rate of pub_desired_state changes, the POINTS_PER_SLICE
        variable needs to change
        """
        if self.filter.is_update:
            return self.noisy_turtle_pose[:2]
        else:
            if (
                np.linalg.norm(self.quad_position - self.path[self.lawnmower_idx, :2])
                > 1.0
            ):
                self.lawnmower_idx = np.argmin(
                    np.linalg.norm(self.path[:, :2] - self.quad_position, axis=1)
                )
            if self.lawnmower_idx <= 0:
                self.increment = 1
            if self.lawnmower_idx >= self.path.shape[0] - 1:
                self.increment = -1
            self.lawnmower_idx += self.increment
            return self.path[int(np.floor(self.lawnmower_idx)), :2]

    def update_goal_position(self, event=None):
        """Get the goal position based on the guidance mode.
        Output: goal_position (numpy.array of shape (2,))"""
        if self.guidance_mode == "Information":
            future_part = np.copy(
                self.filter.particles[:, self.eer_particle, :]
            ).reshape((self.filter.N_th, 1, self.filter.Nx))
            last_future_time = np.copy(self.filter.last_time)
            for k in range(self.K):
                if self.prediction_method in {"NN", "Transformer"}:
                    future_part = self.filter.predict_mml(
                        future_part, np.ones(future_part.shape[0]) * 0.33
                    )
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
        elif self.guidance_mode == "WeightedMean":
            self.goal_position = self.filter.weighted_mean
        elif self.guidance_mode == "Lawnmower":
            mower_position = self.lawnmower()
            self.goal_position = mower_position
        elif self.guidance_mode == "Estimator":
            self.goal_position = self.actual_turtle_pose[:2]

        # set height depending on runtime
        is_height_constant = False
        if is_height_constant:
            self.height = np.clip(self.height, 1.1, 1.8)
        else:
            dheight = 0.02
            if self.filter.is_update:
                self.height -= dheight
            else:
                self.height += dheight
        self.update_FOV_dims_and_measurement_cov()

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
        return np.array([roll_x, pitch_y, yaw_z])

    def update_target_pose(
        self, position, orientation=None, linear_vel=None, angular_vel=None
    ):
        """Update target pose from external source"""
        if orientation is not None:
            if len(orientation) == 4:  # quaternion
                _, _, theta_z = self.euler_from_quaternion(orientation)
                self.actual_turtle_pose = np.array([position[0], position[1], theta_z])
            else:  # assume it's already yaw angle
                self.actual_turtle_pose = np.array(
                    [position[0], position[1], orientation[0]]
                )
        else:
            self.actual_turtle_pose = np.array([position[0], position[1], 0.0])

        if linear_vel is not None:
            self.linear_velocity = np.array(linear_vel[:2])
        if angular_vel is not None:
            self.angular_velocity = np.array([angular_vel[0]])
        
        print("Actual turtle pose:", self.actual_turtle_pose, 
              "Linear velocity:", self.linear_velocity, 
              "Angular velocity:", self.angular_velocity)

        # Add noise to measurement
        self.noisy_turtle_pose[:2] = self.filter.add_noise(
            self.actual_turtle_pose[:2],
            self.filter.measurement_covariance[:2, :2],
        )
        self.noisy_turtle_pose[2] = self.actual_turtle_pose[2]

        # Update FOV and occlusion status
        if self.occlusions.in_occlusion(self.actual_turtle_pose[:2]):
            self.filter.is_occlusion = True
            self.filter.is_update = False
        else:
            self.filter.is_occlusion = False
            if self.is_in_FOV(self.actual_turtle_pose[:2], self.FOV):
                self.filter.is_update = True
            else:
                self.filter.is_update = False

    def update_agent_position(self, position):
        """Update quadcopter position from external source (non-ROS)"""
        self.quad_position = np.array(position[:2])
        self.FOV = self.construct_FOV(self.quad_position)

    def guidance_pf(self, t):
        """Runs the particle (or Kalman) filter loop based on the estimation method"""
        # Select to resample all particles if there is a measurement
        if not self.filter.is_update:
            self.filter.resample_index = np.where(
                np.logical_and(
                    self.is_in_FOV(self.filter.particles[-1], self.FOV),
                    ~self.occlusions.in_occlusion(self.filter.particles[-1, :, :2]),
                )
            )[0]
            if self.filter.t_since_last_update > 1.0:
                # set weights of samples close to zero
                self.filter.weights[self.filter.resample_index] = 1e-10
                # normalize weights
                self.filter.weights = self.filter.weights / np.sum(self.filter.weights)
        else:
            self.filter.resample_index = np.arange(self.N)

        # Run the particle filter loop
        if self.prediction_method == "Unicycle":
            self.filter.pf_loop(
                self.noisy_turtle_pose,
                t,
                self.angular_velocity,
                self.linear_velocity,
            )
        elif self.prediction_method in {"Velocity", "NN", "Transformer"}:
            self.filter.pf_loop(self.noisy_turtle_pose, t)
        elif self.prediction_method == "KF":
            if self.kf.X is None:
                # initialize the KF with the true position
                self.kf.X = np.array(
                    [self.actual_turtle_pose[0], self.actual_turtle_pose[1], 0.1, 0.1]
                )
            self.kf.predict(dt=0.333)
            if self.filter.is_update:
                self.kf.update(self.noisy_turtle_pose[:2])
                self.kf.t_since_last_update = 0.0
            else:
                self.kf.t_since_last_update += 0.333
                if self.kf.t_since_last_update > 9.0:
                    self.kf.t_since_last_update = 0.0
                    self.kf.X = np.array([0.0, 0.0, 0.05, 0.0])
            self.filter.weighted_mean = np.array([self.kf.X[0], self.kf.X[1]])


class Occlusions:
    def __init__(self, positions, widths):
        """List of occlusions with helper functions"""
        self.occlusions = [(positions[ii], widths[ii]) for ii in range(len(positions))]

    def in_occlusion(self, pos):
        """Return true if the position measurement is in occlusion zones for a single
        position but if it is an array of positions, return an array of booleans for
        particles inside occlusion
        Inputs: pos: position to check - numpy.array of shape (2,)
        """
        if pos.ndim == 1:
            for position, width in self.occlusions:
                if np.all(
                    [
                        pos[0] > position[0] - width / 2,
                        pos[0] < position[0] + width / 2,
                        pos[1] > position[1] - width / 2,
                        pos[1] < position[1] + width / 2,
                    ]
                ):
                    return True
            return False
        else:
            return np.array(
                [
                    any(
                        np.all(
                            [
                                pos[ii, 0] > position[0] - width / 2,
                                pos[ii, 0] < position[0] + width / 2,
                                pos[ii, 1] > position[1] - width / 2,
                                pos[ii, 1] < position[1] + width / 2,
                            ]
                        )
                        for position, width in self.occlusions
                    )
                    for ii in range(pos.shape[0])
                ]
            )
