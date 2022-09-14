#!/usr/bin/env python3
import numpy as np
import rospy
import scipy.stats as stats
from geometry_msgs.msg import PointStamped, PoseStamped
from mag_pf_pkg.msg import Particle, ParticleMean
from nav_msgs.msg import Odometry

# from geometry_msgs.msg import Pose
from reef_msgs.msg import DesiredState
from std_msgs.msg import Bool, Float32, Float32MultiArray


class Guidance:
    def __init__(self):
        self.init_finished = False
        # Initialization of variables
        self.turtle_pose = np.array([0, 0, 0])
        self.quad_position = np.array([0, 0])
        self.noisy_turtle_pose = np.array([0, 0, 0])
        self.linear_velocity = np.array([0, 0])
        self.angular_velocity = np.array([0])
        deg2rad = lambda deg: np.pi * deg / 180

        ## PARTICLE FILTER  ##
        self.is_viz = rospy.get_param("/is_viz", False)
        # boundary of the lab [[x_min, y_min], [x_max, y_,max]]
        self.AVL_dims = np.array([[-0.5, -1.5], [2.5, 1.5]])  # road network outline
        # number of particles
        self.N = 500
        self.N_m = 10
        self.measurement_update_time = 5.0
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
            [[0.02, 0, 0], [0, 0.02, 0], [0, 0, deg2rad(5)]]
        )
        self.var = np.diag(self.measurement_covariance)  # variance of particles
        self.H_gauss = 0  # just used for comparison
        self.weighted_mean = np.array([0, 0, 0])
        self.actions = np.array([[0.1, 0], [0, 0.1], [-0.1, 0], [0, -0.1]])
        # Use multivariate normal if you know the initial condition
        # self.particles = np.random.multivariate_normal(
        #    np.array([1.3, -1.26, 0]), 2*self.measurement_covariance, self.N)

        # Measurement Model
        self.height = 1.5
        camera_angle = np.array(
            [deg2rad(69), deg2rad(42)]
        )  # camera angle in radians (horizontal, vertical)
        self.FOV_dims = np.tan(camera_angle) * self.height
        self.FOV = np.array(
            [
                self.quad_position[0] - self.FOV_dims[0] / 2,
                self.quad_position[0] + self.FOV_dims[0] / 2,
                self.quad_position[1] - self.FOV_dims[1] / 2,
                self.quad_position[1] + self.FOV_dims[1] / 2,
            ]
        )
        self.initial_time = rospy.get_time()
        self.time_reset = 0
        self.last_time = 0
        self.last_future_time = 0

        # ROS stuff
        rospy.loginfo("Initializing guidance node")
        self.pose_pub = rospy.Publisher("desired_state", DesiredState, queue_size=1)
        self.turtle_odom_sub = rospy.Subscriber(
            "/robot0/odom", Odometry, self.turtle_odom_cb, queue_size=1
        )
        self.quad_odom_sub = rospy.Subscriber(
            "/pose_stamped", PoseStamped, self.quad_odom_cb, queue_size=1
        )
        #    '/xyEstimate', Odometry, self.quad_odom_cb, queue_size=1)

        if self.is_viz:
            # Particle filter ROS stuff
            self.particle_pub = rospy.Publisher(
                "xyTh_estimate", ParticleMean, queue_size=1
            )
            self.err_estimate_pub = rospy.Publisher(
                "err_estimate", PointStamped, queue_size=1
            )
            self.entropy_pub = rospy.Publisher("entropy", Float32, queue_size=1)
            self.n_eff_pub = rospy.Publisher("n_eff_particles", Float32, queue_size=1)
            self.update_pub = rospy.Publisher("is_update", Bool, queue_size=1)
            self.fov_pub = rospy.Publisher("fov_coord", Float32MultiArray, queue_size=1)

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
        if t - self.time_reset - self.initial_time > self.measurement_update_time:
            # update particles every measurement_update_time seconds
            self.time_reset = t
            rospy.loginfo("Updating weight of particles")
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
            if self.Neff < self.N / 50:
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
                rospy.logwarn(
                    "Resampling particles. Neff: %f < %f",
                    self.Neff,
                    self.N / 2,
                )
                self.resample()

        if self.is_viz:
            self.estimate()
            self.pub_pf()

    def information_driven_guidance(
        self,
    ):
        """Compute the current entropy and future entropy using particles
        to then compute the expected entropy reduction (EER) over predicted
        measurements. The next action is the one that minimizes the EER.
        """
        # print("check2")
        now = rospy.get_time() - self.initial_time
        # Entropy of current distribution
        # TODO: consider only updating this when there is an update
        self.H = self.entropy_particle(
            self.prev_particles,
            self.prev_weights,
            self.particles,
            self.weights,
            self.noisy_turtle_pose,
        )
        entropy_time = rospy.get_time() - self.initial_time
        # print("Entropy time: ", entropy_time - now)

        # print('check3')
        ### Guidance
        # future_weight = np.zeros((self.N, self.N_m))
        # H1 = np.zeros(self.N_m)
        # I = np.zeros(self.N_m)

        # future_part, self.last_future_time  = self.predict(self.particles,
        #                self.prev_particles, self.weights, self.last_future_time)
        ## Future measurement
        # candidates_index = np.random.choice(a=self.N, size=self.N_m,
        #                                                p=None)
        ##update_index = self.is_in_FOV(self.particles)
        ## Future possible measurements
        # z_hat = self.add_noise(
        #        future_part[candidates_index], self.measurement_covariance)
        # if self.update_msg.data:
        #    for jj in range(self.N_m):
        #        future_weight[:,jj] = self.update(self.weights,
        #                                future_part, z_hat[jj])
        #        # H(x_{t+1} | \hat{z}_{t+1})
        #        H1[jj] = self.entropy_particle(self.particles, self.weights,
        #                    future_part, future_weight[:,jj], z_hat[jj])
        #        # Information Gain
        #        I[jj] = self.H - H1[jj]

        #    EER = I.mean()
        #    print("H: ", self.H)
        #    print("H1: ", H1)
        #    print("I: ", I)
        #    print("EER: %f" % EER)

        #    print("\n")
        #    print("EER Time: ", rospy.get_time() - self.initial_time - entropy_time)

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

    def is_in_FOV(self, sparticles):
        """Check if the particles are in the FOV of the camera.
        Input: State of the particles
        Output: Index of the particles in the FOV
        """
        is_inside = np.logical_and(
            sparticles[:, 0] > self.FOV[0],
            sparticles[:, 0] < self.FOV[1],
            sparticles[:, 1] > self.FOV[2],
            sparticles[:, 1] < self.FOV[3],
        )
        return is_inside

    def get_weight(self, particles, y_act, weight):
        """Particles that are closer to the noisy measurements are weighted higher than
        particles which don't match the measurements very well.
        """
        for ii in range(self.N):
            # The factor sqrt(det((2*pi)*measurement_cov)) is not included in the
            # likelihood, but it does not matter since it can be factored
            # and then cancelled out during the normalization.
            like = (
                -0.5
                * (particles[ii, :] - y_act)
                @ self.noise_inv
                @ (particles[ii, :] - y_act)
            )
            weight[ii] = weight[ii] * np.exp(like)

            # another way to implement the above line
        # weight *= stats.multivariate_normal.pdf(x=particles, mean=y_act, cov=self.measurement_covariance)
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

    def entropy_particle(self, prev_particles, prev_weights, particles, weights, y_act):
        """Compute the entropy of the particle distribution based on the equation in the
        paper: Y. Boers, H. Driessen, A. Bagchi, and P. Mandal, 'Particle filter based entropy'
        """
        process_part_like = np.zeros(self.N)
        # likelihoof of measurement p(zt|xt)
        like_meas = stats.multivariate_normal.pdf(
            x=particles, mean=y_act, cov=self.measurement_covariance
        )

        # likelihood of particle p(xt|xt-1)
        for ii in range(self.N):
            # maybe kinematics with gaussian
            # maybe get weight wrt to previous state (distance)
            like_particle = stats.multivariate_normal.pdf(
                x=prev_particles, mean=particles[ii, :], cov=self.proces_covariance
            )
            process_part_like[ii] = np.sum(like_particle)

        # Numerical stability
        cutoff = 1e-4
        like_meas[like_meas < cutoff] = np.nan
        prev_weights[prev_weights < cutoff] = np.nan
        # remove the nans from the likelihoods
        # like_meas = like_meas[~np.isnan(like_meas)]
        process_part_like[process_part_like < cutoff] = np.nan
        product = like_meas * prev_weights
        notnans = product[~np.isnan(product)]
        notnans[notnans < cutoff * 0.01] = np.nan
        product[~np.isnan(product)] = notnans
        first_term = np.log(np.nansum(product))
        first_term = first_term if np.isfinite(first_term) else 0.0
        # second_term = np.nansum(np.log(prev_weights)*weights)
        # third_term = np.nansum(weights*np.log(like_meas))
        # fourth_term = np.nansum(weights*np.log(process_part_like))

        entropy = (
            first_term
            - np.nansum(np.log(prev_weights) * weights)
            - np.nansum(weights * np.log(like_meas))
            - np.nansum(weights * np.log(process_part_like))
        )

        if np.abs(entropy) > 30:
            print("first term: ", np.log(np.nansum(like_meas * prev_weights)))
            print("second term: ", np.nansum(np.log(prev_weights) * weights))
            print("third term: ", np.nansum(weights * np.log(like_meas)))
            print("fourth term: ", np.nansum(weights * np.log(process_part_like)))
            # print('like_meas min: ', like_meas.min())
            # print('like_meas max: ', like_meas.max())
            # print('like_meas mean: ', like_meas.mean())
            # print('like_meas std: ', like_meas.std())

            # print if first term is -inf
            if np.isinf(np.log(np.nansum(like_meas * prev_weights))):
                rospy.logwarn("first term of entropy is -inf. Likelihood is very small")

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

    def quad_odom_cb(self, msg):
        if self.init_finished:
            self.quad_position = np.array([msg.pose.position.x, msg.pose.position.y])
            # self.quad_yaw = self.euler_from_quaternion(msg.pose.orientation)[2]  # TODO: unwrap message before function
            self.FOV = np.array(
                [
                    self.quad_position[0] - self.FOV_dims[0] / 2,
                    self.quad_position[0] + self.FOV_dims[0] / 2,
                    self.quad_position[1] - self.FOV_dims[1] / 2,
                    self.quad_position[1] + self.FOV_dims[1] / 2,
                ]
            )
            # now = rospy.get_time() - self.initial_time
            self.particle_filter()
            # print("particle filter time: ", rospy.get_time() - self.initial_time - now)
            self.information_driven_guidance()

            self.pub_desired_state()

    def pub_desired_state(self, is_velocity=False, xvel=0, yvel=0):
        if self.init_finished:
            ds = DesiredState()
            if is_velocity:
                ds.velocity.x = xvel
                ds.velocity.y = yvel
                ds.position_valid = False
                ds.velocity_valid = True
            else:
                ds.pose.x = self.turtle_pose[0]
                ds.pose.y = -self.turtle_pose[1]
                ds.pose.yaw = 1.571  # 90 degrees
                ds.position_valid = True
                ds.velocity_valid = False
            ds.pose.z = -self.height
            self.pose_pub.publish(ds)
            # Entropy pub
            entropy_msg = Float32()
            entropy_msg.data = self.H
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
        # Particle pub
        self.particle_pub.publish(mean_msg)
        self.err_estimate_pub.publish(err_msg)
        self.update_pub.publish(self.update_msg)
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
        # Number of effective particles pub
        self.fov_pub.publish(fov_msg)
        neff_msg = Float32()
        neff_msg.data = self.Neff
        self.n_eff_pub.publish(neff_msg)


if __name__ == "__main__":
    try:
        rospy.init_node("guidance", anonymous=True)
        square_chain = Guidance()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
        
