import numpy as np
import time # For testing in the absence of rospy.get_time()

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Bool, Float32, Float32MultiArray
from mag_pf_pkg.msg import Particle, ParticleMean
from scipy import stats

FWD_VEL = 0.0
ANG_VEL = 0.0

class ParticleFilter():
    def __init__(self, num_particles=1000):
        deg2rad = lambda deg: np.pi * deg / 180

        self.is_viz = rospy.get_param("/is_viz", False)  # true to visualize plots
        # boundary of the lab [[x_min, y_min], [x_max, y_,max]]
        self.AVL_dims = np.array([[-0.75, -1.75], [2.75, 1.75]])  # road network outline

        self.N = num_particles
        # uniform distribution of particles (x, y, theta)
        self.particles = np.random.uniform(
            [self.AVL_dims[0, 0], self.AVL_dims[0, 1], -np.pi/10.],
            [self.AVL_dims[1, 0], self.AVL_dims[1, 1], np.pi/10.],
            (self.N, 3),
        )
        self.initial_time = time.time()
        self.last_time = 0.
        self.time_reset = 0.
        self.measurement_update_time = 5.0
        self.weights = np.ones(self.N) / self.N
        self.prev_weights = np.copy(self.weights)
        self.measurement_covariance = np.array(
            [[0.1, 0., 0.], [0., 0.1, 0.], [0., 0., deg2rad(5)]]
        )
        self.noise_inv = np.linalg.inv(self.measurement_covariance)
        # Process noise: q11, q22 is meters of error per meter, q33 is radians of error per revolution
        self.proces_covariance = np.array(
            [[0.02, 0., 0.], [0., 0.02, 0.], [0., 0., deg2rad(5)]]
        )
        self.var = np.diag(self.measurement_covariance)  # variance of particles

        self.turtle_pose = np.array([0., 0., 0.])
        #self.fake_sensor()
        self.udpate_msg = Bool()
        

    def pf_loop(self):
        t = rospy.get_time() - self.initial_time
        self.predict(angular_velocity=np.array([ANG_VEL]), linear_velocity=np.array([FWD_VEL]))
        if t - self.time_reset  > self.measurement_update_time:
            self.time_reset = t
            rospy.loginfo("Updating weight of particles")
            self.prev_weights = np.copy(self.weights)
            self.update()
            self.update_msg.data=True
        else:
            self.update_msg.data=False

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

    def update(self):
        """Updates the belief in the system state.
        In our case, the measurement model is the position and orientation of the
        turtlebot with added noise from a gaussian distribution.
        In MML the update step is the camera model.
        Input: Likelihood of the particles from measurement model and prior belief of the particles
        Output: Updated (posterior) weight of the particles
        """
        self.weights *= stats.multivariate_normal.pdf(
            x=self.particles, mean=self.noisy_turtle_pose, cov=self.measurement_covariance
        )

        self.weights = self.weights / np.sum(self.weights) if np.sum(self.weights) > 0 else self.weights

    def predict(self, angular_velocity, linear_velocity):
        """Uses the process model to propagate the belief in the system state.
        In our case, the process model is the motion of the turtlebot in 2D with added gaussian noise.
        In MML the predict step is a forward pass on the NN.
        Input: State of the particles
        Output: Predicted (propagated) state of the particles
        """
        t = rospy.get_time() - self.initial_time
        dt = t - self.last_time

        prev_particles = np.copy(self.particles)
        delta_theta = angular_velocity[0] * dt
        self.particles[:, 2] = (
            prev_particles[:, 2]
            + delta_theta
            + (delta_theta / (2 * np.pi))
            * self.add_noise(
                np.zeros(self.N), self.proces_covariance[2, 2], size=self.N
            )
        )

        for ii in range(self.N):
            if np.abs(self.particles[ii, 2]) > np.pi:
                # Wraps angle
                self.particles[ii, 2] = (
                    self.particles[ii, 2] - np.sign(self.particles[ii, 2]) * 2 * np.pi
                )

        # Component mean in the complex plane to prevent wrong average
        # source: https://www.rosettacode.org/wiki/Averages/Mean_angle#C.2B.2B
        self.yaw_mean = np.arctan2(
            np.sum(self.weights * np.sin(self.particles[:, 2])),
            np.sum(self.weights * np.cos(self.particles[:, 2])),
        )
        delta_distance = linear_velocity[0] * dt + linear_velocity[
            0
        ] * dt * self.add_noise(0, self.proces_covariance[0, 0], size=self.N)
        self.particles[:, :2] = (
            prev_particles[:, :2]
            + np.array(
                [
                    delta_distance * np.cos(self.particles[:, 2]),
                    delta_distance * np.sin(self.particles[:, 2]),
                ]
            ).T
        )

        last_time = t - self.initial_time

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
    def fake_sensor(self):
        self.noisy_turtle_pose = self.turtle_pose + np.random.uniform([-0.01, -0.01, -0.005], [0.01, 0.01, 0.005], (3,))

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

    def neff(self, weights):
        """Compute the number of effective particles
        Source: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
        """
        return 1.0 / np.sum(np.square(weights))

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
        # TODO: change publisher to service
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