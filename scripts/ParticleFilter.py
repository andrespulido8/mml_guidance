import numpy as np
import rospy
import rospkg
from geometry_msgs.msg import PointStamped, PoseStamped
from mag_pf_pkg.msg import Particle, ParticleMean
from scipy import stats
from std_msgs.msg import Bool, Float32, Float32MultiArray

from mml_network import deploy_mml

FWD_VEL = 0.0
ANG_VEL = 0.0


class ParticleFilter:
    def __init__(self, num_particles=10, prediction_method="NN"):

        deg2rad = lambda deg: np.pi * deg / 180

        self.prediction_method = prediction_method
        self.N_th = 10  # Number of time history particles
        self.N = num_particles

        # boundary of the lab [[x_min, y_min], [x_max, y_,max]]
        self.AVL_dims = np.array([[-2, -1.5], [2, 1.8]])  # road network outline
        # self.AVL_dims = np.array([[-1.5, -2.0], [1.8, 2]])  # road network outline  # before coord transform

        if self.prediction_method == "NN":
            pkg_path = rospkg.RosPack().get_path("mml_guidance")
            model_file = pkg_path + "/scripts/mml_network/models/current.pth"
            training_data_filename = (
                pkg_path + "/scripts/mml_network/no_quad_3hz.csv"
            )  # squarest_yaw.csv'
            self.training_data = np.loadtxt(
                training_data_filename, delimiter=",", skiprows=1
            )[
                :, 1:
            ]  # [500:4000, 1:] # Hardcoded samples
            self.n_training_samples = (
                self.training_data.shape[0] - 9
            )  # TODO: Kyle change this to use N_th
            self.motion_model = deploy_mml.Motion_Model(model_file)

        # PF
        self.N = num_particles
        if self.prediction_method == "Velocity":
            self.Nx = 4  # number of states
            self.vmax = 0.7  # m/s
        elif self.prediction_method == "Unicycle":
            self.Nx = 3
        elif self.prediction_method == "NN":
            self.Nx = 2
        self.weights = np.ones(self.N) / self.N
        self.prev_weights = np.copy(self.weights)
        self.weighted_mean = np.array([0, 0, 0])
        if self.prediction_method == "Unicycle":
            self.measurement_covariance = np.array(
                [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, deg2rad(5)]]
            )
            self.process_covariance = np.array(
                [
                    [0.001, 0.0, 0.0],
                    [0.0, 0.001, 0.0],
                    [0.0, 0.0, 0.0001],
                ]
            )
        else:
            self.measurement_covariance = np.array([[0.01, 0.0], [0.0, 0.01]])
            self.process_covariance = np.array(
                [
                    [0.001, 0.0],
                    [0.0, 0.001],
                ]
            )

        self.measurement_history = np.zeros(
            (self.N_th, self.measurement_covariance.shape[0])
        )
        # Unformly sample particles
        self.particles = self.uniform_sample()
        # Use multivariate normal if you know the initial condition
        # self.particles = np.array([
        #    np.random.multivariate_normal(
        #    np.array([1.3, -1.26, 0]), self.measurement_covariance, self.N
        #    )
        # ])
        self.is_update = False

        self.neff = self.nEff(self.weights)
        self.noise_inv = np.linalg.inv(self.measurement_covariance[:2, :2])
        # Process noise: q11, q22 is meters of error per meter, q33 is radians of error per revolution

        self.var = np.array(
            [
                self.process_covariance[0, 0],
                self.process_covariance[1, 1],
                self.process_covariance[0, 0],
            ]
        )  # initialization of variance of particles

        self.initial_time = rospy.get_time()
        self.last_time = 0.0

    def uniform_sample(self):
        SAMPLE_ALONG_PATH = False
        if SAMPLE_ALONG_PATH:
            rng = np.random.default_rng()  # This is newly recommended method
            indices = rng.integers(0, self.n_training_samples, self.N)
            local_particles = np.empty((10, 0, 3))
            for i in indices:
                local_particles = np.concatenate(
                    (
                        local_particles,
                        np.expand_dims(self.training_data[i : i + 10, :], 1),
                    ),
                    axis=1,
                )

        else:
            if self.prediction_method == "Velocity":
                local_particles = np.random.uniform(
                    [self.AVL_dims[0, 0], self.AVL_dims[0, 1], -self.vmax, -self.vmax],
                    [self.AVL_dims[1, 0], self.AVL_dims[1, 1], self.vmax, self.vmax],
                    (2, self.N, self.Nx),
                )
            elif self.prediction_method == "Unicycle":
                local_particles = np.random.uniform(
                    [self.AVL_dims[0, 0], self.AVL_dims[0, 1], -np.pi],
                    [self.AVL_dims[1, 0], self.AVL_dims[1, 1], np.pi],
                    (2, self.N, self.Nx),
                )
            elif self.prediction_method == "NN":
                local_particles = np.random.uniform(
                    [self.AVL_dims[0, 0], self.AVL_dims[0, 1]],
                    [self.AVL_dims[1, 0], self.AVL_dims[1, 1]],
                    (self.N_th, self.N, self.Nx),
                )
        return local_particles

    def pf_loop(
        self,
        noisy_measurement,
        ang_vel=np.array([ANG_VEL]),
        lin_vel=np.array([FWD_VEL]),
    ):
        """Main function of the particle filter
        where the predict, update, resample, estimate
        and publish PF values for visualization if needed
        """
        t = rospy.get_time() - self.initial_time

        # update measurement history with noisy_measurement
        self.measurement_history = np.roll(self.measurement_history, -1, axis=0)
        self.measurement_history[-1, :2] = noisy_measurement[:2]

        # Prediction step
        if self.prediction_method == "NN":
            self.particles = self.predict_mml(self.particles)
        elif self.prediction_method == "Unicycle":
            self.measurement_history[-1, 2] = noisy_measurement[2]
            self.particles, self.last_time = self.predict(
                self.particles,
                self.weights,
                self.last_time,
                angular_velocity=ang_vel,
                linear_velocity=lin_vel,
            )
        elif self.prediction_method == "Velocity":
            dt = t - self.last_time
            estimate_velocity = (
                self.measurement_history[-1, :] - self.measurement_history[-2, :]
            ) * dt

            self.particles, self.last_time = self.predict(
                self.particles,
                self.weights,
                self.last_time,
            )
        # rospy.logwarn("Mean: %.3f, %.3f | Var: %.3f, %.3f || True: %.3f, %.3f"%(np.mean(self.particles[-1,:,0]), np.mean(self.particles[-1,:,1]), np.var(self.particles[-1,:,0]), np.var(self.particles[-1,:,1]), self.turtle_pose[0], self.turtle_pose[1]))

        # Update step
        if self.is_update:
            self.prev_weights = np.copy(self.weights)
            self.weights = self.update(
                self.weights, self.particles, self.measurement_history[-1]
            )

        # Resampling step
        self.neff = self.nEff(self.weights)
        if self.neff < self.N * 0.9 or self.neff == np.inf and self.is_update:
            if self.neff < self.N * 0.3 or self.neff == np.inf:
                if self.prediction_method == "Velocity":
                    self.particles[-1, :, :2] = np.random.multivariate_normal(
                        self.measurement_history[-1, :2],
                        self.measurement_covariance,
                        self.N,
                    )
                    self.particles[-1, :, 2:] = np.random.multivariate_normal(
                        estimate_velocity,
                        dt * self.measurement_covariance,
                        self.N,
                    )
                else:
                    # Use multivariate normal if you know the initial condition
                    noise = np.random.multivariate_normal(
                        np.zeros(self.measurement_history.shape[1]),
                        self.measurement_covariance,
                        size=(self.N_th, self.N),
                    )
                    # repeat the measurement history to be the same size as the particles
                    measurement_history_repeated = np.tile(
                        self.measurement_history.reshape(self.N_th, 1, self.Nx),
                        (1, self.N, 1),
                    )
                    self.particles = measurement_history_repeated + noise

                self.weights = np.ones(self.N) / self.N
            else:
                # some are good but some are bad, resample
                self.resample()
                # self.systematic_resample()

        self.estimate()
        # print("PF time: ", rospy.get_time() - t - self.initial_time)

    def update(self, weights, particles, noisy_turtle_pose):
        """Updates the belief in the system state.
        In our case, the measurement model is the position and orientation of the
        turtlebot with added noise from a gaussian distribution.
        In MML the update step is the camera model.
        Input: Likelihood of the particles from measurement model and prior belief of the particles
        Output: Updated (posterior) weight of the particles
        """
        weights = weights * self.likelihood(
            particles[-1, :, :], np.tile(noisy_turtle_pose, (self.N, 1))
        )
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        return weights

    def likelihood(self, particles, y_act):
        """Particles that are closer to the noisy measurements are weighted higher than
        particles which don't match the measurements very well.
        There are two methods to compute this.
        """

        # Method 1: Manual for loop with normal multivariate equation
        shape = particles.shape[0]
        like = np.zeros(shape)
        for ii in range(shape):
            # The factor sqrt(det((2*pi)*measurement_cov)) is not included in the
            # likelihood, but it does not matter since it can be factored
            # and then cancelled out during the normalization or expectation.
            like[ii] = np.exp(
                -0.5
                * (particles[ii, :2] - y_act[ii, :2]).T
                @ self.noise_inv
                @ (particles[ii, :2] - y_act[ii, :2])
            )

        # Method 2: Vectorized using scipy.stats
        # TODO fix to account for different measurements
        # like = stats.multivariate_normal.pdf(
        #    x=particles, mean=y_act[0], cov=self.measurement_covariance
        # )
        return like

    def predict_mml(self, particles):
        particles[:, :, :2] = self.motion_model.predict(particles[:, :, :2])
        # for i in range (2):
        #    self.particles[-1,:,i] += self.add_noise( np.zeros(self.N), 0.01*self.process_covariance[i, i], size=self.N )
        return particles

    def predict(
        self,
        particles,
        wgts,
        last_time,
        linear_velocity=np.zeros(2),
        angular_velocity=np.zeros(1),
    ):
        """Uses the process model to propagate the belief in the system state.
        In our case, the process model is the motion of the turtlebot in 2D with added gaussian noise.
        In MML the predict step is a forward pass on the NN.
        Input: State of the particles
        Output: Predicted (propagated) state of the particles
        """
        t = rospy.get_time() - self.initial_time
        dt = t - last_time

        particles[:-1, :, :] = particles[1:, :, :]
        if self.prediction_method == "Unicycle":
            delta_theta = angular_velocity[0] * dt
            particles[-1, :, 2] = (
                particles[-2, :, 2]
                + delta_theta
                + (delta_theta / (2 * np.pi))
                * self.add_noise(
                    np.zeros(self.N), self.process_covariance[2, 2], size=self.N
                )
            )

            for ii in range(self.N):
                if np.abs(particles[-1, ii, 2]) > np.pi:
                    # Wraps angle
                    particles[-1, ii, 2] = (
                        particles[-1, ii, 2] - np.sign(particles[-1, ii, 2]) * 2 * np.pi
                    )

            # Component mean in the complex plane to prevent wrong average
            # source: https://www.rosettacode.org/wiki/Averages/Mean_angle#C.2B.2B
            self.yaw_mean = np.arctan2(
                np.sum(wgts * np.sin(particles[-1, :, 2])),
                np.sum(wgts * np.cos(particles[-1, :, 2])),
            )
            norm_lin_vel = np.linalg.norm(linear_velocity)
            delta_distance = norm_lin_vel * dt + norm_lin_vel * dt * self.add_noise(
                0, self.process_covariance[0, 0], size=self.N
            )
            particles[-1, :, :2] = (
                particles[-2, :, :2]
                + np.array(
                    [
                        delta_distance * np.cos(particles[-1, :, 2]),
                        delta_distance * np.sin(particles[-1, :, 2]),
                    ]
                ).T
            )
        elif self.prediction_method == "Velocity":
            delta_distance = particles[-1, :, 2:] * dt + particles[
                -1, :, 2:
            ] * dt * self.add_noise(
                np.array([0, 0]), self.process_covariance, size=self.N
            )
            particles[-1, :, :2] = (
                particles[-2, :, :2] + delta_distance * particles[-1, :, :2]
            )

        last_time = t

        return particles, last_time

    def systematic_resample(self):
        """Systemic resampling. As with stratified resampling the space is divided into divisions.
        We then choose a random offset to use for all of the divisions, ensuring that each sample
        is exactly 1/N apart"""
        random = np.random.rand(self.N)
        positions = (random + np.arange(self.N)) / self.N

        indexes = np.zeros(self.N, "i")
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        self.particles[-1, :, :] = self.particles[-1, indexes, :]
        self.weights = np.ones(self.N) / self.N

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
        # print("Min: %.4f, Max: %.4f, Dot: %.4f"%(self.weights.min(), self.weights.max(), self.weights.dot(self.weights)))
        indexes = np.random.choice(a=self.N, size=self.N, p=self.weights)
        self.particles[-1, :, :] = self.particles[-1, indexes, :]
        self.weights = self.weights[indexes]
        # Roughening. See Bootstrap Filter from Crassidis and Junkins.
        G = 0.2
        E = np.zeros(self.Nx)
        for ii in range(self.Nx):
            E[ii] = np.max(self.particles[-1, :, ii]) - np.min(
                self.particles[-1, :, ii]
            )
        cov = (G * E * self.N ** (-1 / 3)) ** 2
        P_sigmas = np.diag(cov)

        for ii in range(self.N):
            self.particles[-1, ii, :] = self.add_noise(
                self.particles[-1, ii, :], P_sigmas
            )

    def estimate(self):
        """returns mean and variance of the weighted particles"""
        if np.sum(self.weights) > 0.0:
            self.weighted_mean = np.average(
                self.particles[-1, :, :], weights=self.weights, axis=0
            )
            # TODO: change in pf_viz to only use 2 covariance
            self.var[:2] = np.average(
                (self.particles[-1, :, :2] - self.weighted_mean[:2]) ** 2,
                weights=self.weights,
                axis=0,
            )
            if self.prediction_method == "Unicycle":
                self.var[2] = np.average(
                    (self.particles[-1, :, 2] - self.weighted_mean[2]) ** 2,
                    weights=self.weights,
                    axis=0,
                )
            # source: Differential Entropy in Wikipedia - https://en.wikipedia.org/wiki/Differential_entropy
            self.H_gauss = (
                np.log((2 * np.pi * np.e) ** (3) * np.linalg.det(np.diag(self.var))) / 2
            )

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

    @staticmethod
    def nEff(wgts):
        """Compute the number of effective particles
        Source: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
        """
        return 1.0 / np.sum(np.square(wgts))
