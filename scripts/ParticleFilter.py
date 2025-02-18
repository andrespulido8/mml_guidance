import os
import numpy as np
import rospy
import rospkg
import torch
import torch.nn as nn
import torch.optim as optim

import mml_network.motion_model_network as dmmn
from mml_network.simple_dnn import SimpleDNN
from mml_network.scratch_transformer import ScratchTransformer
from mml_network.train import train

FWD_VEL = 0.0
ANG_VEL = 0.0


class ParticleFilter:
    def __init__(
        self, num_particles=10, prediction_method="NN", is_sim=False, drone_height=2.0
    ):

        # Initialize variables
        deg2rad = lambda deg: np.pi * deg / 180
        self.prediction_method = prediction_method
        # boundary of the lab [[x_min, y_min], [x_max, y_,max]] [m]
        self.AVL_dims = np.array([[-1.7, -1.0], [1.5, 1.6]])  # hardware
        self.AVL_dims = (
            self.AVL_dims if not is_sim else np.array([[-1.7, -1.0], [0.9, 2.0]])
        )

        if self.prediction_method == "NN" or self.prediction_method == "Transformer":
            self.N_th = 5  # Number of time history particles
            pkg_path = rospkg.RosPack().get_path("mml_guidance")
            self.is_velocity = True
            is_occlusions_weights = True
            input_dim = 3  # x, y, time
            self.nn_input_size = self.N_th * input_dim
            rospy.loginfo(f"Input dim and size: {input_dim}, {self.nn_input_size}")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            prefix_name = "noisy_5_"  # "connected_graph_noisy_5_" or "noisy_5_" for pretrained on road network
            prefix_name = (
                prefix_name + "velocities_"
                if self.is_velocity
                else prefix_name + "time_"
            )
            prefix_name = (
                prefix_name + "occlusions_" if is_occlusions_weights else prefix_name
            )
            if self.prediction_method == "NN":
                prefix_name = prefix_name + "SimpleDNN"
                self.motion_model = SimpleDNN(
                    input_size=self.nn_input_size,
                    num_layers=2,
                    nodes_per_layer=80,
                    output_size=2,
                    activation_fn="relu",
                )
            elif self.prediction_method == "Transformer":
                prefix_name = prefix_name + "ScratchTransformer"
                self.motion_model = ScratchTransformer(
                    input_dim=input_dim,
                    block_size=self.N_th,
                    n_embed=5,
                    n_head=2,
                    n_layer=2,
                )

            model_file = f"{pkg_path}/scripts/mml_network/models/best_{prefix_name}.pth"
            # load weights
            self.motion_model.load_state_dict(
                torch.load(model_file, map_location=self.device, weights_only=True)
            )
        else:
            self.N_th = 2

        # PF
        self.N = num_particles
        self.weights = np.ones(self.N) / self.N
        self.prev_weights = np.copy(self.weights)
        self.weighted_mean = np.array([0, 0, 0])
        self.is_update = False
        self.is_occlusion = False
        self.neff = self.nEff(self.weights)
        if self.prediction_method == "Velocity":
            self.Nx = 4  # number of states
            self.vmax = 0.7  # m/s
            self.best_measurement_covariance = np.diag([0.01, 0.01])
            self.process_covariance = np.diag([0.05, 0.05])
        elif self.prediction_method == "Unicycle":
            self.Nx = 3
            self.best_measurement_covariance = np.diag([0.01, 0.01, deg2rad(5)])
            self.process_covariance = np.diag([0.05, 0.05, 0.005])
        elif self.prediction_method == "KF" or self.prediction_method == "DMMN":
            self.Nx = 2
            self.best_measurement_covariance = np.diag([0.01, 0.01])
            self.process_covariance = np.diag([0.003, 0.003, 0.0003, 0.0003])
        if self.prediction_method == "DMMN":
            inputSize = 2  # number of inputs to NN
            gamma = 0.1  # a learning for online learning
            k1 = 5.0  # a learning for online learning
            # NN parameters
            numberHiddenLayers = 2  # number of hidden layers
            hiddenSize = 6  # size of each hidden layer
            outputSize = 5  # size of the output of the NN
            probHiddenDrop = 0.0  # dropout probability for hidden layers
            useAttention = False  # use attention at output layer

            # learning parameters
            alpha = 0.001  # learning rate for SGD
            memorySize = 150  # number of data points to save for each training
            batchSize = 20  # size of a batch for each training epoch, should be multiple of memory size
            numberEpochs = 25  # 25  # number of iterations for each training
            minDistance = 0.005  # minimum distance between any two data points

            self.motion_model = dmmn.MotionModel(
                alpha,
                numberHiddenLayers,
                inputSize,
                hiddenSize,
                outputSize,
                probHiddenDrop,
                useAttention,
                memorySize,
                batchSize,
                numberEpochs,
                minDistance,
                gamma,
                k1,
            )

        if self.prediction_method == "NN" or self.prediction_method == "Transformer":
            self.Nx = 2
            self.best_measurement_covariance = np.diag([0.006, 0.006])
            self.process_covariance = np.diag([0.0005, 0.0005])
            buffer_size = (
                300  # large but not infinite to not have dynamic memory allocation
            )
            self.max_batch_size = 64
            assert (
                self.N_th + self.max_batch_size < buffer_size
            ), "Buffer size too small"
        else:
            buffer_size = 1

        self.update_measurement_covariance(height=drone_height)

        self.noise_inv = np.linalg.inv(self.measurement_covariance[:2, :2])
        self.wm_history = np.zeros((self.N_th + buffer_size, 2))
        self.dt_history = np.ones(self.N_th) * 0.333
        self.dt_wm_history = np.zeros(self.wm_history.shape[0])
        self.particles = self.uniform_sample()
        # Use multivariate normal if you know the initial condition
        # self.particles[-1,:, :2] = np.array([
        #   np.random.multivariate_normal(
        #   np.array([1.3, -1.26]), self.measurement_covariance, self.N
        #   )
        # ])

        # Process noise: q11, q22 is meters of error per meter, q33 is radians of error per revolution
        self.var = np.array(
            [
                self.process_covariance[0, 0],
                self.process_covariance[1, 1],
                self.process_covariance[0, 0],
            ]
        )  # initialization of variance of particles

        # Particles to be resampled whether we have measurements or not (in guidance.py)
        self.resample_index = np.arange(self.N)
        self.initial_time = rospy.get_time()
        self.last_time = 0.0
        self.last_update_time = 0.0
        self.iteration = 0

    def uniform_sample(self) -> np.ndarray:
        """Uniformly samples the particles between min and max values per state.
        Positions are always sampled according to the lab dimensions.
        Output:
            local_particles: N_th sets (time histories) of N particles with Nx states
        """
        if self.prediction_method == "Velocity":
            local_particles = np.random.uniform(
                [self.AVL_dims[0, 0], self.AVL_dims[0, 1], -self.vmax, -self.vmax],
                [self.AVL_dims[1, 0], self.AVL_dims[1, 1], self.vmax, self.vmax],
                (self.N_th, self.N, self.Nx),
            )
        elif self.prediction_method == "Unicycle":
            local_particles = np.random.uniform(
                [self.AVL_dims[0, 0], self.AVL_dims[0, 1], -np.pi],
                [self.AVL_dims[1, 0], self.AVL_dims[1, 1], np.pi],
                (self.N_th, self.N, self.Nx),
            )
        elif (
            self.prediction_method == "NN"
            or self.prediction_method == "KF"
            or self.prediction_method == "DMMN"
            or self.prediction_method == "Transformer"
        ):
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
        """Main function of the particle filter where the predict,
        update, resample and estimate steps are called.
        """
        # t = rospy.get_time() - self.initial_time
        noisy_measurement = noisy_measurement[:2] if self.Nx > 2 else noisy_measurement

        # Prediction step
        if self.prediction_method == "NN" or self.prediction_method == "Transformer":
            self.particles = self.predict_mml(np.copy(self.particles), self.dt_history)
        elif self.prediction_method == "Unicycle":
            self.particles = self.predict(
                self.particles,
                self.dt_history[-1],
                angular_velocity=ang_vel,
                linear_velocity=lin_vel,
            )
        elif self.prediction_method == "Velocity":
            estimate_velocity = (
                noisy_measurement - self.wm_history[-1, :]
            ) * self.dt_history[-1]

            self.particles = self.predict(
                self.particles,
                self.dt_history[-1],
            )

        # Update step
        if self.is_update:
            self.prev_weights = np.copy(self.weights)
            self.weights = self.update(
                self.weights, self.particles, noisy_measurement 
            )

        # Resampling step
        outbounds = self.outside_bounds(self.particles[-1])
        self.neff = self.nEff(self.weights)
        if outbounds > self.N * 0.8:
            # Resample if fraction of particles are outside the lab boundaries
            rospy.logwarn(
                f"{self.outside_bounds(self.particles[-1])} particles outside the lab boundaries. Uniformly resampling."
            )
            self.particles = self.uniform_sample()
        else:
            if (
                self.neff < self.N * 0.99
            ):  # nEff is only infinity when something went wrong
                if (self.neff < self.N * 0.4 or self.neff == self.N) and self.is_update:
                    # most particles are bad, resample from Gaussian around the measurement
                    if self.prediction_method == "Velocity":
                        self.particles[-1, :, :2] = np.random.multivariate_normal(
                            noisy_measurement,
                            self.measurement_covariance,
                            self.N,
                        )
                        # backwards difference for velocities
                        self.particles[-1, :, 2:] = np.random.multivariate_normal(
                            estimate_velocity,
                            self.dt_history[-1] * self.measurement_covariance,
                            self.N,
                        )
                    else:
                        noise = np.random.multivariate_normal(
                            np.zeros(self.Nx),
                            self.measurement_covariance,
                            size=(self.N_th, self.N),
                        )
                        # repeat the measurement history to be the same size as the particles
                        wm_history_repeated = np.tile(
                            self.wm_history[-self.N_th :].reshape(
                                self.N_th, 1, self.Nx
                            ),
                            (1, self.N, 1),
                        )
                        self.particles = wm_history_repeated + noise

                    self.weights = np.ones(self.N) / self.N
                else:
                    # some are good but some are bad, resample
                    self.multinomial_resample()

        stuck_threshold = 0.01 if self.prediction_method == "Velocity" else 0.03
        wm_pos_diff = np.linalg.norm(self.wm_history[0][:2] - self.wm_history[-1][:2])
        # print("Weighted mean position difference: ", wm_pos_diff)
        if  wm_pos_diff < stuck_threshold or self.dt_wm_history[-1] > 3.:
            rospy.logwarn("Particle filter stuck or lost measurement. Uniformly resampling.")
            self.particles = self.uniform_sample()

        # estimate mean and variance
        self.weighted_mean, self.var = self.estimate(self.particles[-1], self.weights)

        self.update_weighted_mean_and_dt_history(noisy_measurement)

        self.last_time = rospy.get_time() - self.initial_time

    def optimize_learned_model_callback(self, event=None):
        """Optimize the neural network model with the particles and weights"""
        filled_elements = np.count_nonzero(self.wm_history) // self.Nx
        if not np.all(self.wm_history):
            rospy.loginfo(
                f"Filled elements: {filled_elements} / {self.wm_history.shape[0]}"
            )
        # if measurement history buffer is full
        if (
            filled_elements
            > self.N_th + self.max_batch_size * 0.6
            and (self.prediction_method == "NN" or self.prediction_method == "Transformer") 
            # and False
        ):
            self.optimize_learned_model(filled_elements)

    def update(self, weights, particles, noisy_turtle_pose):
        """Updates the belief (weights) of the particle distribution.
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

    def predict_mml(self, particles, dt_history):
        """Propagates the current particles through the motion model learned with the
        neural network.
        Input: N_th (number of time histories) sets of particles up to time k-1
        Output: N_th sets of propagated particles up to time k
        """
        tiled_dt_hist = np.tile(
            dt_history.reshape(-1, 1), (1, particles.shape[1])
        ).reshape(-1, particles.shape[1], 1)
        particles = np.concatenate((particles, tiled_dt_hist), axis=2)

        next_parts = self.motion_model.predict(
            np.transpose(particles[:, :, :], (1, 0, 2)).reshape(
                particles.shape[1], self.nn_input_size
            )
        )

        # print("shape of next_parts: ", next_parts.shape)
        # print("next_parts: ", next_parts[0])
        if self.is_velocity:
            next_parts = (
                particles[-1, :, :2] + next_parts * dt_history[-1]
            )  # x(t+1) = x(t) + v(t) * dt

        # roll particles in time, add latest prediction
        particles = np.concatenate(
            (
                particles[1:, :, :2],
                next_parts.reshape(1, next_parts.shape[0], next_parts.shape[1]),
            ),
            axis=0,
        )
        # important: will not work without this
        particles[-1, :, :] = self.add_noise(
            particles[-1, :, :], 2 * self.process_covariance
        )
        return particles

    def predict(
        self,
        particles,
        dt,
        linear_velocity=np.zeros(2),
        angular_velocity=np.zeros(1),
    ):
        """Uses the unicycle model plus noise or the to propagate the belief in the system state.
        The position and orientation of the particles are updated according to the linear
        and angular velocity of the turtlebot.
        In the Velocity method, the particles are propagated according to the velocity of each particle.
        Input: State of the particles at time k-1
        Output: Predicted (propagated) state of the particles up to time k
        """
        particles[:-1, :, :] = particles[1:, :, :]  # shift particles in time
        if self.prediction_method == "Unicycle":
            delta_theta = angular_velocity[0] * dt
            particles[-1, :, 2] = (
                particles[-2, :, 2]
                + delta_theta
                + (delta_theta / (2 * np.pi))
                * self.add_noise(
                    np.zeros(particles.shape[1]),
                    self.process_covariance[2, 2],
                    size=particles.shape[1],
                )
            )

            for ii in range(particles.shape[1]):
                if np.abs(particles[-1, ii, 2]) > np.pi:
                    # Wraps angle
                    particles[-1, ii, 2] = (
                        particles[-1, ii, 2] - np.sign(particles[-1, ii, 2]) * 2 * np.pi
                    )

            norm_lin_vel = np.linalg.norm(linear_velocity)
            delta_distance = norm_lin_vel * dt + norm_lin_vel * dt * self.add_noise(
                0, self.process_covariance[0, 0], size=particles.shape[1]
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
                np.array([0, 0]), self.process_covariance, size=particles.shape[1]
            )
            particles[-1, :, :2] = (
                particles[-2, :, :2] + delta_distance * particles[-1, :, :2]
            )

        return particles

    def multinomial_resample(self):
        """Uses the multinomial resampling algorithm to update the belief in the system state.
        The particles are copied randomly with probability proportional to the weights plus
        some roughening based on the spread of the states.
        Inputs: Updated state of the particles
        Outputs: Resampled updated state of the particles
        """
        N_r = len(self.resample_index)
        if N_r > 0:
            # Normalize weights
            self.weights = (
                self.weights / np.sum(self.weights)
                if np.sum(self.weights) > 0
                else self.weights
            )
            # Copy particles that are weighted higher
            rand_ind = np.random.choice(a=self.N, size=N_r, p=self.weights)
            self.particles[-1, self.resample_index, :] = self.particles[-1, rand_ind, :]
            self.weights[self.resample_index] = self.weights[rand_ind]
            # Roughening. See Bootstrap Filter from Crassidis and Junkins.
            G = 0.2
            E = np.zeros(self.Nx)
            for ii in range(self.Nx):
                E[ii] = np.max(self.particles[-1, :, ii]) - np.min(
                    self.particles[-1, :, ii]
                )
            cov = (G * E * N_r ** (-1 / 3)) ** 2
            P_sigmas = np.diag(cov)

            for ii in self.resample_index:
                self.particles[-1, ii, :] = self.add_noise(
                    self.particles[-1, ii, :], P_sigmas
                )

    def estimate(self, particles, weights):
        """returns mean and variance of the weighted particles"""
        if np.sum(weights) > 0.0:
            weighted_mean = np.average(particles, weights=weights, axis=0)
            # TODO: change in pf_viz to only use 2 covariance
            var = np.zeros_like(self.var)
            var[:2] = np.average(
                (particles[:, :2] - weighted_mean[:2]) ** 2,
                weights=weights,
                axis=0,
            )
            if self.prediction_method == "Unicycle":
                # Component mean in the complex plane to prevent wrong average
                # source: https://www.rosettacode.org/wiki/Averages/Mean_angle#C.2B.2B
                self.yaw_mean = np.arctan2(
                    np.sum(self.weights * np.sin(particles[:, 2])),
                    np.sum(self.weights * np.cos(particles[:, 2])),
                )
                weighted_mean[2] = self.yaw_mean
                var[2] = np.average(
                    (particles[:, 2] - weighted_mean[2]) ** 2,
                    weights=weights,
                    axis=0,
                )
            ## source: Differential Entropy for a Gaussian in Wikipedia
            ## https://en.wikipedia.org/wiki/Differential_entropy
            # H_gauss = (
            #    np.log((2 * np.pi * np.e) ** (3) * np.linalg.det(np.diag(var))) / 2
            # )
            return weighted_mean, var

    def outside_bounds(self, particles):
        """returns the number of particles outside of the lab boundaries"""
        return np.sum(
            (particles[:, 0] < self.AVL_dims[0, 0])
            | (particles[:, 0] > self.AVL_dims[1, 0])
            | (particles[:, 1] < self.AVL_dims[0, 1])
            | (particles[:, 1] > self.AVL_dims[1, 1])
        )

    def update_weighted_mean_and_dt_history(self, noisy_measurement):
        """Update the measurement history and the time history"""
        t = rospy.get_time() - self.initial_time

        dt = t - self.last_time
        # pf dt history
        self.dt_history[:-1] = self.dt_history[1:] + dt  # delta time from index i to -1
        self.dt_history[-1] = dt

        if self.is_update:
            self.wm_history = np.roll(self.wm_history, -1, axis=0)
            self.wm_history[-1, :2] = noisy_measurement[:2]
            update_dt = t - self.last_update_time
            self.dt_wm_history[:-1] = (
                self.dt_wm_history[1:] + update_dt
            )
            self.dt_wm_history[-1] = update_dt
            self.last_update_time = t

    def update_measurement_covariance(self, height):
        """ Update the measurement covariance matrix based on the height of the drone agent
        """
        gain = 1 + (max(height, 1.1) - 1.1) / (
            2.0 - 1.1
        )  # start with 2 (h=2.) and decrease to 1 (h=1.1)
        # print("Gain: ", gain)
        self.measurement_covariance = gain * self.best_measurement_covariance
        # print("Measurement covariance: ", np.diag(self.measurement_covariance))

    def convert_wm_history_to_training_batches(self, filled_elements, all_data=False):
        """Convert the measurement history to training batches for the neural network
            Input: filled_elements: number of elements in the measurement history buffer
            Output: X_train: input training data, y_train: output training data
        """
        wm_history = self.wm_history[-filled_elements:]
        dt_wm_history = self.dt_wm_history[-filled_elements:]
        if all_data:
            batch_size = filled_elements - self.N_th
        else:
            batch_size = np.min((self.max_batch_size, filled_elements - self.N_th))
            if batch_size <= 0:
                return None, None
        X_train = np.zeros((batch_size, self.N_th, self.Nx))
        dt_time_features = np.zeros((batch_size, self.N_th))
        y_train = np.zeros((batch_size, self.Nx))

        # sample n_batches indexes from the data
        if all_data:
            idx = np.arange(batch_size)
        else:
            idx = np.random.choice(
                wm_history.shape[0] - self.N_th, batch_size, replace=False
            )

        # TODO: avoid for loop
        include_index = []
        for i, sampled_i in enumerate(idx):
            X_train[i] = wm_history[sampled_i : sampled_i + self.N_th].copy()
            dt_time_features[i] = (
                dt_wm_history[sampled_i : sampled_i + self.N_th].copy()
                - dt_wm_history[sampled_i + self.N_th]
            )
            y_train[i] = wm_history[sampled_i + self.N_th].copy()

            # remove samples with dt > 4 seconds
            if dt_time_features[i][0] < 4.:
                include_index.append(i)

        X_train = np.concatenate((X_train, dt_time_features[..., np.newaxis]), axis=2)

        include_index = np.array(include_index)
        rospy.loginfo(f"eliminated samples: {batch_size - len(include_index)}")
        X_train = X_train[include_index]
        y_train = y_train[include_index]

        if self.is_velocity:
            y_train = (y_train - X_train[:, -1, :2]) / X_train[:, -1, 2].reshape(
                -1, 1
            )  # vel = (x(t+1) - x(t)) / dt

        return X_train, y_train

    def optimize_learned_model(self, filled_elements):
        """Optimize the neural network model with the particles and weights"""
        print("\n")
        rospy.loginfo("Optimizing the learned model")
        # training parameters
        epochs = 5
        criterion = nn.MSELoss().to(torch.float32)
        optimizer = optim.Adam(self.motion_model.parameters(), lr=0.001)
        # data transformation
        X_train, y_train = self.convert_wm_history_to_training_batches(filled_elements)
        if self.prediction_method == "NN":
            X_train = X_train.reshape(X_train.shape[0], -1)  # flatten last two dimensions

        X_train = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
        y_train = torch.from_numpy(y_train.astype(np.float32)).to(self.device)

        losses = train(
            self.motion_model,
            criterion,
            optimizer,
            X_train,
            y_train,
            epochs,
            online=True,
        )

    def save_model(self):
        """Save the model to a file"""
        pkg_path = rospkg.RosPack().get_path("mml_guidance")
        model_file = f"{pkg_path}/scripts/mml_network/models/online_model.pth"
        if self.prediction_method in {"NN", "Transformer"}:
            torch.save(self.motion_model.state_dict(), model_file)
            csv_file = f"{pkg_path}/sim_data/training_data/online_data.csv"
            if not os.path.exists(csv_file):
                os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            filled_elements = np.count_nonzero(self.wm_history) // self.Nx
            X_train, y_train = self.convert_wm_history_to_training_batches(filled_elements, all_data=True)
            if X_train is None:
                rospy.logwarn("No data to save")
                return
            X_train_flattened = X_train.reshape(X_train.shape[0], -1)
            wm_history_flattened = np.concatenate((X_train_flattened, y_train, np.zeros((X_train_flattened.shape[0], 1))), axis=1)
            np.savetxt(csv_file, wm_history_flattened, delimiter=",")
            rospy.loginfo(f"Data saved to {csv_file}")
        elif self.prediction_method == "DMMN":
            self.motion_model.saveModel(time=rospy.get_time(), savePath=model_file)
        rospy.loginfo(f"Model saved to {model_file}")

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
