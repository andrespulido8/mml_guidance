#!/home/kyle/pytorch_env/bin/python3
import numpy as np
import rospy
import scipy.stats as stats
from geometry_msgs.msg import PointStamped, PoseStamped
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

        # number of particles
        self.N = 25 # 500 # 
        # Number of future measurements per sampled particle to consider in EER
        self.N_m = 1
        # Number of sampled particles
        self.N_s = 10
        # Time steps to propagate in the future for EER
        self.k = 1
        self.filter = ParticleFilter(self.N)
        self.H_gauss = 0  # just used for comparison
        self.weighted_mean = np.array([0, 0, 0])
        self.actions = np.array([[0.1, 0], [0, 0.1], [-0.1, 0], [0, -0.1]])
        self.position_following = False
        # Use multivariate normal if you know the initial condition
        # self.filter.particles = np.random.multivariate_normal(
        #    np.array([1.3, -1.26, 0]), 2*self.measurement_covariance, self.N)

        # Measurement Model
        self.height = 1.5
        camera_angle = np.array(
            [deg2rad(69), deg2rad(42)]
        )  # camera angle in radians (horizontal, vertical)
        self.FOV_dims = np.tan(camera_angle) * self.height
        self.FOV = self.construct_FOV(self.quad_position)
        self.initial_time = rospy.get_time()
        self.time_reset = 0
        self.last_time = 0
        self.idg_counter = 0
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
            self.fov_pub = rospy.Publisher("fov_coord", Float32MultiArray, queue_size=1)
            self.des_fov_pub = rospy.Publisher(
                "des_fov_coord", Float32MultiArray, queue_size=1
            )
            self.entropy_pub = rospy.Publisher("entropy", Float32, queue_size=1)

        rospy.loginfo("Number of particles for the Bayes Filter: %d", self.N)
        rospy.sleep(1)
        self.init_finished = True

    def current_entropy(self):
        # print("check2")
        now = rospy.get_time() - self.initial_time
        # Entropy of current distribution
        if self.is_in_FOV(self.noisy_turtle_pose, self.FOV):
            self.H_t = self.entropy_particle(
                self.filter.prev_particles,
                np.copy(self.filter.prev_weights),
                self.filter.particles,
                np.copy(self.filter.weights),
                self.noisy_turtle_pose,
            )
        else:
            self.H_t = self.entropy_particle(
                self.filter.prev_particles,
                np.copy(
                    self.filter.weights
                ),  # current weights are the (t-1) weights because no update
                self.filter.particles,
            )

        entropy_time = rospy.get_time() - self.initial_time
        # print("Entropy time: ", entropy_time - now)

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
        self.idg_counter+=1
        now = rospy.get_time() - self.initial_time
        rospy.logwarn("Counter: %d - Elapsed: %f" %(self.idg_counter, now))
        # print('check3')
        ## Guidance
        future_weight = np.zeros((self.N, self.N_s))
        H1 = np.zeros(self.N_s)
        I = np.zeros(self.N_s)

        prev_future_parts = np.copy(self.filter.prev_particles)
        future_parts = np.copy(self.filter.particles)
        last_future_time = np.copy(self.last_time)
        for k in range(self.k):
            future_parts = self.filter.motion_model.predict(future_parts)
            #future_parts, last_future_time = self.filter.predict(
            #    future_parts,
            #    prev_future_parts,
            #    self.filter.weights,
            #   last_future_time + 0.1,
            #    angular_velocity=self.angular_velocity,
            #    linear_velocity=self.linear_velocity,
            #)
        # Future measurements
        prob = np.nan_to_num(self.filter.weights, copy=True, nan=0)
        prob = prob / np.sum(prob)
        candidates_index = np.random.choice(
            a=self.N, size=self.N_s, p=prob
        )  # right now the new choice is uniform
        # Future possible measurements
        # TODO: implement N_m sampled measurements
        z_hat = self.filter.add_noise(
            future_parts[-1,candidates_index,:], self.filter.measurement_covariance
        )
        # TODO: implement N_m sampled measurements (double loop)
        for jj in range(self.N_s):
            k_fov = self.construct_FOV(z_hat[jj])
            # currently next if statement will always be true
            # N_m implementation will change this by
            # checking for measrement outside of fov
            if self.is_in_FOV(z_hat[jj], k_fov):
                # TODO: read Janes math to see exactly how EER is computed
                # the expectation can be either over all possible measurements
                # or over only the measurements from each sampled particle (1 in this case)
                future_weight[:, jj] = self.filter.update(
                    self.filter.weights, future_parts, z_hat[jj]
                )
            # H (x_{t+k} | \hat{z}_{t+k})
            # TODO: figure out how to prevent weights from changing inseide this function
            H1[jj] = self.entropy_particle(
                self.filter.particles,
                np.copy(self.filter.weights),
                future_parts,
                future_weight[:, jj],
                z_hat[jj],
            )
            # Information Gain
            I[jj] = self.H_t - H1[jj]

        # EER = I.mean() # implemented when N_m is implemented
        if False:
            print("H: ", self.H_t)
            print("H1: ", H1)
            print("I: ", I)
            print("EER: %f" % EER)

            print("\n")
        print("EER Time: ", rospy.get_time() - self.initial_time - now)

        action_index = np.argmax(I)

        # print("possible actions: ", z_hat[:, :2])
        # print("information gain: ", I)
        # print("Chosen action:", z_hat[action_index, :2])
        return z_hat[action_index][:2]

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
                x=particles, mean=y_meas, cov=self.filter.measurement_covariance
            )

            # likelihood of particle p(xt|xt-1)
            process_part_like = np.zeros(self.N)
            for ii in range(self.N):
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
            process_part_like = np.zeros(self.N)
            for ii in range(self.N):
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
                self.noisy_turtle_pose = self.filter.add_noise(
                    self.turtle_pose, self.filter.measurement_covariance
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
            self.quad_position[1] = (
                -self.quad_position[1]
                if not self.is_info_guidance
                else self.quad_position[1]
            )
            self.FOV = self.construct_FOV(self.quad_position)
            # now = rospy.get_time() - self.initial_time
            self.filter.pf_loop(
                self.noisy_turtle_pose, self.angular_velocity, self.linear_velocity
            )
            # print("particle filter time: ", rospy.get_time() - self.initial_time - now)
            self.current_entropy()
            if self.filter.update_msg.data and self.is_info_guidance:
                self.goal_position = self.information_driven_guidance()

            self.pub_desired_state()

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
            if self.is_sim and self.is_viz:
                # Entropy pub
                entropy_msg = Float32()
                entropy_msg.data = self.H_t
                self.entropy_pub.publish(entropy_msg)

    def pub_fov(self):
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


if __name__ == "__main__":
    try:
        rospy.init_node("guidance", anonymous=True)
        square_chain = Guidance()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
