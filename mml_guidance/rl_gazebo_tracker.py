#!/usr/bin/env python3
"""
RL Gazebo Tracker - Adapted from rl_offboard_tracker.py
Tracks TurtleBot3 in Gazebo using SJTU Drone (no MAVROS)
Uses trained RL policy with boundary safety
"""
import sys
sys.path.append('/home/basestation/ros2_ws/src/mml_python_sim')
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Bool
import numpy as np

# NumPy compatibility patch
try:
    import numpy._core.numeric
except ImportError:
    print("Applying NumPy 2.0 -> 1.x Compatibility Patch...")
    import numpy
    if hasattr(numpy, 'core'):
        sys.modules['numpy._core'] = numpy.core
        sys.modules['numpy._core.numeric'] = numpy.core.numeric

from mml_guidance.guidance import Guidance
from mml_guidance.ParticleFilter import ParticleFilter


class RLGazeboTracker(Node):
    """
    RL-based tracker for SJTU Drone in Gazebo
    Adapted from MAVROS version to use Gazebo native control
    """
    
    def __init__(self):
        super().__init__('rl_gazebo_tracker')

        # ═══════════════════════════════════════════════════════════════
        # PARAMETERS
        # ═══════════════════════════════════════════════════════════════
        self.declare_parameter('model_path', '')
        self.declare_parameter('target_topic', '/odom')  # TurtleBot3 odometry
        self.declare_parameter('drone_pose_topic', '/simple_drone/gt_pose')
        self.declare_parameter('drone_height', 3.0)
        self.declare_parameter('num_particles', 200)
        self.declare_parameter('prediction_method', 'Velocity')
        self.declare_parameter('max_vel', 0.8)
        self.declare_parameter('max_yaw_rate', 1.0)
        
        # Boundary parameters
        self.declare_parameter('boundary_x_min', -5.0)
        self.declare_parameter('boundary_x_max', 5.0)
        self.declare_parameter('boundary_y_min', -5.0)
        self.declare_parameter('boundary_y_max', 5.0)
        self.declare_parameter('boundary_z_min', 0.5)
        self.declare_parameter('boundary_z_max', 5.0)
        self.declare_parameter('safe_zone_radius', 1.0)
        self.declare_parameter('warning_distance', 0.5)
        self.declare_parameter('emergency_distance', 0.3)
        self.declare_parameter('boundary_return_speed', 0.5)
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        target_topic = self.get_parameter('target_topic').value
        drone_pose_topic = self.get_parameter('drone_pose_topic').value
        self.target_height = self.get_parameter('drone_height').value
        num_particles = self.get_parameter('num_particles').value
        prediction_method = self.get_parameter('prediction_method').value
        self.max_vel = self.get_parameter('max_vel').value
        self.max_yaw_rate = self.get_parameter('max_yaw_rate').value
        
        # Boundaries
        self.boundary_x_min = self.get_parameter('boundary_x_min').value
        self.boundary_x_max = self.get_parameter('boundary_x_max').value
        self.boundary_y_min = self.get_parameter('boundary_y_min').value
        self.boundary_y_max = self.get_parameter('boundary_y_max').value
        self.boundary_z_min = self.get_parameter('boundary_z_min').value
        self.boundary_z_max = self.get_parameter('boundary_z_max').value
        self.safe_zone_radius = self.get_parameter('safe_zone_radius').value
        self.warning_distance = self.get_parameter('warning_distance').value
        self.emergency_distance = self.get_parameter('emergency_distance').value
        self.boundary_return_speed = self.get_parameter('boundary_return_speed').value
        
        # Calculate safe center
        self.safe_center_x = (self.boundary_x_min + self.boundary_x_max) / 2.0
        self.safe_center_y = (self.boundary_y_min + self.boundary_y_max) / 2.0
        self.safe_center_z = (self.boundary_z_min + self.boundary_z_max) / 2.0
        
        self.in_emergency_return = False
        self.boundary_violations = 0
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("RL GAZEBO TRACKER (SJTU DRONE + TURTLEBOT3)")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Model: {model_path}")
        self.get_logger().info(f"Target: {target_topic}")
        self.get_logger().info(f"Drone pose: {drone_pose_topic}")
        self.get_logger().info(f"Target height: {self.target_height}m")
        self.get_logger().info(f"Max velocity: {self.max_vel}m/s")
        self.get_logger().info(f"Boundaries: X[{self.boundary_x_min}, {self.boundary_x_max}] "
                              f"Y[{self.boundary_y_min}, {self.boundary_y_max}]")
        self.get_logger().info("=" * 60)
        
        # ═══════════════════════════════════════════════════════════════
        # INITIALIZE GUIDANCE SYSTEM
        # ═══════════════════════════════════════════════════════════════
        self.pf = ParticleFilter(
            num_particles=num_particles,
            prediction_method=prediction_method,
            drone_height=self.target_height
        )
        
        self.guidance = Guidance(
            guidance_mode="MultiPFInfoPPO_LSTM",
            prediction_method=prediction_method,
            drone_height=self.target_height,
            filter=self.pf,
            ppo_model_path=model_path
        )
        
        self.lstm_states = None
        self.episode_start = np.array([True])
        
        # ═══════════════════════════════════════════════════════════════
        # STATE STORAGE
        # ═══════════════════════════════════════════════════════════════
        self.drone_pos = None
        self.drone_orient = None
        self.drone_vel = np.array([0.0, 0.0, 0.0])
        self.target_pos = None
        self.target_vel = np.array([0.0, 0.0])
        self.last_target_time = None
        self.takeoff_complete = False
        
        # QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # ═══════════════════════════════════════════════════════════════
        # SUBSCRIBERS
        # ═══════════════════════════════════════════════════════════════
        self.sub_drone_pose = self.create_subscription(
            Pose, drone_pose_topic, self.drone_pose_cb, qos_profile)
        
        self.sub_drone_odom = self.create_subscription(
            Odometry, '/simple_drone/odom', self.drone_odom_cb, qos_profile
        )
        
        self.sub_target = self.create_subscription(
            Odometry, target_topic, self.target_cb, qos_profile)
        
        # ═══════════════════════════════════════════════════════════════
        # PUBLISHERS
        # ═══════════════════════════════════════════════════════════════
        self.vel_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.posctrl_pub = self.create_publisher(Bool, '/simple_drone/posctrl', 10)
        
        # ═══════════════════════════════════════════════════════════════
        # TAKEOFF SEQUENCE
        # ═══════════════════════════════════════════════════════════════
        self.get_logger().info("Initiating takeoff sequence...")
        
        # Send takeoff command
        for _ in range(10):
            self.takeoff_pub.publish(Empty())
        
        # Enable position control (optional for SJTU drone)
        posctrl_msg = Bool()
        posctrl_msg.data = False
        self.posctrl_pub.publish(posctrl_msg)
        
        # ═══════════════════════════════════════════════════════════════
        # START CONTROL LOOP
        # ═══════════════════════════════════════════════════════════════
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20Hz
        self.get_logger().info("RL Tracker active! Following TurtleBot3...")
    
    # ═══════════════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════════════
    
    def drone_pose_cb(self, msg):
        """Update drone position from Gazebo"""
        current_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.drone_orient = [msg.orientation.x, msg.orientation.y,
                            msg.orientation.z, msg.orientation.w]
        
        # Estimate velocity if odometry not available
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        if self.last_drone_pos is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0.001:  # Avoid division by zero
                # Numerical derivative (global frame)
                self.drone_vel = (current_pos - self.last_drone_pos) / dt
        
        self.last_drone_pos = current_pos.copy()
        self.last_time = current_time
        self.drone_pos = current_pos
        
        self.guidance.update_agent_position(self.drone_pos[:2])
    
    # def target_cb(self, msg):
    #     """Update target from TurtleBot3 odometry"""
    #     pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
    #     self.guidance.update_target_pose(position=pos)
        
    #     current_time = self.get_clock().now().nanoseconds / 1e9
    #     self.guidance.guidance_filter_loop(t=current_time)
        
    #     self.target_pos = np.array(pos)
        
    #     # Estimate velocity
    #     if hasattr(msg.twist, 'twist'):
    #         v = msg.twist.twist.linear
    #     else:
    #         v = msg.twist.linear
    #     self.target_vel = np.array([v.x, v.y])
        
    #     self.last_target_time = current_time
    def target_cb(self, msg):
        """Update target from TurtleBot3 odometry - NO PARTICLE FILTER"""
        pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        
        # SKIP PARTICLE FILTER - Use raw data directly
        # self.guidance.update_target_pose(position=pos)
        # self.guidance.guidance_filter_loop(t=current_time)
        
        self.target_pos = np.array(pos)
        
        # Estimate velocity
        if hasattr(msg.twist, 'twist'):
            v = msg.twist.twist.linear
        else:
            v = msg.twist.linear
        self.target_vel = np.array([v.x, v.y])
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        self.last_target_time = current_time

    def drone_odom_cb(self, msg):
        """Update drone velocity from odometry (if available)"""
        # Extract velocity from Odometry message
        v = msg.twist.twist.linear
        self.drone_vel = np.array([v.x, v.y, v.z])
    
    # ═══════════════════════════════════════════════════════════════
    # BOUNDARY SAFETY
    # ═══════════════════════════════════════════════════════════════
    
    def calculate_boundary_status(self, pos):
        """Calculate boundary proximity status"""
        if pos is None:
            return 'UNKNOWN', np.zeros(3), 999.0, 999.0
        
        x, y, z = pos
        
        dist_x_min = x - self.boundary_x_min
        dist_x_max = self.boundary_x_max - x
        dist_y_min = y - self.boundary_y_min
        dist_y_max = self.boundary_y_max - y
        dist_z_min = z - self.boundary_z_min
        dist_z_max = self.boundary_z_max - z
        
        min_dist = min(dist_x_min, dist_x_max, dist_y_min, dist_y_max, dist_z_min, dist_z_max)
        
        if min_dist < 0:
            status = 'VIOLATION'
        elif min_dist < self.emergency_distance:
            status = 'EMERGENCY'
        elif min_dist < self.warning_distance:
            status = 'WARNING'
        else:
            status = 'SAFE'
        
        direction = np.array([
            self.safe_center_x - x,
            self.safe_center_y - y,
            self.safe_center_z - z
        ])
        
        dist_to_center = np.linalg.norm(direction[:2])
        
        if dist_to_center > 0.01:
            direction_norm = direction / np.linalg.norm(direction)
        else:
            direction_norm = np.zeros(3)
        
        return status, direction_norm, min_dist, dist_to_center
    
    def apply_boundary_safety(self, rl_cmd):
        """Apply boundary safety to RL command"""
        if self.drone_pos is None:
            return Twist(), "NO_POSITION"
        
        status, dir_to_center, min_dist, dist_to_center = \
            self.calculate_boundary_status(self.drone_pos)
        
        safe_cmd = Twist()
        safe_cmd.linear.x = rl_cmd.linear.x
        safe_cmd.linear.y = rl_cmd.linear.y
        safe_cmd.linear.z = rl_cmd.linear.z
        safe_cmd.angular.z = rl_cmd.angular.z
        
        if status == 'VIOLATION' or status == 'EMERGENCY':
            if not self.in_emergency_return:
                self.get_logger().error(f"🚨 {status}! Forcing return to center!")
                self.in_emergency_return = True
                self.boundary_violations += 1
            
            return_speed = 0.6 if status == 'VIOLATION' else self.boundary_return_speed
            safe_cmd.linear.x = float(dir_to_center[0] * return_speed)
            safe_cmd.linear.y = float(dir_to_center[1] * return_speed)
            safe_cmd.linear.z = float(dir_to_center[2] * 0.3)
            safe_cmd.angular.z = 0.0
            
            return safe_cmd, f"{status} - RETURN TO CENTER"
        
        elif self.in_emergency_return:
            if dist_to_center < self.safe_zone_radius:
                self.get_logger().info("✅ Back in safe zone - resuming RL control")
                self.in_emergency_return = False
            else:
                safe_cmd.linear.x = float(dir_to_center[0] * self.boundary_return_speed)
                safe_cmd.linear.y = float(dir_to_center[1] * self.boundary_return_speed)
                safe_cmd.angular.z = 0.0
                return safe_cmd, "RETURNING TO SAFE ZONE"
        
        elif status == 'WARNING':
            x, y, z = self.drone_pos
            
            if x < (self.boundary_x_min + self.warning_distance) and safe_cmd.linear.x < 0:
                safe_cmd.linear.x *= 0.3
            elif x > (self.boundary_x_max - self.warning_distance) and safe_cmd.linear.x > 0:
                safe_cmd.linear.x *= 0.3
            
            if y < (self.boundary_y_min + self.warning_distance) and safe_cmd.linear.y < 0:
                safe_cmd.linear.y *= 0.3
            elif y > (self.boundary_y_max - self.warning_distance) and safe_cmd.linear.y > 0:
                safe_cmd.linear.y *= 0.3
            
            return safe_cmd, f"WARNING - Moderated (dist={min_dist:.2f}m)"
        
        return safe_cmd, f"SAFE (dist={min_dist:.2f}m)"
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN CONTROL LOOP
    # ═══════════════════════════════════════════════════════════════
    
    def control_loop(self):
        """Main RL control loop - BYPASSING PARTICLE FILTER"""
        
        cmd = Twist()
        
        # Altitude control
        if self.drone_pos is not None:
            err_z = self.target_height - self.drone_pos[2]
            cmd.linear.z = float(np.clip(err_z * 0.8, -0.5, 0.5))
        
        # Check prerequisites
        if self.drone_pos is None or self.drone_orient is None:
            self.vel_pub.publish(Twist())
            return
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_target_time is None or (current_time - self.last_target_time) > 2.0:
            self.get_logger().warn("No recent target data - hovering", throttle_duration_sec=2.0)
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            self.vel_pub.publish(cmd)
            return
        
        # Get RL action - BUILD DICT OBSERVATION (matches training format!)
        try:
            # Build agent state
            yaw = self.euler_from_quaternion(self.drone_orient)[2]
            
            # Agent observation: [x, y, vx, vy]
            agent_obs = np.array([
                self.drone_pos[0], 
                self.drone_pos[1],
                self.drone_vel[0], 
                self.drone_vel[1]
            ], dtype=np.float32)
            
            # Target observation (raw, no particle filter)
            if self.target_pos is None:
                self.vel_pub.publish(cmd)
                return
            
            # Calculate relative position/velocity in GLOBAL frame
            rel_x = self.target_pos[0] - self.drone_pos[0]
            rel_y = self.target_pos[1] - self.drone_pos[1]
            rel_vx = self.target_vel[0] - self.drone_vel[0]
            rel_vy = self.target_vel[1] - self.drone_vel[1]
            
            # Targets observation: [[rel_x, rel_y, rel_vx, rel_vy], ...]
            max_targets = 1
            targets_obs = np.zeros((max_targets, 4), dtype=np.float32)
            mask = np.zeros(max_targets, dtype=np.float32)
            
            # Fill first target (only 1 TurtleBot)
            targets_obs[0] = [rel_x, rel_y, rel_vx, rel_vy]
            mask[0] = 1.0  # Valid target
            
            # Build Dict observation (matches gym_pursuit_env.py format!)
            obs = {
                'agent': agent_obs,
                'targets': targets_obs,
                'mask': mask
            }
            
            # Get action from RL policy
            action, self.lstm_states = self.guidance.ppo_model.predict(
                obs,
                state=self.lstm_states,
                episode_start=self.episode_start,
                deterministic=True
            )
            
            self.episode_start = np.array([False])
            
            if action is not None:
                cmd.linear.x = float(np.clip(action[0], -self.max_vel, self.max_vel))
                cmd.linear.y = float(np.clip(action[1], -self.max_vel, self.max_vel))
                cmd.linear.z = 0.0
                #cmd.angular.z = float(np.clip(action[2], -self.max_yaw_rate, self.max_yaw_rate))
            
            # Debug logging
            if not hasattr(self, '_last_log') or (current_time - self._last_log) > 1.0:
                self.get_logger().info(
                    f"🎯 Drone: [{self.drone_pos[0]:.2f}, {self.drone_pos[1]:.2f}, {self.drone_pos[2]:.2f}] | "
                    f"TB3: [{self.target_pos[0]:.2f}, {self.target_pos[1]:.2f}] | "
                    f"Rel: [{rel_x:.2f}, {rel_y:.2f}] | "
                    f"Cmd: [{cmd.linear.x:.2f}, {cmd.linear.y:.2f}, {cmd.angular.z:.2f}]"
                )
                self._last_log = current_time
            
        except Exception as e:
            self.get_logger().error(f"RL policy error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0

        # Apply boundary safety
        #safe_cmd, status = self.apply_boundary_safety(cmd)
        
        # Publish
        self.vel_pub.publish(cmd)
        
        # Log
        # if not hasattr(self, '_last_log') or (current_time - self._last_log) > 1.0:
        #     self.get_logger().info(
        #         f"Status: {status} | "
        #         f"Pos: [{self.drone_pos[0]:.2f}, {self.drone_pos[1]:.2f}, {self.drone_pos[2]:.2f}] | "
        #         f"Cmd: [{safe_cmd.linear.x:.2f}, {safe_cmd.linear.y:.2f}, {safe_cmd.angular.z:.2f}]"
        #     )
        #     self._last_log = current_time
    
    @staticmethod
    def euler_from_quaternion(q):
        """Convert quaternion to euler angles"""
        x, y, z, w = q
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return 0.0, 0.0, yaw


def main(args=None):
    rclpy.init(args=args)
    node = RLGazeboTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.vel_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()