#!/usr/bin/env python3
"""
RL Gazebo Multi-Target Tracker
Tracks TWO TurtleBot3 robots in Gazebo using SJTU Drone
Extends rl_gazebo_tracker.py to handle multiple targets
"""
import sys
sys.path.append('/home/user/PPO_drone_Gazebo_ROS2/src/mml_python_sim')
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Bool
import numpy as np

#try:
#import numpy._core.numeric
# except ImportError:
#     print("Applying NumPy 2.0 -> 1.x Compatibility Patch...")
#     import numpy
#     if hasattr(numpy, 'core'):
#         sys.modules['numpy._core'] = numpy.core
#         sys.modules['numpy._core.numeric'] = numpy.core.numeric

from mml_guidance.guidance import Guidance
from mml_guidance.ParticleFilter import ParticleFilter


class RLGazeboMultiTracker(Node):
    """
    RL-based tracker for SJTU Drone tracking TWO TurtleBot3 robots
    
    KEY CHANGES FROM SINGLE TARGET:
    1. Two target subscriptions (tb3_1 and tb3_2)
    2. Separate state storage for each target
    3. Observation dict with targets array shape (2, 4)
    4. Mask array indicates which targets are valid
    """
    
    def __init__(self):
        super().__init__('rl_gazebo_multi_tracker')

        # ═══════════════════════════════════════════════════════════════
        # PARAMETERS (same as single target)
        # ═══════════════════════════════════════════════════════════════
        self.declare_parameter('model_path', '')
        self.declare_parameter('target1_topic', '/tb3_1/odom')  # ✅ NEW: First target
        self.declare_parameter('target2_topic', '/tb3_2/odom')  # ✅ NEW: Second target
        self.declare_parameter('drone_pose_topic', '/simple_drone/gt_pose')
        self.declare_parameter('drone_height', 3.0)
        self.declare_parameter('num_particles', 200)
        self.declare_parameter('prediction_method', 'Velocity')
        self.declare_parameter('max_vel', 0.8)
        self.declare_parameter('max_yaw_rate', 1.0)
        self.declare_parameter('max_tracked_targets', 10)
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        target1_topic = self.get_parameter('target1_topic').value  # ✅ NEW
        target2_topic = self.get_parameter('target2_topic').value  # ✅ NEW
        drone_pose_topic = self.get_parameter('drone_pose_topic').value
        self.target_height = self.get_parameter('drone_height').value
        self.max_vel = self.get_parameter('max_vel').value
        self.max_yaw_rate = self.get_parameter('max_yaw_rate').value
        self.max_tracked_targets = self.get_parameter('max_tracked_targets').value

        self.get_logger().info("=" * 60)
        self.get_logger().info("RL GAZEBO MULTI-TARGET TRACKER")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Model: {model_path}")
        self.get_logger().info(f"Target 1: {target1_topic}")  
        self.get_logger().info(f"Target 2: {target2_topic}")  
        self.get_logger().info(f"Drone pose: {drone_pose_topic}")
        self.get_logger().info(f"Target height: {self.target_height}m")
        self.get_logger().info(f"Max velocity: {self.max_vel}m/s")
        self.get_logger().info(f"Max tracked targets: {self.max_tracked_targets}")
        self.get_logger().info("=" * 60)
        
        # ═══════════════════════════════════════════════════════════════
        # INITIALIZE GUIDANCE SYSTEM (same as single target)
        # ═══════════════════════════════════════════════════════════════
        self.pf = ParticleFilter(
            num_particles=200,
            prediction_method='Velocity',
            drone_height=self.target_height
        )
        
        self.guidance = Guidance(
            guidance_mode="MultiPFInfoPPO_LSTM",
            prediction_method='Velocity',
            drone_height=self.target_height,
            filter=self.pf,
            ppo_model_path=model_path
        )
        
        self.lstm_states = None
        self.episode_start = np.array([True])
        
        # ═══════════════════════════════════════════════════════════════
        # STATE STORAGE - MODIFIED FOR TWO TARGETS
        # ═══════════════════════════════════════════════════════════════
        # Drone state (unchanged)
        self.drone_pos = None
        self.drone_orient = None
        self.drone_vel = np.array([0.0, 0.0, 0.0])
        self.last_drone_pos = None
        self.last_time = None
        
        # ✅ NEW: Target 1 state (TB3_1)
        self.target1_pos = None
        self.target1_vel = np.array([0.0, 0.0])
        self.last_target1_time = None
        
        # ✅ NEW: Target 2 state (TB3_2)
        self.target2_pos = None
        self.target2_vel = np.array([0.0, 0.0])
        self.last_target2_time = None
        
        # QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # ═══════════════════════════════════════════════════════════════
        # SUBSCRIBERS - MODIFIED FOR TWO TARGETS
        # ═══════════════════════════════════════════════════════════════
        # Drone subscribers (unchanged)
        self.sub_drone_pose = self.create_subscription(
            Pose, drone_pose_topic, self.drone_pose_cb, qos_profile)
        
        self.sub_drone_odom = self.create_subscription(
            Odometry, '/simple_drone/odom', self.drone_odom_cb, qos_profile
        )
        
        # ✅ NEW: Target 1 subscriber (TB3_1)
        self.sub_target1 = self.create_subscription(
            Odometry, target1_topic, self.target1_cb, qos_profile)
        
        # ✅ NEW: Target 2 subscriber (TB3_2)
        self.sub_target2 = self.create_subscription(
            Odometry, target2_topic, self.target2_cb, qos_profile)
        
        # ═══════════════════════════════════════════════════════════════
        # PUBLISHERS (unchanged from single target)
        # ═══════════════════════════════════════════════════════════════
        self.vel_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.posctrl_pub = self.create_publisher(Bool, '/simple_drone/posctrl', 10)
        
        # ═══════════════════════════════════════════════════════════════
        # TAKEOFF SEQUENCE (unchanged)
        # ═══════════════════════════════════════════════════════════════
        self.get_logger().info("Initiating takeoff sequence...")
        
        for _ in range(10):
            self.takeoff_pub.publish(Empty())
        
        posctrl_msg = Bool()
        posctrl_msg.data = False
        self.posctrl_pub.publish(posctrl_msg)
        
        # ═══════════════════════════════════════════════════════════════
        # START CONTROL LOOP (unchanged)
        # ═══════════════════════════════════════════════════════════════
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20Hz
        self.get_logger().info("RL Multi-Tracker active! Following 2 TurtleBot3s...")
    
    # ═══════════════════════════════════════════════════════════════
    # CALLBACKS - MODIFIED FOR TWO TARGETS
    # ═══════════════════════════════════════════════════════════════
    
    def drone_pose_cb(self, msg):
        """Update drone position (unchanged from single target)"""
        current_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.drone_orient = [msg.orientation.x, msg.orientation.y,
                            msg.orientation.z, msg.orientation.w]
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        if self.last_drone_pos is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0.001:
                self.drone_vel = (current_pos - self.last_drone_pos) / dt
        
        self.last_drone_pos = current_pos.copy()
        self.last_time = current_time
        self.drone_pos = current_pos
    
    def drone_odom_cb(self, msg):
        """Update drone velocity (unchanged)"""
        v = msg.twist.twist.linear
        self.drone_vel = np.array([v.x, v.y, v.z])
    
    # ✅ NEW: Target 1 callback
    def target1_cb(self, msg):
        """
        Update Target 1 (TB3_1) state from odometry
        
        WHAT THIS DOES:
        1. Extracts position [x, y] from odometry message
        2. Extracts velocity [vx, vy] from odometry message
        3. Records timestamp for staleness checking
        4. Stores in self.target1_pos and self.target1_vel
        """
        # Extract position
        pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.target1_pos = np.array(pos)
        
        # Extract velocity
        if hasattr(msg.twist, 'twist'):
            v = msg.twist.twist.linear
        else:
            v = msg.twist.linear
        self.target1_vel = np.array([v.x, v.y])
        
        # Update timestamp
        current_time = self.get_clock().now().nanoseconds / 1e9
        self.last_target1_time = current_time
    
    # ✅ NEW: Target 2 callback
    def target2_cb(self, msg):
        """
        Update Target 2 (TB3_2) state from odometry
        
        IDENTICAL LOGIC TO target1_cb, but stores to different variables:
        - self.target2_pos
        - self.target2_vel
        - self.last_target2_time
        """
        # Extract position
        pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.target2_pos = np.array(pos)
        
        # Extract velocity
        if hasattr(msg.twist, 'twist'):
            v = msg.twist.twist.linear
        else:
            v = msg.twist.linear
        self.target2_vel = np.array([v.x, v.y])
        
        # Update timestamp
        current_time = self.get_clock().now().nanoseconds / 1e9
        self.last_target2_time = current_time
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN CONTROL LOOP - MODIFIED FOR TWO TARGETS
    # ═══════════════════════════════════════════════════════════════
    
    def control_loop(self):
        """
        Main RL control loop with TWO targets
        
        ✅ FIX: Pad observation to (max_tracked_targets, 4) to match training
        """
        
        cmd = Twist()
        
        # Altitude control
        if self.drone_pos is not None:
            err_z = self.target_height - self.drone_pos[2]
            cmd.linear.z = float(np.clip(err_z * 0.8, -0.5, 0.5))
        
        # Check drone prerequisites
        if self.drone_pos is None or self.drone_orient is None:
            self.vel_pub.publish(Twist())
            return
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Check BOTH targets for staleness
        target1_stale = (self.last_target1_time is None or 
                        (current_time - self.last_target1_time) > 2.0)
        target2_stale = (self.last_target2_time is None or 
                        (current_time - self.last_target2_time) > 2.0)
        
        # If BOTH targets are stale, hover
        if target1_stale and target2_stale:
            self.get_logger().warn("No recent target data - hovering", 
                                  throttle_duration_sec=2.0)
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            self.vel_pub.publish(cmd)
            return
        
        # ═══════════════════════════════════════════════════════════════
        # BUILD OBSERVATION FOR RL POLICY - ✅ FIXED SHAPE
        # ═══════════════════════════════════════════════════════════════
        try:
            # ──────────────────────────────────────────────────────────
            # 1. BUILD AGENT OBSERVATION
            # ──────────────────────────────────────────────────────────
            agent_obs = np.array([
                self.drone_pos[0], 
                self.drone_pos[1],
                self.drone_vel[0], 
                self.drone_vel[1]
            ], dtype=np.float32)
            
            # ──────────────────────────────────────────────────────────
            # 2. BUILD TARGETS OBSERVATION - ✅ PADDED TO MAX SIZE
            # ──────────────────────────────────────────────────────────
            """
            ✅ KEY FIX: Initialize with max_tracked_targets slots (e.g., 10)
            
            BEFORE (broken):
                targets_obs = np.zeros((2, 4))  # Only 2 slots
            
            AFTER (correct):
                targets_obs = np.zeros((max_tracked_targets, 4))  # 10 slots
            
            This matches the training observation space:
                spaces.Box(shape=(10, 4), ...)
            """
            targets_obs = np.zeros((self.max_tracked_targets, 4), dtype=np.float32)
            mask = np.zeros(self.max_tracked_targets, dtype=np.float32)
            
            # ──────────────────────────────────────────────────────────
            # 2a. ADD TARGET 1 (TB3_1) if available
            # ──────────────────────────────────────────────────────────
            if not target1_stale and self.target1_pos is not None:
                # Calculate relative position (global frame)
                rel_x1 = self.target1_pos[0] - self.drone_pos[0]
                rel_y1 = self.target1_pos[1] - self.drone_pos[1]
                
                # Calculate relative velocity (global frame)
                rel_vx1 = self.target1_vel[0] - self.drone_vel[0]
                rel_vy1 = self.target1_vel[1] - self.drone_vel[1]
                
                # Fill first slot in targets array
                targets_obs[0] = [rel_x1, rel_y1, rel_vx1, rel_vy1]
                mask[0] = 1.0  # Mark as valid
            
            # ──────────────────────────────────────────────────────────
            # 2b. ADD TARGET 2 (TB3_2) if available
            # ──────────────────────────────────────────────────────────
            if not target2_stale and self.target2_pos is not None:
                # Calculate relative position (global frame)
                rel_x2 = self.target2_pos[0] - self.drone_pos[0]
                rel_y2 = self.target2_pos[1] - self.drone_pos[1]
                
                # Calculate relative velocity (global frame)
                rel_vx2 = self.target2_vel[0] - self.drone_vel[0]
                rel_vy2 = self.target2_vel[1] - self.drone_vel[1]
                
                # Fill second slot in targets array
                targets_obs[1] = [rel_x2, rel_y2, rel_vx2, rel_vy2]
                mask[1] = 1.0  # Mark as valid
            
            # ──────────────────────────────────────────────────────────
            # 2c. REMAINING SLOTS (2-9) ARE ALREADY ZEROS WITH MASK=0
            # ──────────────────────────────────────────────────────────
            """
            Slots 2-9 remain as zeros with mask=0.0
            The attention mechanism will ignore these via masking
            
            Final state:
                targets_obs[0] = [rel_x1, rel_y1, rel_vx1, rel_vy1]  ← Valid (mask=1)
                targets_obs[1] = [rel_x2, rel_y2, rel_vx2, rel_vy2]  ← Valid (mask=1)
                targets_obs[2] = [0, 0, 0, 0]                        ← Padding (mask=0)
                targets_obs[3] = [0, 0, 0, 0]                        ← Padding (mask=0)
                ...
                targets_obs[9] = [0, 0, 0, 0]                        ← Padding (mask=0)
            """
            
            # ──────────────────────────────────────────────────────────
            # 3. BUILD DICT OBSERVATION (matches training format)
            # ──────────────────────────────────────────────────────────
            obs = {
                'agent': agent_obs,          # shape (4,)
                'targets': targets_obs,      # shape (10, 4) ✅ FIXED
                'mask': mask                 # shape (10,)   ✅ FIXED
            }
            
            # ──────────────────────────────────────────────────────────
            # 4. GET ACTION FROM RL POLICY
            # ──────────────────────────────────────────────────────────
            action, self.lstm_states = self.guidance.ppo_model.predict(
                obs,
                state=self.lstm_states,
                episode_start=self.episode_start,
                deterministic=True
            )
            
            self.episode_start = np.array([False])
            
            # ──────────────────────────────────────────────────────────
            # 5. CONVERT ACTION TO VELOCITY COMMANDS
            # ──────────────────────────────────────────────────────────
            if action is not None:
                cmd.linear.x = float(np.clip(action[0], -self.max_vel, self.max_vel))
                cmd.linear.y = float(np.clip(action[1], -self.max_vel, self.max_vel))
                cmd.linear.z = 0.0  # Altitude handled separately
            
            # ──────────────────────────────────────────────────────────
            # 6. DEBUG LOGGING
            # ──────────────────────────────────────────────────────────
            if not hasattr(self, '_last_log') or (current_time - self._last_log) > 1.0:
                # Build target info string
                t1_str = f"TB3_1: [{self.target1_pos[0]:.2f}, {self.target1_pos[1]:.2f}]" if not target1_stale else "TB3_1: STALE"
                t2_str = f"TB3_2: [{self.target2_pos[0]:.2f}, {self.target2_pos[1]:.2f}]" if not target2_stale else "TB3_2: STALE"
                
                valid_count = int(mask.sum())
                
                self.get_logger().info(
                    f"🎯 Drone: [{self.drone_pos[0]:.2f}, {self.drone_pos[1]:.2f}, {self.drone_pos[2]:.2f}] | "
                    f"{t1_str} | {t2_str} | "
                    f"Valid targets: {valid_count}/{self.max_tracked_targets} | "
                    f"Cmd: [{cmd.linear.x:.2f}, {cmd.linear.y:.2f}]"
                )
                self._last_log = current_time
            
        except Exception as e:
            self.get_logger().error(f"RL policy error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
        
        # ══════════════════════════════════════════════════════════════
        # 7. PUBLISH COMMAND
        # ══════════════════════════════════════════════════════════════
        self.vel_pub.publish(cmd)
    
    @staticmethod
    def euler_from_quaternion(q):
        """Convert quaternion to euler angles (unchanged)"""
        x, y, z, w = q
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return 0.0, 0.0, yaw


def main(args=None):
    rclpy.init(args=args)
    node = RLGazeboMultiTracker()  # ✅ Use new node class
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
