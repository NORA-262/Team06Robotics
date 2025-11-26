#!/usr/bin/env python3
"""
move_up.py
ROS 2 + CustomTkinter controller for a 6-DOF low-cost robot arm.

Features:
- Forward kinematics (symbolic, SymPy → NumPy)
- Publishes joint angles to /joint_commands
- Publishes end-effector position to /ee_position
- Interactive GUI with sliders
- Optional wave motion animation
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import customtkinter as ctk
import numpy as np
import sympy as sp
import math, time, threading

# ============================================================
# ⚙️ 1. Constants and symbolic FK setup
# ============================================================

JOINT_COUNT = 6  # number of joints / sliders

# Publish and motion tuning (adjust to increase responsiveness)
PUBLISH_HZ = 50               # default publish rate (Hz). Was 20 Hz.
WAVE_AMPLITUDE = 0.8          # wave motion amplitude (radians)
WAVE_FREQ_BASE = 1.0          # base angular frequency (rad/s)
WAVE_PHASE_SCALE = 0.4        # per-joint frequency scaling

# PD Controller Gains
KP = 50.0  # Proportional gain
KD = 1.0   # Derivative gain

# Velocity control parameters
TARGET_VELOCITY = 0.1  # m/s

# Acceleration control parameters
TARGET_ACCELERATION = 0.2  # m/s^2
MAX_TARGET_ACCELERATION = 0.5  # m/s^2, adjust based on system limits

# symbolic joint vars for all 6 joints
θ1, θ2, θ3, θ4, θ5, θ6 = sp.symbols('θ1 θ2 θ3 θ4 θ5 θ6')

# simplified symbolic forward-kinematics model (replace constants with your DH table)
T1 = sp.Matrix([ # Joint 1: Base Rotation
    [sp.cos(θ1), -sp.sin(θ1), 0, 0],
    [sp.sin(θ1),  sp.cos(θ1), 0, 0],
    [0,            0,          1, 0.04],
    [0,            0,          0, 1]
])
T2 = sp.Matrix([ # Joint 2: Shoulder Pitch
    [sp.cos(θ2), 0, sp.sin(θ2), 0],
    [0,          1, 0,           0],
    [-sp.sin(θ2),0, sp.cos(θ2),  0],
    [0,          0, 0,           1]
])
T3 = sp.Matrix([ # Joint 3: Elbow
    [sp.cos(θ3), 0, sp.sin(θ3), 0.108*sp.cos(θ3)],
    [0,          1, 0,           0],
    [-sp.sin(θ3),0, sp.cos(θ3),  0.108*sp.sin(θ3)],
    [0,          0, 0,           1]
])
T4 = sp.Matrix([ # Joint 4: Wrist Pitch
    [sp.cos(θ4), 0, sp.sin(θ4), 0.1*sp.cos(θ4)],
    [0,          1, 0,           0],
    [-sp.sin(θ4),0, sp.cos(θ4),  0.1*sp.sin(θ4)],
    [0,          0, 0,           1]
])
T5 = sp.Matrix([ # Joint 5: Wrist Roll
    [sp.cos(θ5), -sp.sin(θ5), 0, 0.045*sp.cos(θ5)],
    [sp.sin(θ5),  sp.cos(θ5), 0, 0.045*sp.sin(θ5)],
    [0,            0,          1, 0],
    [0,            0,          0, 1]
])

T6 = sp.Matrix([ # Joint 6: Gripper
    [sp.cos(θ6), -sp.sin(θ6), 0, 0],
    [sp.sin(θ6),  sp.cos(θ6), 0, 0],
    [0,            0,          1, 0.0145], # Assuming a simple translation for the gripper
    [0,            0,          0, 1]
])

T_0_5 = T1 * T2 * T3 * T4 * T5
T_0_6 = T1 * T2 * T3 * T4 * T5 * T6

# --- Velocity Kinematics (Jacobian) ---
# Extract the position vector and rotation matrix from the transformation
P = T_0_6[:3, 3]
R = T_0_6[:3, :3]

# Calculate the linear part of the Jacobian by differentiating the position vector
J_linear = P.jacobian([θ1, θ2, θ3, θ4, θ5, θ6])

# The angular part of the Jacobian is composed of the z-axes of each joint's frame
z_axes = [
    T1[:3, 2],
    (T1*T2)[:3, 2],
    (T1*T2*T3)[:3, 2],
    (T1*T2*T3*T4)[:3, 2],
    (T1*T2*T3*T4*T5)[:3, 2],
    (T1*T2*T3*T4*T5*T6)[:3, 2]
]
J_angular = sp.Matrix.hstack(*z_axes)

# Combine into the full 6x6 Jacobian
Jacobian = sp.Matrix.vstack(J_linear, J_angular)

# --- Acceleration Kinematics (Jacobian Derivative) ---
q_sym = [θ1, θ2, θ3, θ4, θ5, θ6]
q_dot_syms = sp.symbols('θ_dot1 θ_dot2 θ_dot3 θ_dot4 θ_dot5 θ_dot6')

# Build symbolic J_dot = dJ/dt = sum_k (dJ/dq_k * q_dot_k)
J_dot = sp.zeros(6, 6)
for i in range(Jacobian.rows):
    for j in range(Jacobian.cols):
        J_dot[i, j] = sum(sp.diff(Jacobian[i, j], q_sym[k]) * q_dot_syms[k] for k in range(6))

# Create fast numerical function for J_dot: (θ1..θ6, θ_dot1..θ_dot6) -> 6x6
j_dot_numeric = sp.lambdify((*q_sym, *q_dot_syms), J_dot, "numpy")

# Create fast numerical functions for FK and Jacobian
fk_numeric = sp.lambdify((θ1, θ2, θ3, θ4, θ5, θ6), T_0_6, "numpy")
jacobian_numeric = sp.lambdify((θ1, θ2, θ3, θ4, θ5, θ6), Jacobian, "numpy")

def forward_kinematics(joint_values):
    j = joint_values[:6]
    T = np.array(fk_numeric(*j), dtype=float)
    return T[:3, 3], T[:3, :3]  # Return position and rotation

# ============================================================
# 🤖 2. Node + GUI
# ============================================================

class ArmMoverGUI(Node):
    def __init__(self):
        super().__init__("move_up")

        # Publishers
        self.joint_pub = self.create_publisher(Float64MultiArray, "joint_commands", 10)
        self.commanded_joint_pub = self.create_publisher(Float64MultiArray, "commanded_joint_positions", 10)
        self.ee_pub = self.create_publisher(Float64MultiArray, "ee_position", 10)
        self.target_path_pub = self.create_publisher(Path, "target_path", 10)
        self.actual_path_pub = self.create_publisher(Path, "actual_path", 10)

        # Subscriber for joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            "joint_state",
            self.joint_state_callback,
            10)
        self.current_joint_positions = np.zeros(JOINT_COUNT)
        self.current_joint_velocities = np.zeros(JOINT_COUNT)
        self.joint_state_lock = threading.Lock()

        # Velocity control state
        self.velocity_command = np.zeros(6) # [vx, vy, vz, wx, wy, wz]
        self.velocity_mode = False

        # Acceleration control state
        self.acceleration_command = np.zeros(6)  # [ax, ay, az, ang_ax, ang_ay, ang_az]
        self.acceleration_mode = False
        self.accel_sliders = []
        self.accel_value_labels = []

        # Trajectory following state
        self.trajectory_mode = False
        self.trajectory_points = []
        self.current_trajectory_index = 0
        self.trajectory_start_pos = np.zeros(3)
        self.trajectory_end_pos = np.zeros(3)
        self.trajectory_start_entries = []
        self.trajectory_end_entries = []
        self.trajectory_duration_entry = None

        # Path messages for RViz
        self.target_path_msg = Path()
        self.actual_path_msg = Path()

        # Joint-space trajectory state
        self.joint_trajectory_mode = False
        self.joint_trajectory_points = []
        self.current_joint_trajectory_index = 0
        self.joint_trajectory_start_entries = []
        self.joint_trajectory_end_entries = []
        self.joint_trajectory_duration_entry = None

        # Timing & flags
        self.start_time = time.time()
        self.wave_active = False
        self.gui_ready = False
        self.sliders = []
        self.value_labels = []

        # Timing state
        self.last_log_time = 0.0

        # Start GUI in separate thread
        gui_thread = threading.Thread(target=self.build_gui, daemon=True)
        gui_thread.start()

        # Timer (default: PUBLISH_HZ)
        self.create_timer(1.0 / PUBLISH_HZ, self.safe_publish_command)

        self.get_logger().info("Arm Mover GUI Node started.")

    # --------------------------------------------------------
    def safe_publish_command(self):
        if self.gui_ready:
            self.publish_command()

    # --------------------------------------------------------
    def publish_command(self):
        msg = Float64MultiArray()
        target_joints = []

        if self.joint_trajectory_mode:
            # --- JOINT-SPACE TRAJECTORY FOLLOWING ---
            if self.current_joint_trajectory_index < len(self.joint_trajectory_points):
                # Set the target for the main PD controller
                target_joints = self.joint_trajectory_points[self.current_joint_trajectory_index]

                # Update sliders to reflect the target
                for i, sld in enumerate(self.sliders):
                    sld.set(target_joints[i])

                self.current_joint_trajectory_index += 1
            else:
                self.toggle_joint_trajectory_mode() # Finished
                return # Exit to avoid using empty target_joints in this cycle

        elif self.trajectory_mode:
            # --- TRAJECTORY FOLLOWING ---
            if self.current_trajectory_index < len(self.trajectory_points):
                target_pos = self.trajectory_points[self.current_trajectory_index]
                
                with self.joint_state_lock:
                    current_pos, _ = forward_kinematics(self.current_joint_positions)
                    dt = 1.0 / PUBLISH_HZ
                    
                    # Calculate required velocity to reach the next point in one time step
                    required_velocity = (target_pos - current_pos) / dt
                    
                    self.velocity_command = np.zeros(6)
                    self.velocity_command[:3] = required_velocity

                    # Use velocity controller logic to get target joint positions
                    q = np.array(self.current_joint_positions, dtype=float)
                    J = np.array(jacobian_numeric(*q), dtype=float)
                    J_pinv = np.linalg.pinv(J)
                    target_joint_velocities = J_pinv @ self.velocity_command
                    target_joints = (q + target_joint_velocities * dt)

                self.current_trajectory_index += 1
            else:
                self.toggle_trajectory_mode() # Stop if trajectory is empty or finished
                return # Exit to avoid using empty target_joints

        elif self.acceleration_mode:
            # --- ACCELERATION CONTROL ---
            with self.joint_state_lock:
                # Ensure we work with float arrays
                q = np.array(self.current_joint_positions, dtype=float)
                q_dot = np.array(self.current_joint_velocities, dtype=float)

                # Evaluate numeric Jacobian and its derivative
                J = np.array(jacobian_numeric(*q), dtype=float)
                J_pinv = np.linalg.pinv(J)

                J_dot_val = np.array(j_dot_numeric(*q, *q_dot), dtype=float)

                # Target end-effector acceleration (from sliders) - ensure float and length 6
                x_ddot = np.array(self.acceleration_command, dtype=float)

                # Solve for joint accelerations (use pseudo-inverse of J)
                q_ddot = J_pinv @ (x_ddot - (J_dot_val @ q_dot))

                # Integrate to get target joint positions
                dt = 1.0 / PUBLISH_HZ
                target_joint_velocities = q_dot + q_ddot * dt
                target_joints = (q + target_joint_velocities * dt)

                # Update sliders to reflect the new target
                for i, sld in enumerate(self.sliders):
                    sld.set(target_joints[i])

        elif self.velocity_mode:
            # --- VELOCITY CONTROL ---
            with self.joint_state_lock:
                q = np.array(self.current_joint_positions, dtype=float)
                J = np.array(jacobian_numeric(*q), dtype=float)
                
                # Use the pseudo-inverse for robust control
                J_pinv = np.linalg.pinv(J)
                
                # Calculate target joint velocities
                target_joint_velocities = J_pinv @ self.velocity_command
                
                # Integrate to get target joint positions
                dt = 1.0 / PUBLISH_HZ
                target_joints = (q + target_joint_velocities * dt)
                
                # Update sliders to reflect the new target
                for i, sld in enumerate(self.sliders):
                    sld.set(target_joints[i])

        elif self.wave_active:
            # --- WAVE MOTION ---
            t = time.time() - self.start_time
            for i in range(JOINT_COUNT):
                val = WAVE_AMPLITUDE * math.sin(t * (WAVE_FREQ_BASE + i * WAVE_PHASE_SCALE))
                self.sliders[i].set(val)
                target_joints.append(val)
        
        elif self.gui_ready:
            # --- MANUAL SLIDER CONTROL (FALLBACK) ---
            target_joints = [s.get() for s in self.sliders]

        # This check is necessary because some modes might not set target_joints immediately.
        # We check if the variable is a list and is empty, or if it's a numpy array and its size is 0.
        is_empty = (isinstance(target_joints, list) and not target_joints) or \
                   (hasattr(target_joints, 'size') and target_joints.size == 0)

        if is_empty:
             if self.gui_ready:
                # Fallback to current slider values if no other mode is active
                target_joints = [s.get() for s in self.sliders]
             else:
                return # Nothing to do if GUI isn't ready and no mode is active

        # Publish the commanded joint positions for the plotter
        commanded_msg = Float64MultiArray()
        if len(target_joints) == JOINT_COUNT:
            commanded_msg.data = np.array(target_joints, dtype=float).tolist()
            self.commanded_joint_pub.publish(commanded_msg)

        # PD Controller
        with self.joint_state_lock:
            position_error = np.array(target_joints) - self.current_joint_positions
            # In torque mode, we send torques. The mujoco_node adds gravity compensation.
            torques = KP * position_error - KD * self.current_joint_velocities
            msg.data = torques.tolist()
            self.joint_pub.publish(msg)

        # FK on all 6 joints
        try:
            if len(target_joints) == JOINT_COUNT:
                pos, rot = forward_kinematics(target_joints)
                
                # Publish current EE position array
                ee_msg = Float64MultiArray()
                ee_msg.data = pos.tolist()
                self.ee_pub.publish(ee_msg)

                # Append to and publish the actual path for RViz
                self.actual_path_msg.header.stamp = self.get_clock().now().to_msg()
                self.actual_path_msg.header.frame_id = "world" # Or your base frame
                pose = PoseStamped()
                pose.header = self.actual_path_msg.header
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = pos
                self.actual_path_msg.poses.append(pose)
                self.actual_path_pub.publish(self.actual_path_msg)

                # log every ~1 s (avoid spamming at high publish rates)
                now = time.time()
                if now - self.last_log_time >= 1.0:
                    self.get_logger().info(f"EE pos: {pos}")
                    self.last_log_time = now
        except Exception as e:
            self.get_logger().warn(f"FK error: {e}")

    # --------------------------------------------------------
    def joint_state_callback(self, msg):
        with self.joint_state_lock:
            # Assuming the names in the JointState message match the order of our joints
            self.current_joint_positions = np.array(msg.position)
            self.current_joint_velocities = np.array(msg.velocity)

    # --------------------------------------------------------
    def build_gui(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        app = ctk.CTk()
        app.title("6-DOF Arm Controller")
        app.geometry("640x900") # Adjusted for scrollbar

        # Create a scrollable frame to hold all content
        scrollable_main_frame = ctk.CTkScrollableFrame(app)
        scrollable_main_frame.pack(pady=10, padx=10, fill="both", expand=True)

        title = ctk.CTkLabel(scrollable_main_frame, text="Low-Cost Robot Arm Controller",
                             font=("Arial", 22, "bold"))
        title.pack(pady=10)

        frame = ctk.CTkFrame(scrollable_main_frame)
        frame.pack(pady=10)

        for i in range(JOINT_COUNT):
            row = ctk.CTkFrame(frame)
            row.pack(pady=3, fill="x")

            lbl = ctk.CTkLabel(row, text=f"Joint {i+1}", width=80)
            lbl.pack(side="left", padx=5)

            sld = ctk.CTkSlider(row, from_=-math.pi, to=math.pi, width=320)
            sld.set(0.0)
            sld.pack(side="left", padx=5)
            self.sliders.append(sld)

            val_lbl = ctk.CTkLabel(row, text="0.00", width=60)
            val_lbl.pack(side="left", padx=5)
            self.value_labels.append(val_lbl)

            def make_callback(lbl_ref, sld_ref):
                def _cb(value):
                    lbl_ref.configure(text=f"{value:+.2f}")
                return _cb

            sld.configure(command=make_callback(val_lbl, sld))

        # Velocity Control Buttons
        vel_frame = ctk.CTkFrame(scrollable_main_frame)
        vel_frame.pack(pady=10)

        ctk.CTkLabel(vel_frame, text="Velocity Control", font=("Arial", 16, "bold")).pack(pady=5)

        axes = ['X', 'Y', 'Z']
        for i, axis in enumerate(axes):
            row = ctk.CTkFrame(vel_frame)
            row.pack(pady=2)
            
            btn_plus = ctk.CTkButton(row, text=f"+{axis}", width=60)
            btn_plus.pack(side="left", padx=5)
            btn_plus.bind("<ButtonPress-1>", lambda e, a=i, d=1: self.start_velocity_control(a, d))
            btn_plus.bind("<ButtonRelease-1>", lambda e: self.stop_velocity_control())

            btn_minus = ctk.CTkButton(row, text=f"-{axis}", width=60)
            btn_minus.pack(side="left", padx=5)
            btn_minus.bind("<ButtonPress-1>", lambda e, a=i, d=-1: self.start_velocity_control(a, d))
            btn_minus.bind("<ButtonRelease-1>", lambda e: self.stop_velocity_control())

        # Acceleration Control Sliders
        accel_frame = ctk.CTkFrame(scrollable_main_frame)
        accel_frame.pack(pady=10, padx=10, fill="x")

        ctk.CTkLabel(accel_frame, text="Acceleration Control", font=("Arial", 16, "bold")).pack(pady=5)

        accel_axes = ['Accel X', 'Accel Y', 'Accel Z']
        for i, axis_label in enumerate(accel_axes):
            row = ctk.CTkFrame(accel_frame)
            row.pack(pady=3, fill="x")

            lbl = ctk.CTkLabel(row, text=axis_label, width=80)
            lbl.pack(side="left", padx=5)

            sld = ctk.CTkSlider(row, from_=-TARGET_ACCELERATION, to=TARGET_ACCELERATION, width=320)
            sld.set(0.0)
            sld.pack(side="left", padx=5)
            self.accel_sliders.append(sld)

            val_lbl = ctk.CTkLabel(row, text="0.00", width=60)
            val_lbl.pack(side="left", padx=5)
            self.accel_value_labels.append(val_lbl)

            def make_accel_callback(lbl_ref, axis_idx):
                def _cb(value):
                    lbl_ref.configure(text=f"{value:+.2f}")
                    self.acceleration_command[axis_idx] = value
                return _cb

            sld.configure(command=make_accel_callback(val_lbl, i))

        self.btn_accel_mode = ctk.CTkButton(accel_frame, text="Start Acceleration Mode", command=self.toggle_acceleration_mode)
        self.btn_accel_mode.pack(pady=10)

        # Trajectory Control
        traj_frame = ctk.CTkFrame(scrollable_main_frame)
        traj_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(traj_frame, text="Task-Space Trajectory", font=("Arial", 16, "bold")).pack(pady=5)

        # Start and End point entries
        for label_text, entries_list, pos_variable in [
            ("Start Point (X,Y,Z)", self.trajectory_start_entries, self.trajectory_start_pos),
            ("End Point (X,Y,Z)", self.trajectory_end_entries, self.trajectory_end_pos)
        ]:
            row = ctk.CTkFrame(traj_frame)
            row.pack(pady=2, fill='x')
            ctk.CTkLabel(row, text=label_text, width=140).pack(side="left", padx=5)
            for i in range(3):
                entry = ctk.CTkEntry(row, width=70)
                entry.insert(0, "0.0")
                entry.pack(side="left", padx=3)
                entries_list.append(entry)
            
            btn_set_current = ctk.CTkButton(row, text="Set Current", width=80,
                                            command=lambda p=pos_variable, e=entries_list: self.set_current_pos_for_trajectory(p, e))
            btn_set_current.pack(side="left", padx=5)

        # Duration entry
        duration_row = ctk.CTkFrame(traj_frame)
        duration_row.pack(pady=2, fill='x')
        ctk.CTkLabel(duration_row, text="Duration (s)", width=140).pack(side="left", padx=5)
        self.trajectory_duration_entry = ctk.CTkEntry(duration_row, width=70)
        self.trajectory_duration_entry.insert(0, "2.0")
        self.trajectory_duration_entry.pack(side="left", padx=3)

        self.btn_traj_mode = ctk.CTkButton(traj_frame, text="Execute Trajectory", command=self.toggle_trajectory_mode)
        self.btn_traj_mode.pack(pady=10)

        # Joint-Space Trajectory Control
        joint_traj_frame = ctk.CTkFrame(scrollable_main_frame)
        joint_traj_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(joint_traj_frame, text="Joint-Space Trajectory", font=("Arial", 16, "bold")).pack(pady=5)

        # Start and End angle entries
        for label_text, entries_list in [
            ("Start Angles", self.joint_trajectory_start_entries),
            ("End Angles", self.joint_trajectory_end_entries)
        ]:
            row = ctk.CTkFrame(joint_traj_frame)
            row.pack(pady=2, fill='x')
            ctk.CTkLabel(row, text=label_text, width=100).pack(side="left", padx=5)
            for i in range(JOINT_COUNT):
                entry = ctk.CTkEntry(row, width=60)
                entry.insert(0, "0.0")
                entry.pack(side="left", padx=2)
                entries_list.append(entry)
            
            btn_set_current = ctk.CTkButton(row, text="Set Current", width=80,
                                            command=lambda e=entries_list: self.set_current_angles_for_joint_trajectory(e))
            btn_set_current.pack(side="left", padx=5)

        # Duration entry for Joint-Space
        joint_duration_row = ctk.CTkFrame(joint_traj_frame)
        joint_duration_row.pack(pady=2, fill='x')
        ctk.CTkLabel(joint_duration_row, text="Duration (s)", width=100).pack(side="left", padx=5)
        self.joint_trajectory_duration_entry = ctk.CTkEntry(joint_duration_row, width=60)
        self.joint_trajectory_duration_entry.insert(0, "2.0")
        self.joint_trajectory_duration_entry.pack(side="left", padx=2)

        self.btn_joint_traj_mode = ctk.CTkButton(joint_traj_frame, text="Execute Joint Trajectory", command=self.toggle_joint_trajectory_mode)
        self.btn_joint_traj_mode.pack(pady=10)

        # Buttons
        btn_frame = ctk.CTkFrame(scrollable_main_frame)
        btn_frame.pack(pady=20)

        self.btn_send = ctk.CTkButton(btn_frame, text="Send Once", command=self.send_once)
        self.btn_send.grid(row=0, column=0, padx=10)

        self.btn_wave = ctk.CTkButton(btn_frame, text="Start Wave Motion", fg_color="blue",
                                      command=self.toggle_wave)
        self.btn_wave.grid(row=0, column=1, padx=10)

        self.btn_quit = ctk.CTkButton(btn_frame, text="Quit", fg_color="red",
                                      command=lambda: self.close_gui(app))
        self.btn_quit.grid(row=0, column=2, padx=10)

        footer = ctk.CTkLabel(scrollable_main_frame,
            text="Publishes: /joint_commands and /ee_position (50 Hz)",
            font=("Arial", 12))
        footer.pack(pady=10)

        self.gui_ready = True
        app.mainloop()

    # --------------------------------------------------------
    def send_once(self):
        joints = [s.get() for s in self.sliders]
        msg = Float64MultiArray()
        msg.data = joints
        self.joint_pub.publish(msg)

        try:
            pos, rot = forward_kinematics(joints)
            ee_msg = Float64MultiArray()
            ee_msg.data = pos.tolist()
            self.ee_pub.publish(ee_msg)
            self.get_logger().info(f"Sent once — joints: {np.round(joints,2)}  EE pos: {pos}")
        except Exception as e:
            self.get_logger().error(f"Send once FK error: {e}")

    # --------------------------------------------------------
    def toggle_wave(self):
        self.wave_active = not self.wave_active
        if self.wave_active:
            self.start_time = time.time()
            self.btn_wave.configure(text="Stop Wave Motion", fg_color="orange")
        else:
            self.btn_wave.configure(text="Start Wave Motion", fg_color="blue")

    # --------------------------------------------------------
    def start_velocity_control(self, axis, direction):
        # Deactivate other modes
        self.acceleration_mode = False
        self.wave_active = False
        self.btn_accel_mode.configure(text="Start Acceleration Mode", fg_color="blue")
        self.btn_wave.configure(text="Start Wave Motion", fg_color="blue")

        self.velocity_mode = True
        self.velocity_command = np.zeros(6)
        self.velocity_command[axis] = direction * TARGET_VELOCITY
        self.get_logger().info(f"Starting velocity control: axis={axis}, direction={direction}")

    def stop_velocity_control(self):
        self.velocity_mode = False
        self.velocity_command = np.zeros(6)
        self.get_logger().info("Stopping velocity control.")

    # --------------------------------------------------------
    def toggle_joint_trajectory_mode(self):
        # If trajectory is running, the button press should stop it.
        if self.joint_trajectory_mode:
            self.joint_trajectory_mode = False
            self.btn_joint_traj_mode.configure(text="Execute Joint Trajectory", fg_color="blue")
            self.joint_trajectory_points = []
            self.current_joint_trajectory_index = 0
            self.get_logger().info("Joint trajectory manually stopped.")
            return

        # If trajectory is not running, the button press should start it.
        self.joint_trajectory_mode = True
        # Deactivate other modes
        self.wave_active = False
        self.velocity_mode = False
        self.acceleration_mode = False
        self.trajectory_mode = False
        self.btn_wave.configure(text="Start Wave Motion", fg_color="blue")
        self.btn_accel_mode.configure(text="Start Acceleration Mode")
        self.btn_traj_mode.configure(text="Execute Trajectory", fg_color="blue")

        # Get start and end angles from GUI
        try:
            start_angles = np.array([float(e.get()) for e in self.joint_trajectory_start_entries])
            end_angles = np.array([float(e.get()) for e in self.joint_trajectory_end_entries])
            duration = float(self.joint_trajectory_duration_entry.get())
            if duration <= 0:
                self.get_logger().error("Duration must be a positive number.")
                self.joint_trajectory_mode = False
                return
        except (ValueError, TypeError):
            self.get_logger().error("Invalid joint trajectory angles or duration. Please enter numbers.")
            self.joint_trajectory_mode = False
            return

        self.joint_trajectory_points = self.generate_joint_space_trajectory(start_angles, end_angles, duration)
        self.current_joint_trajectory_index = 0
        
        self.btn_joint_traj_mode.configure(text="Stop Joint Trajectory", fg_color="orange")
        self.get_logger().info(f"Starting joint trajectory over {duration}s.")

    def generate_joint_space_trajectory(self, start_angles, end_angles, duration):
        num_points = int(duration * PUBLISH_HZ)
        return np.linspace(start_angles, end_angles, num_points)

    def set_current_angles_for_joint_trajectory(self, entries_list):
        with self.joint_state_lock:
            current_angles = self.current_joint_positions
        
        for i in range(JOINT_COUNT):
            entries_list[i].delete(0, 'end')
            entries_list[i].insert(0, f"{current_angles[i]:.3f}")
        self.get_logger().info(f"Set joint trajectory point to current angles: {current_angles}")

    # --------------------------------------------------------
    def toggle_trajectory_mode(self):
        self.trajectory_mode = not self.trajectory_mode
        if self.trajectory_mode:
            # Deactivate other modes
            self.wave_active = False
            self.velocity_mode = False
            self.acceleration_mode = False
            self.joint_trajectory_mode = False
            self.btn_wave.configure(text="Start Wave Motion", fg_color="blue")
            self.btn_accel_mode.configure(text="Start Acceleration Mode")
            self.btn_joint_traj_mode.configure(text="Execute Joint Trajectory", fg_color="blue")

            # Get start and end points from GUI
            try:
                start_pos = np.array([float(e.get()) for e in self.trajectory_start_entries])
                end_pos = np.array([float(e.get()) for e in self.trajectory_end_entries])
                duration = float(self.trajectory_duration_entry.get())
            except (ValueError, TypeError):
                self.get_logger().error("Invalid trajectory coordinates or duration. Please enter numbers.")
                self.trajectory_mode = False
                return

            self.trajectory_points = self.generate_linear_trajectory(start_pos, end_pos, duration)
            self.current_trajectory_index = 0

            # Prepare and publish the target path for RViz
            self.target_path_msg = Path()
            self.target_path_msg.header.stamp = self.get_clock().now().to_msg()
            self.target_path_msg.header.frame_id = "world" # Or your base frame
            for point in self.trajectory_points:
                pose = PoseStamped()
                pose.header = self.target_path_msg.header
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = point
                self.target_path_msg.poses.append(pose)
            self.target_path_pub.publish(self.target_path_msg)

            # Clear the previous actual path
            self.actual_path_msg = Path()
            
            self.btn_traj_mode.configure(text="Stop Trajectory", fg_color="orange")
            self.get_logger().info(f"Starting trajectory from {start_pos} to {end_pos} over {duration}s.")
        else:
            self.btn_traj_mode.configure(text="Execute Trajectory", fg_color="blue")
            self.trajectory_points = []
            self.current_trajectory_index = 0
            self.get_logger().info("Trajectory mode disabled.")

    def generate_linear_trajectory(self, start_pos, end_pos, duration):
        num_points = int(duration * PUBLISH_HZ)
        return np.linspace(start_pos, end_pos, num_points)

    def set_current_pos_for_trajectory(self, pos_variable, entries_list):
        with self.joint_state_lock:
            current_pos, _ = forward_kinematics(self.current_joint_positions)
        
        pos_variable[:] = current_pos
        for i in range(3):
            entries_list[i].delete(0, 'end')
            entries_list[i].insert(0, f"{current_pos[i]:.3f}")
        self.get_logger().info(f"Set trajectory point to current EE pos: {current_pos}")

    # --------------------------------------------------------
    def toggle_acceleration_mode(self):
        self.acceleration_mode = not self.acceleration_mode
        if self.acceleration_mode:
            # Deactivate other modes
            self.wave_active = False
            self.velocity_mode = False
            self.btn_wave.configure(text="Start Wave Motion", fg_color="blue")
            
            self.btn_accel_mode.configure(text="Stop Acceleration Mode", fg_color="orange")
            self.get_logger().info("Acceleration control enabled.")
        else:
            self.btn_accel_mode.configure(text="Start Acceleration Mode", fg_color="blue")
            # Reset sliders and command to zero when stopping
            for i, sld in enumerate(self.accel_sliders):
                sld.set(0.0)
                self.accel_value_labels[i].configure(text="0.00")
            self.acceleration_command = np.zeros(6)
            self.get_logger().info("Acceleration control disabled.")

    # --------------------------------------------------------
    def close_gui(self, app):
        self.get_logger().info("Closing GUI and ROS node …")
        app.destroy()
        self.destroy_node()

# ============================================================
# 🚀 3. Entry point
# ============================================================

def main():
    rclpy.init()
    node = ArmMoverGUI()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()