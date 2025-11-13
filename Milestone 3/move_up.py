#!/usr/bin/env python3
"""
move_up.py — Velocity Kinematics Integrated Controller
------------------------------------------------------
Implements the concepts from Lecture 05 (Velocity Kinematics):
    X_dot = J(q) * q_dot

Features:
- Computes Forward Kinematics (position)
- Computes Jacobian and End-Effector Velocity
- Publishes: /joint_commands, /ee_position, /ee_velocity
- GUI with sliders + optional wave motion
- High-frequency 1000 Hz update rate
"""

# ============================================================
# 🧩 IMPORTS
# ============================================================

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import customtkinter as ctk
import numpy as np
import math, time, threading

# Import the kinematics functions
from velocity_kinematics_complete import (
    fkine_numeric,
    geometric_jacobian_numeric
)

# ============================================================
# ⚙ ROBOT PARAMETERS (DH TABLE)
# ============================================================

JOINT_COUNT = 6
JOINT_TYPES = ['R'] * JOINT_COUNT

# Example DH table (replace with your robot’s)
DH_PARAMS = [
    (0, math.pi/2, 0.1, 0),
    (0.25, 0, 0, 0),
    (0.15, 0, 0, 0),
    (0, math.pi/2, 0.18, 0),
    (0, -math.pi/2, 0, 0),
    (0, 0, 0.06, 0)
]

# Speed multiplier (change this to increase/decrease robot speed)
VELOCITY_SCALE = 1000_000   # increase this value (e.g. 20_000 or 50_000) for faster movement

# ============================================================
# 🤖 ROS2 NODE + GUI
# ============================================================

class ArmMoverGUI(Node):
    def __init__(self):
        super().__init__("velocity_kinematics_gui")

        # Publishers
        self.joint_pub = self.create_publisher(Float64MultiArray, "joint_commands", 10)
        self.ee_pos_pub = self.create_publisher(Float64MultiArray, "ee_position", 10)
        self.ee_vel_pub = self.create_publisher(Float64MultiArray, "ee_velocity", 10)

        # Initialize parameters
        self.wave_active = False
        self.gui_ready = False
        self.sliders = []
        self.value_labels = []
        self.prev_joint_values = np.zeros(JOINT_COUNT)
        self.prev_time = time.time()

        # Build GUI in a separate thread
        gui_thread = threading.Thread(target=self.build_gui, daemon=True)
        gui_thread.start()

        # Timer: 1000 Hz (Velocity Kinematics Loop)
        self.create_timer(0.000001, self.safe_publish_command)
        self.get_logger().info("✅ Velocity Kinematics Controller started (1000 Hz).")

    # --------------------------------------------------------
    def safe_publish_command(self):
        """Ensure GUI is initialized before running."""
        if self.gui_ready:
            self.publish_velocity_kinematics()

    # --------------------------------------------------------
    def publish_velocity_kinematics(self):
        """
        Implements:  Ẋ = J(q) q̇
        Publishes: joint commands, end-effector position, velocity.
        """
        # Get joint angles
        joints = np.array([s.get() for s in self.sliders])

        # Compute time delta and joint velocities
        t_now = time.time()
        dt = t_now - self.prev_time if self.prev_time else 0.001
        q_dot = (joints - self.prev_joint_values) / dt
        q_dot *= VELOCITY_SCALE  # Apply speed multiplier here ✅

        # Publish joint commands
        joint_msg = Float64MultiArray()
        joint_msg.data = joints.tolist()
        self.joint_pub.publish(joint_msg)

        try:
            # ---- Forward Kinematics ----
            T, origins, z_axes = fkine_numeric(DH_PARAMS, joints, JOINT_TYPES)
            ee_pos = T[:3, 3]

            # Publish EE position
            ee_pos_msg = Float64MultiArray()
            ee_pos_msg.data = ee_pos.tolist()
            self.ee_pos_pub.publish(ee_pos_msg)

            # ---- Velocity Kinematics (Jacobian) ----
            J, _ = geometric_jacobian_numeric(DH_PARAMS, joints, JOINT_TYPES)
            ee_twist = J @ q_dot  # [vx, vy, vz, wx, wy, wz]

            # Publish EE velocity
            ee_vel_msg = Float64MultiArray()
            ee_vel_msg.data = ee_twist.tolist()
            self.ee_vel_pub.publish(ee_vel_msg)

            # Update memory for next iteration
            self.prev_joint_values = joints
            self.prev_time = t_now

        except Exception as e:
            self.get_logger().warn(f"⚠ Velocity kinematics error: {e}")

    # --------------------------------------------------------
    def build_gui(self):
        """Create the CustomTkinter interface."""
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        app = ctk.CTk()
        app.title("Velocity Kinematics GUI (6-DOF)")
        app.geometry("660x600")

        title = ctk.CTkLabel(app, text="Velocity Kinematics Controller (6R Robot)",
                             font=("Arial", 22, "bold"))
        title.pack(pady=10)

        frame = ctk.CTkFrame(app)
        frame.pack(pady=10)

        # Create sliders
        for i in range(JOINT_COUNT):
            row = ctk.CTkFrame(frame)
            row.pack(pady=4, fill="x")

            lbl = ctk.CTkLabel(row, text=f"Joint {i+1}", width=80)
            lbl.pack(side="left", padx=5)

            sld = ctk.CTkSlider(row, from_=-math.pi, to=math.pi, width=350)
            sld.set(0.0)
            sld.pack(side="left", padx=5)
            self.sliders.append(sld)

            val_lbl = ctk.CTkLabel(row, text="0.00", width=60)
            val_lbl.pack(side="left", padx=5)
            self.value_labels.append(val_lbl)

            def make_callback(lbl_ref):
                def _cb(value):
                    lbl_ref.configure(text=f"{value:+.2f}")
                return _cb

            sld.configure(command=make_callback(val_lbl))

        # Buttons
        btn_frame = ctk.CTkFrame(app)
        btn_frame.pack(pady=20)

        self.btn_quit = ctk.CTkButton(btn_frame, text="Quit", fg_color="red",
                                      command=lambda: self.close_gui(app))
        self.btn_quit.grid(row=0, column=0, padx=10)

        footer = ctk.CTkLabel(app,
            text="Implements Ẋ = J(q) q̇  |  Publishes at 1000 Hz",
            font=("Arial", 12))
        footer.pack(pady=10)

        self.gui_ready = True
        app.mainloop()

    # --------------------------------------------------------
    def close_gui(self, app):
        self.get_logger().info("🛑 Closing Velocity Kinematics Node...")
        app.destroy()
        self.destroy_node()

# ============================================================
# 🚀 MAIN ENTRY
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
