#!/usr/bin/env python3
"""
move_up.py — Velocity Kinematics Integrated Controller
------------------------------------------------------
Implements Velocity Kinematics:
    X_dot = J(q) * q_dot

Features:
- Computes Forward Kinematics (position)
- Computes Jacobian and End-Effector Velocity
- Publishes: /joint_commands, /ee_position, /ee_velocity
- GUI with sliders
- Adjustable velocity scaling (change VELOCITY_SCALE)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import customtkinter as ctk
import numpy as np
import math, time, threading

# Import your robot-specific kinematics functions
from velocity_kinematics_complete import (
    fkine_numeric,
    geometric_jacobian_numeric
)

# ============================================================
# ⚙️ ROBOT PARAMETERS
# ============================================================

JOINT_COUNT = 6
JOINT_TYPES = ['R'] * JOINT_COUNT

# Example DH parameters — adjust for your robot
DH_PARAMS = [
    (0, math.pi/2, 0.1, 0),
    (0.25, 0, 0, 0),
    (0.15, 0, 0, 0),
    (0, math.pi/2, 0.18, 0),
    (0, -math.pi/2, 0, 0),
    (0, 0, 0.06, 0)
]

# ============================================================
# 🚀 CONFIGURATION
# ============================================================

VELOCITY_SCALE = 5000000000000  # 🔧 Increase this value to move faster (default: 1.0)

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

        self.get_logger().info("✅ Velocity Kinematics Node started.")

        # Initialize variables
        self.gui_ready = False
        self.sliders = []
        self.value_labels = []
        self.prev_joint_values = np.zeros(JOINT_COUNT)
        self.prev_time = time.time()

        # Launch GUI in a separate thread
        threading.Thread(target=self.build_gui, daemon=True).start()

        # Timer for ROS2 loop
        self.create_timer(55, self.safe_publish_command)

    # --------------------------------------------------------
    def safe_publish_command(self):
        if self.gui_ready:
            self.publish_velocity_kinematics()

    # --------------------------------------------------------
    def publish_velocity_kinematics(self):
        joints = np.array([s.get() for s in self.sliders])
        t_now = time.time()
        dt = max(t_now - self.prev_time, 1e-4)
        q_dot = (joints - self.prev_joint_values) / dt

        # 🏎️ Apply global velocity scaling
        q_dot *= VELOCITY_SCALE

        # Publish joint commands
        msg = Float64MultiArray()
        msg.data = q_dot.tolist()  # Sending velocities
        self.joint_pub.publish(msg)

        try:
            # Forward Kinematics
            T, origins, z_axes = fkine_numeric(DH_PARAMS, joints, JOINT_TYPES)
            ee_pos = T[:3, 3]

            # Jacobian and EE velocity
            J, _ = geometric_jacobian_numeric(DH_PARAMS, joints, JOINT_TYPES)
            ee_twist = J @ q_dot

            # Publish
            ee_pos_msg = Float64MultiArray()
            ee_pos_msg.data = ee_pos.tolist()
            self.ee_pos_pub.publish(ee_pos_msg)

            ee_vel_msg = Float64MultiArray()
            ee_vel_msg.data = ee_twist.tolist()
            self.ee_vel_pub.publish(ee_vel_msg)

        except Exception as e:
            self.get_logger().warn(f"⚠️ Kinematics error: {e}")

        self.prev_joint_values = joints
        self.prev_time = t_now

    # --------------------------------------------------------
    def build_gui(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        app = ctk.CTk()
        app.title("Velocity Kinematics Controller (6-DOF)")
        app.geometry("700x600")

        title = ctk.CTkLabel(app, text="Velocity Kinematics Controller",
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

        # Quit Button
        btn_quit = ctk.CTkButton(app, text="Quit", fg_color="red",
                                 command=lambda: self.close_gui(app))
        btn_quit.pack(pady=20)

        footer = ctk.CTkLabel(app, text=f"Publishing @100 Hz | Velocity Scale: {VELOCITY_SCALE}",
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
# 🏁 MAIN
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
