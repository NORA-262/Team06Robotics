#!/usr/bin/env python3
"""
acceleration_kinematics_gui.py
---------------------------------
Interactive GUI for end-effector acceleration control.

Implements:
    X_ddot = J(q) * q_ddot + J_dot(q, q_dot) * q_dot
→  q_ddot = J⁺(q) * [X_ddot_des - J_dot * q_dot]

Publishes joint velocity updates to /joint_commands in real time.

Author: ChatGPT (based on velocity_kinematics_complete.py)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import time
import threading
import customtkinter as ctk

# Import your kinematics library
from velocity_kinematics_complete import geometric_jacobian_numeric

# ============================================================
# ⚙ ROBOT PARAMETERS
# ============================================================
JOINT_COUNT = 6
JOINT_TYPES = ['R'] * JOINT_COUNT
DH_PARAMS = [
    (0, np.pi / 2, 0.1, 0),
    (0.25, 0, 0, 0),
    (0.15, 0, 0, 0),
    (0, np.pi / 2, 0.18, 0),
    (0, -np.pi / 2, 0, 0),
    (0, 0, 0.06, 0)
]

# ============================================================
# 🤖 ROS2 NODE
# ============================================================
class AccelerationGUI(Node):
    def __init__(self):
        super().__init__("acceleration_gui_controller")

        # ROS topics
        self.joint_sub = self.create_subscription(JointState, "joint_state", self.joint_state_callback, 10)
        self.joint_pub = self.create_publisher(Float64MultiArray, "joint_commands", 10)

        # State variables
        self.q = np.zeros(JOINT_COUNT)
        self.q_dot = np.zeros(JOINT_COUNT)
        self.prev_J = np.zeros((6, JOINT_COUNT))
        self.prev_time = time.time()
        self.initialized = False

        # Desired end-effector acceleration [vx_ddot, vy_ddot, vz_ddot, wx_ddot, wy_ddot, wz_ddot]
        self.xdd_des = np.zeros(6)

        # GUI
        gui_thread = threading.Thread(target=self.build_gui, daemon=True)
        gui_thread.start()

        # Control loop (100 Hz)
        self.create_timer(0.01, self.control_loop)
        self.get_logger().info("🚀 Acceleration GUI Controller started (100 Hz).")

    # ------------------------------------------------------------
    def joint_state_callback(self, msg):
        if len(msg.position) >= JOINT_COUNT:
            self.q = np.array(msg.position[:JOINT_COUNT])
        if len(msg.velocity) >= JOINT_COUNT:
            self.q_dot = np.array(msg.velocity[:JOINT_COUNT])
        self.initialized = True

    # ------------------------------------------------------------
    def compute_Jdot(self, q, q_dot, dt=0.01):
        """Numerical approximation of J_dot."""
        J_curr, _ = geometric_jacobian_numeric(DH_PARAMS, q, JOINT_TYPES)
        J_dot = (J_curr - self.prev_J) / dt
        self.prev_J = J_curr
        return J_dot

    # ------------------------------------------------------------
    def control_loop(self):
        if not self.initialized:
            return

        t_now = time.time()
        dt = t_now - self.prev_time if self.prev_time else 0.01
        self.prev_time = t_now

        # Compute Jacobians
        J, _ = geometric_jacobian_numeric(DH_PARAMS, self.q, JOINT_TYPES)
        J_dot = self.compute_Jdot(self.q, self.q_dot, dt)

        # Compute joint accelerations
        q_ddot = np.linalg.pinv(J) @ (self.xdd_des - J_dot @ self.q_dot)

        # Integrate acceleration → velocity
        self.q_dot += q_ddot * dt

        # Publish
        cmd = Float64MultiArray()
        cmd.data = self.q_dot.tolist()
        self.joint_pub.publish(cmd)

    # ------------------------------------------------------------
    def build_gui(self):
        """CustomTkinter interface for end-effector acceleration control."""
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        app = ctk.CTk()
        app.title("Acceleration Kinematics Control (6-DOF)")
        app.geometry("650x600")

        title = ctk.CTkLabel(app, text="End-Effector Acceleration Control GUI",
                             font=("Arial", 22, "bold"))
        title.pack(pady=10)

        frame = ctk.CTkFrame(app)
        frame.pack(pady=10)

        self.sliders = []
        self.labels = []
        axes = ['vẍ', 'vÿ', 'vz̈', 'wẍ', 'wÿ', 'wz̈']

        for i, ax in enumerate(axes):
            row = ctk.CTkFrame(frame)
            row.pack(pady=4, fill="x")

            lbl = ctk.CTkLabel(row, text=f"{ax}", width=50)
            lbl.pack(side="left", padx=5)

            sld = ctk.CTkSlider(row, from_=-2.0, to=2.0, width=350)
            sld.set(0.0)
            sld.pack(side="left", padx=5)
            self.sliders.append(sld)

            val_lbl = ctk.CTkLabel(row, text="0.00", width=60)
            val_lbl.pack(side="left", padx=5)
            self.labels.append(val_lbl)

            def make_cb(lbl_ref, idx):
                def _cb(value):
                    lbl_ref.configure(text=f"{value:+.2f}")
                    self.xdd_des[idx] = value
                return _cb
            sld.configure(command=make_cb(val_lbl, i))

        # Buttons
        btn_frame = ctk.CTkFrame(app)
        btn_frame.pack(pady=20)

        reset_btn = ctk.CTkButton(btn_frame, text="Reset Accelerations",
                                  command=self.reset_accels, fg_color="blue")
        reset_btn.grid(row=0, column=0, padx=10)

        quit_btn = ctk.CTkButton(btn_frame, text="Quit", fg_color="red",
                                 command=lambda: self.close_gui(app))
        quit_btn.grid(row=0, column=1, padx=10)

        footer = ctk.CTkLabel(app, text="Ẍ = J(q)q̈ + J̇(q,q̇)q̇ | Publishes to /joint_commands @100Hz",
                              font=("Arial", 12))
        footer.pack(pady=10)

        app.mainloop()

    # ------------------------------------------------------------
    def reset_accels(self):
        self.xdd_des[:] = 0.0
        for s, lbl in zip(self.sliders, self.labels):
            s.set(0.0)
            lbl.configure(text="0.00")

    def close_gui(self, app):
        self.get_logger().info("🛑 Closing Acceleration GUI Node...")
        app.destroy()
        self.destroy_node()


# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    rclpy.init()
    node = AccelerationGUI()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down GUI controller.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
