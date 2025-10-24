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
import customtkinter as ctk
import numpy as np
import sympy as sp
import math, time, threading

# ============================================================
# ⚙️ 1. Constants and symbolic FK setup
# ============================================================

JOINT_COUNT = 6  # number of joints / sliders

# symbolic joint vars for first 5 joints
θ1, θ2, θ3, θ4, θ5 = sp.symbols('θ1 θ2 θ3 θ4 θ5')

# simplified symbolic forward-kinematics model (replace constants with your DH table)
T1 = sp.Matrix([
    [sp.cos(θ1), -sp.sin(θ1), 0, 0],
    [sp.sin(θ1),  sp.cos(θ1), 0, 0],
    [0,            0,          1, 0.04],
    [0,            0,          0, 1]
])
T2 = sp.Matrix([
    [sp.cos(θ2), 0, sp.sin(θ2), 0],
    [0,          1, 0,           0],
    [-sp.sin(θ2),0, sp.cos(θ2),  0],
    [0,          0, 0,           1]
])
T3 = sp.Matrix([
    [sp.cos(θ3), 0, sp.sin(θ3), 0.108*sp.cos(θ3)],
    [0,          1, 0,           0],
    [-sp.sin(θ3),0, sp.cos(θ3),  0.108*sp.sin(θ3)],
    [0,          0, 0,           1]
])
T4 = sp.Matrix([
    [sp.cos(θ4), 0, sp.sin(θ4), 0.1*sp.cos(θ4)],
    [0,          1, 0,           0],
    [-sp.sin(θ4),0, sp.cos(θ4),  0.1*sp.sin(θ4)],
    [0,          0, 0,           1]
])
T5 = sp.Matrix([
    [sp.cos(θ5), -sp.sin(θ5), 0, 0.045*sp.cos(θ5)],
    [sp.sin(θ5),  sp.cos(θ5), 0, 0.045*sp.sin(θ5)],
    [0,            0,          1, 0],
    [0,            0,          0, 1]
])

T_0_5 = T1 * T2 * T3 * T4 * T5

# numeric version
fk_numeric = sp.lambdify((θ1, θ2, θ3, θ4, θ5), T_0_5, "numpy")

def forward_kinematics_position_from_first5(joint_values):
    j = joint_values[:5]
    T = np.array(fk_numeric(*j), dtype=float)
    return T[:3, 3]  # x, y, z

# ============================================================
# 🤖 2. Node + GUI
# ============================================================

class ArmMoverGUI(Node):
    def __init__(self):
        super().__init__("move_up")

        # Publishers
        self.joint_pub = self.create_publisher(Float64MultiArray, "joint_commands", 10)
        self.ee_pub = self.create_publisher(Float64MultiArray, "ee_position", 10)

        # Timing & flags
        self.start_time = time.time()
        self.wave_active = False
        self.gui_ready = False
        self.sliders = []
        self.value_labels = []

        # Start GUI in separate thread
        gui_thread = threading.Thread(target=self.build_gui, daemon=True)
        gui_thread.start()

        # Timer (20 Hz)
        self.create_timer(0.05, self.safe_publish_command)

        self.get_logger().info("Arm Mover GUI Node started.")

    # --------------------------------------------------------
    def safe_publish_command(self):
        if self.gui_ready:
            self.publish_command()

    # --------------------------------------------------------
    def publish_command(self):
        msg = Float64MultiArray()
        joints = []

        if self.wave_active:
            t = time.time() - self.start_time
            for i in range(JOINT_COUNT):
                val = 0.5 * math.sin(t * (0.5 + i*0.2))
                self.sliders[i].set(val)
                joints.append(val)
        else:
            joints = [s.get() for s in self.sliders]

        msg.data = joints
        self.joint_pub.publish(msg)

        # FK on first 5 joints
        try:
            pos = forward_kinematics_position_from_first5(joints)
            ee_msg = Float64MultiArray()
            ee_msg.data = pos.tolist()
            self.ee_pub.publish(ee_msg)

            # log every 1 s
            if int(time.time()) % 1 == 0:
                self.get_logger().info(f"EE pos: {pos}")
        except Exception as e:
            self.get_logger().warn(f"FK error: {e}")

    # --------------------------------------------------------
    def build_gui(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        app = ctk.CTk()
        app.title("6-DOF Arm Controller")
        app.geometry("620x560")

        title = ctk.CTkLabel(app, text="Low-Cost Robot Arm Controller",
                             font=("Arial", 22, "bold"))
        title.pack(pady=10)

        frame = ctk.CTkFrame(app)
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

        # Buttons
        btn_frame = ctk.CTkFrame(app)
        btn_frame.pack(pady=20)

        self.btn_send = ctk.CTkButton(btn_frame, text="Send Once", command=self.send_once)
        self.btn_send.grid(row=0, column=0, padx=10)

        self.btn_wave = ctk.CTkButton(btn_frame, text="Start Wave Motion", fg_color="blue",
                                      command=self.toggle_wave)
        self.btn_wave.grid(row=0, column=1, padx=10)

        self.btn_quit = ctk.CTkButton(btn_frame, text="Quit", fg_color="red",
                                      command=lambda: self.close_gui(app))
        self.btn_quit.grid(row=0, column=2, padx=10)

        footer = ctk.CTkLabel(app,
            text="Publishes: /joint_commands and /ee_position (20 Hz)",
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
            pos = forward_kinematics_position_from_first5(joints)
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
