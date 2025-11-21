#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time
import argparse

class MoveUpArray(Node):
    def __init__(self, n_joints=6):
        super().__init__('move_up_array')
        self.pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.n_joints = n_joints

    def send_velocity(self, velocities):
        msg = Float64MultiArray()
        msg.data = velocities
        self.pub.publish(msg)

    def move_up(self, joint_index=1, vel=0.5, duration=3.0):
        """Move one joint up (by velocity) for a duration."""
        rate_hz = 20
        dt = 1.0 / rate_hz
        n_steps = int(duration * rate_hz)
        velocities = [0.0] * self.n_joints
        velocities[joint_index] = vel
        self.get_logger().info(f"Moving joint {joint_index} up with vel={vel} for {duration}s")

        for _ in range(n_steps):
            self.send_velocity(velocities)
            time.sleep(dt)

        # stop all joints
        self.send_velocity([0.0] * self.n_joints)
        self.get_logger().info("Stopped (zero velocity published)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint', type=int, default=1, help='Joint index (0-based)')
    parser.add_argument('--vel', type=float, default=0.5, help='Velocity value')
    parser.add_argument('--duration', type=float, default=3.0, help='Time in seconds')
    args = parser.parse_args()

    rclpy.init()
    node = MoveUpArray()
    node.move_up(joint_index=args.joint, vel=args.vel, duration=args.duration)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
