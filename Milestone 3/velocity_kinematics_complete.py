#!/usr/bin/env python3
"""
velocity_kinematics_complete.py
--------------------------------
Complete library for Forward & Inverse Velocity Kinematics
of an N-joint (e.g. 6-DOF) manipulator.

Includes:
- Symbolic and Numeric Forward Kinematics
- Geometric Jacobian computation
- Inverse Velocity Kinematics (Pseudo-Inverse & Damped LS)
- Utility math functions
- Example test case for a 6R robot

Author: <Your Name>
"""

# ============================================================
# 📦 IMPORTS
# ============================================================

import numpy as np
import sympy as sp
from numpy.linalg import pinv, inv
from sympy import sin, cos, Matrix, symbols, simplify
from typing import List, Tuple, Union

# Optional (for visualization / debugging)
import math
import pprint
import itertools
import warnings

# ============================================================
# ⚙️ 1. HELPER FUNCTIONS
# ============================================================

def dh_transform_numeric(a, alpha, d, theta):
    """Return numeric homogeneous transform (4x4) using DH parameters."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,      ca,     d   ],
        [0.0,   0.0,     0.0,    1.0  ]
    ], dtype=float)

def dh_transform_symbolic(a, alpha, d, theta):
    """Return symbolic homogeneous transform (4x4) using DH parameters."""
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    ct, st = sp.cos(theta), sp.sin(theta)
    return sp.Matrix([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,      ca,     d   ],
        [0,       0,       0,     1   ]
    ])

def skew(v):
    """Return skew-symmetric matrix of a 3x1 vector."""
    vx, vy, vz = v
    return np.array([
        [0, -vz, vy],
        [vz, 0, -vx],
        [-vy, vx, 0]
    ])

def normalize(v):
    """Return normalized 3D vector."""
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def print_matrix(name, M):
    """Pretty print a matrix (NumPy or SymPy)."""
    print(f"\n{name} =")
    if isinstance(M, sp.Matrix):
        sp.pprint(M)
    else:
        with np.printoptions(precision=4, suppress=True):
            print(np.array(M, dtype=float))

# ============================================================
# 🦿 2. FORWARD KINEMATICS (NUMERIC)
# ============================================================

def fkine_numeric(dh_params, joint_values, joint_types=None):
    """
    Forward kinematics (numeric)
    Args:
      dh_params: list of (a, alpha, d, theta_offset)
      joint_values: list of N joint values
      joint_types: list of 'R' or 'P' (defaults to all 'R')
    Returns:
      T_0_n: 4x4 transform, origins[], z_axes[]
    """
    N = len(dh_params)
    if joint_types is None:
        joint_types = ['R'] * N

    T = np.eye(4)
    origins, z_axes = [T[:3, 3]], [T[:3, 2]]
    for i in range(N):
        a, alpha, d0, theta0 = dh_params[i]
        jt = joint_types[i].upper()
        q = joint_values[i]
        if jt == 'R':
            theta, d = theta0 + q, d0
        elif jt == 'P':
            theta, d = theta0, d0 + q
        else:
            raise ValueError("joint_types must be 'R' or 'P'")
        A = dh_transform_numeric(a, alpha, d, theta)
        T = T @ A
        origins.append(T[:3, 3])
        z_axes.append(T[:3, 2])

    return T, origins, z_axes

# ============================================================
# 🧩 3. JACOBIAN (GEOMETRIC)
# ============================================================

def geometric_jacobian_numeric(dh_params, joint_values, joint_types=None):
    """Compute 6xN geometric Jacobian numerically."""
    N = len(dh_params)
    T, origins, z_axes = fkine_numeric(dh_params, joint_values, joint_types)
    O_n = origins[-1]
    J = np.zeros((6, N))
    for i in range(N):
        z = z_axes[i]
        O_i = origins[i]
        jt = 'R' if joint_types is None else joint_types[i].upper()
        if jt == 'R':
            Jv = np.cross(z, O_n - O_i)
            Jw = z
        elif jt == 'P':
            Jv = z
            Jw = np.zeros(3)
        J[0:3, i] = Jv
        J[3:6, i] = Jw
    return J, T

# ============================================================
# 🔁 4. INVERSE VELOCITY KINEMATICS
# ============================================================

def inverse_velocity(J, twist, method='pseudoinv', damping=0.01):
    """
    Inverse velocity kinematics: solve q_dot = J⁺ * twist
    twist: [vx, vy, vz, wx, wy, wz]
    """
    twist = np.array(twist, dtype=float).reshape(6,)
    if method == 'pseudoinv':
        qdot = pinv(J) @ twist
    elif method == 'dls':
        JJt = J @ J.T
        lam2 = damping ** 2
        qdot = J.T @ inv(JJt + lam2 * np.eye(6)) @ twist
    else:
        raise ValueError("method must be 'pseudoinv' or 'dls'")
    return qdot

# ============================================================
# 🧮 5. SYMBOLIC KINEMATICS (Optional)
# ============================================================

def fkine_symbolic(dh_params, q_symbols, joint_types=None):
    """Return symbolic forward kinematics (SymPy Matrix)."""
    N = len(dh_params)
    if joint_types is None:
        joint_types = ['R'] * N
    T = sp.eye(4)
    for i in range(N):
        a, alpha, d, theta = dh_params[i]
        jt = joint_types[i].upper()
        if jt == 'R':
            T_i = dh_transform_symbolic(a, alpha, d, theta + q_symbols[i])
        elif jt == 'P':
            T_i = dh_transform_symbolic(a, alpha, d + q_symbols[i], theta)
        T = T * T_i
    return simplify(T)

# ============================================================
# 🧪 6. EXAMPLE TEST (6R ROBOT)
# ============================================================

if __name__ == "__main__":
    # Example DH table for a simple 6R robot
    dh_params = [
        (0, np.pi/2, 0.1, 0),
        (0.25, 0, 0, 0),
        (0.15, 0, 0, 0),
        (0, np.pi/2, 0.18, 0),
        (0, -np.pi/2, 0, 0),
        (0, 0, 0.06, 0)
    ]
    joint_types = ['R'] * 6
    q = np.array([0.2, -0.3, 0.4, -0.2, 0.1, 0.0])

    print("\n========== FORWARD KINEMATICS ==========")
    T, origins, z_axes = fkine_numeric(dh_params, q, joint_types)
    print_matrix("T_0_6", T)
    print("\nEnd-effector position:", np.round(T[:3, 3], 4))

    print("\n========== JACOBIAN MATRIX ==========")
    J, _ = geometric_jacobian_numeric(dh_params, q, joint_types)
    print_matrix("J", J)

    print("\n========== INVERSE VELOCITY (Test) ==========")
    twist = np.array([0.05, 0, 0, 0, 0, 0.1])  # desired end-effector velocities
    qdot = inverse_velocity(J, twist, method='dls', damping=0.05)
    print_matrix("q_dot (DLS)", qdot)
    print_matrix("J*q_dot", J @ qdot)

    print("\n✅ Done.")

