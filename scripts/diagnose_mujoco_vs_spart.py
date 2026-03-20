"""
Diagnose SPART vs MuJoCo discrepancy.

Test 1: Inertia parameters - compare mass, inertia, CoM, FK positions
Test 2: Quaternion convention - verify xyzw <-> wxyz conversion
Test 3: Base angular velocity - SPART momentum conservation vs MuJoCo actual
"""

import os
import sys
import numpy as np

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import mujoco
from src.dynamics.urdf2robot import urdf2robot
import src.dynamics.spart_functions as spart_np

np.set_printoptions(precision=6, suppress=True, linewidth=120)


def quat_wxyz_to_rotmat(q_wxyz):
    """MuJoCo convention [w,x,y,z] -> 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])


def quat_xyzw_to_rotmat(q_xyzw):
    """SPART convention [x,y,z,w] -> 3x3 rotation matrix."""
    x, y, z, w = q_xyzw
    return quat_wxyz_to_rotmat([w, x, y, z])


def euler_to_quat_xyzw(roll, pitch, yaw):
    """ZYX Euler -> quaternion [x,y,z,w]."""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    qw = cr*cp*cy + sr*sp*sy
    q = np.array([qx, qy, qz, qw])
    return q / np.linalg.norm(q)


# ============================================================================
# Load models
# ============================================================================
urdf_path = os.path.join(ROOT_DIR, "assets", "SC_ur10e.urdf")
xml_path = os.path.join(ROOT_DIR, "assets", "spacerobot_cjt.xml")

robot, _ = urdf2robot(urdf_path, verbose_flag=False)
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)

n_q = robot['n_q']
R0 = np.eye(3)
r0 = np.zeros((3, 1))


def separator(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


# ============================================================================
# TEST 1: Inertia Parameters Comparison
# ============================================================================
separator("TEST 1: INERTIA PARAMETERS (SPART from URDF vs MuJoCo)")

print("\n--- Mass Comparison ---")
# MuJoCo body order
mj_body_names = [mj_model.body(i).name for i in range(mj_model.nbody)]
spart_link_names = ['base_link'] + [l['name'] for l in robot['links']]

print(f"{'Link':<25} {'SPART mass':>12} {'MuJoCo mass':>12} {'Diff':>12}")
print("-" * 65)

total_spart = robot['base_link']['mass']
total_mj = 0.0

for i in range(mj_model.nbody):
    mj_name = mj_model.body(i).name
    mj_mass = mj_model.body(i).mass[0]
    total_mj += mj_mass

    # Find matching SPART link
    if mj_name == 'world':
        continue
    elif mj_name == 'base_link':
        spart_mass = robot['base_link']['mass']
    else:
        matched = [l for l in robot['links'] if l['name'] == mj_name]
        if matched:
            spart_mass = matched[0]['mass']
        else:
            spart_mass = float('nan')

    diff = abs(spart_mass - mj_mass)
    print(f"{mj_name:<25} {spart_mass:>12.4f} {mj_mass:>12.4f} {diff:>12.6f}")

print(f"\n{'Total':<25} {total_spart + sum(l['mass'] for l in robot['links']):>12.4f} {total_mj:>12.4f}")

print("\n--- Inertia Tensor Comparison ---")
print(f"{'Link':<25} {'Max |I_spart - I_mujoco|':>25}")
print("-" * 55)

for i in range(mj_model.nbody):
    mj_name = mj_model.body(i).name
    if mj_name == 'world':
        continue

    # MuJoCo inertia: body_inertia is diagonal in body frame, body_iquat rotates it
    mj_inertia_diag = mj_model.body(i).inertia.copy()
    mj_iquat = mj_model.body(i).iquat.copy()  # [w,x,y,z]
    R_inertia = quat_wxyz_to_rotmat(mj_iquat)
    # Full inertia in body frame: R @ diag(I) @ R^T
    mj_I = R_inertia @ np.diag(mj_inertia_diag) @ R_inertia.T

    if mj_name == 'base_link':
        spart_I = robot['base_link']['inertia']
    else:
        matched = [l for l in robot['links'] if l['name'] == mj_name]
        if matched:
            # SPART stores inertia in the inertial frame (which may be rotated)
            # The 'T' transform includes the inertial frame rotation
            T = matched[0]['T']
            R_T = T[:3, :3]  # rotation of inertial frame
            I_raw = matched[0]['inertia']
            # Inertia in link frame = R_T @ I_raw @ R_T^T
            spart_I = R_T @ I_raw @ R_T.T
        else:
            continue

    max_diff = np.max(np.abs(spart_I - mj_I))
    status = "OK" if max_diff < 1e-6 else "MISMATCH!"
    print(f"{mj_name:<25} {max_diff:>25.8f}  {status}")
    if max_diff > 1e-6:
        print(f"  SPART: {spart_I.flatten()}")
        print(f"  MuJoCo: {mj_I.flatten()}")


# --- Forward Kinematics comparison ---
print("\n--- Forward Kinematics Comparison (q=0) ---")
q_test = np.zeros(n_q)

# SPART FK
RJ, RL, rJ, rL, e, g = spart_np.kinematics(R0, r0, q_test, robot)

# MuJoCo FK
mj_data.qpos[:] = 0
mj_data.qpos[3] = 1.0  # wxyz identity quaternion
mujoco.mj_kinematics(mj_model, mj_data)

print(f"{'Link':<25} {'SPART pos':>35} {'MuJoCo pos':>35} {'Diff norm':>12}")
print("-" * 110)

for i in range(n_q):
    link_name = robot['links'][i]['name']
    spart_pos = rL[:, i].flatten()

    # Find matching MuJoCo body
    mj_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    if mj_bid >= 0:
        mj_pos = mj_data.xpos[mj_bid]
        diff = np.linalg.norm(spart_pos - mj_pos)
        print(f"{link_name:<25} {str(spart_pos):>35} {str(mj_pos):>35} {diff:>12.6f}")
    else:
        print(f"{link_name:<25} {str(spart_pos):>35} {'NOT FOUND':>35}")


# --- FK at non-zero q ---
print("\n--- Forward Kinematics Comparison (q = [0.3, -0.5, 0.2, 0.1, -0.4, 0.6]) ---")
q_test2 = np.array([0.3, -0.5, 0.2, 0.1, -0.4, 0.6])

RJ2, RL2, rJ2, rL2, e2, g2 = spart_np.kinematics(R0, r0, q_test2, robot)

mj_data.qpos[:] = 0
mj_data.qpos[3] = 1.0
mj_data.qpos[7:7+n_q] = q_test2
mujoco.mj_kinematics(mj_model, mj_data)

print(f"{'Link':<25} {'|pos diff|':>12} {'|rot diff| (deg)':>18}")
print("-" * 60)

for i in range(n_q):
    link_name = robot['links'][i]['name']
    spart_pos = rL2[:, i].flatten()
    spart_R = RL2[:, :, i]

    mj_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    if mj_bid >= 0:
        mj_pos = mj_data.xpos[mj_bid]
        mj_R = mj_data.xmat[mj_bid].reshape(3, 3)
        pos_diff = np.linalg.norm(spart_pos - mj_pos)

        R_err = spart_R.T @ mj_R
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        print(f"{link_name:<25} {pos_diff:>12.6f} {np.degrees(angle):>18.4f}")


# ============================================================================
# TEST 2: Quaternion Convention Check
# ============================================================================
separator("TEST 2: QUATERNION CONVENTION")

print("\nVerify: SPART [x,y,z,w] <-> MuJoCo [w,x,y,z] conversion")

# Known rotation: 30° about Z axis
angle = np.radians(30)
# Expected rotation matrix
R_expected = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle),  np.cos(angle), 0],
    [0,              0,             1]
])

# Quaternion representations
q_wxyz = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])  # MuJoCo
q_xyzw = np.array([0, 0, np.sin(angle/2), np.cos(angle/2)])  # SPART

R_from_wxyz = quat_wxyz_to_rotmat(q_wxyz)
R_from_xyzw = quat_xyzw_to_rotmat(q_xyzw)

print(f"\n30° about Z axis:")
print(f"  q_wxyz (MuJoCo): {q_wxyz}")
print(f"  q_xyzw (SPART):  {q_xyzw}")
print(f"  R_expected:\n{R_expected}")
print(f"  R_from_wxyz:\n{R_from_wxyz}")
print(f"  R_from_xyzw:\n{R_from_xyzw}")
print(f"  |R_wxyz - R_expected| = {np.max(np.abs(R_from_wxyz - R_expected)):.2e}")
print(f"  |R_xyzw - R_expected| = {np.max(np.abs(R_from_xyzw - R_expected)):.2e}")

# Check the conversion used in simulate_ddp_and_compare.py
print("\n--- Check simulate_ddp_and_compare.py conversion ---")
print("Code does: quat_ddp_wxyz = quat_ddp_xyzw[:, [3, 0, 1, 2]]")
print("i.e., [x,y,z,w] -> [w,x,y,z] via indices [3,0,1,2]")

q_spart = np.array([0.131, 0.131, -0.062, 0.984])  # [x,y,z,w] from scenario
q_converted = q_spart[[3, 0, 1, 2]]  # -> [w,x,y,z]
print(f"  SPART goal quat [x,y,z,w]: {q_spart}")
print(f"  Converted [w,x,y,z]:       {q_converted}")

R_spart = quat_xyzw_to_rotmat(q_spart)
R_mujoco = quat_wxyz_to_rotmat(q_converted)
print(f"  |R_spart - R_mujoco| = {np.max(np.abs(R_spart - R_mujoco)):.2e}")
print(f"  Conversion is {'CORRECT' if np.max(np.abs(R_spart - R_mujoco)) < 1e-10 else 'WRONG'}!")

# Check DDP's quat integration output
print("\n--- Check DDP output quaternion interpretation ---")
X_ddp = np.load(os.path.join(ROOT_DIR, "results", "trajectory_casadi_ddp_states.npy"))
q_final_xyzw = X_ddp[-1, 2*n_q:]
q_final_xyzw /= np.linalg.norm(q_final_xyzw)
print(f"  DDP final quat [x,y,z,w]: {q_final_xyzw}")
R_ddp_final = quat_xyzw_to_rotmat(q_final_xyzw)

# Convert to Euler
def rotmat_to_euler(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0
    return np.array([roll, pitch, yaw])

euler_ddp = np.degrees(rotmat_to_euler(R_ddp_final))
print(f"  DDP final Euler (deg): Roll={euler_ddp[0]:.2f}, Pitch={euler_ddp[1]:.2f}, Yaw={euler_ddp[2]:.2f}")
print(f"  Goal: Roll=15, Pitch=15, Yaw=-15")


# ============================================================================
# TEST 3: Base Angular Velocity - SPART Momentum vs MuJoCo
# ============================================================================
separator("TEST 3: BASE ANGULAR VELOCITY (Momentum Conservation)")

print("\nCompare SPART momentum-conservation base velocity vs MuJoCo free-floating response")

# Strategy: Set a known joint velocity in MuJoCo with zero gravity, zero torque.
# Step forward and measure base angular velocity.
# Compare with SPART's momentum conservation prediction.

test_configs = [
    ("q=0, qd=[1,0,0,0,0,0]", np.zeros(n_q), np.array([1.0, 0, 0, 0, 0, 0])),
    ("q=0, qd=[0,1,0,0,0,0]", np.zeros(n_q), np.array([0, 1.0, 0, 0, 0, 0])),
    ("q=0, qd=[0,0,0,0,0,1]", np.zeros(n_q), np.array([0, 0, 0, 0, 0, 1.0])),
    ("q=[0.3,-0.5,0.2,0.1,-0.4,0.6], qd=[0.5,0.3,-0.2,0.1,0.4,-0.3]",
     np.array([0.3, -0.5, 0.2, 0.1, -0.4, 0.6]),
     np.array([0.5, 0.3, -0.2, 0.1, 0.4, -0.3])),
]

print(f"\n{'Config':<60} {'SPART wb':>30} {'MuJoCo wb':>30} {'Diff norm':>12}")
print("-" * 135)

for desc, q_cfg, qd_cfg in test_configs:
    # --- SPART: compute base angular velocity via momentum conservation ---
    RJ_s, RL_s, rJ_s, rL_s, e_s, g_s = spart_np.kinematics(R0, r0, q_cfg, robot)
    Bij_s, Bi0_s, P0_s, pm_s = spart_np.diff_kinematics(R0, r0, rL_s, e_s, g_s, robot)
    I0_s, Im_s = spart_np.inertia_projection(R0, RL_s, robot)
    M0_s, Mm_s = spart_np.mass_composite_body(I0_s, Im_s, Bij_s, Bi0_s, robot)
    H0_s, H0m_s, Hm_s = spart_np.generalized_inertia_matrix(M0_s, Mm_s, Bij_s, Bi0_s, P0_s, pm_s, robot)

    # Momentum conservation: H0 * u0 + H0m * qd = 0 => u0 = -H0^{-1} H0m qd
    u0_spart = -np.linalg.solve(H0_s, H0m_s @ qd_cfg)
    wb_spart = u0_spart[:3]  # angular velocity (body frame)

    # --- MuJoCo: set state and let system evolve ---
    mj_data2 = mujoco.MjData(mj_model)
    mj_data2.qpos[:3] = 0  # base position
    mj_data2.qpos[3] = 1.0  # base quat w
    mj_data2.qpos[4:7] = 0  # base quat xyz

    # Apply base rotation for non-zero q configs
    mj_data2.qpos[7:7+n_q] = q_cfg

    # Set velocities
    mj_data2.qvel[:3] = u0_spart[3:]  # base linear velocity from SPART
    mj_data2.qvel[3:6] = wb_spart     # base angular velocity from SPART
    mj_data2.qvel[6:6+n_q] = qd_cfg   # joint velocities

    mj_data2.ctrl[:] = 0  # zero torque

    # Forward to compute forces
    mujoco.mj_forward(mj_model, mj_data2)

    # Check: with correct momentum conservation, the system should have zero
    # linear and angular momentum. MuJoCo computes cvel (body velocities).
    # Instead, let's check if the system drifts when we step forward.

    # Record initial state
    qpos_init = mj_data2.qpos.copy()
    qvel_init = mj_data2.qvel.copy()

    # Step forward many times
    n_steps = 100
    for _ in range(n_steps):
        mujoco.mj_step(mj_model, mj_data2)

    # After stepping, if momentum is truly conserved, base angular velocity should be
    # consistent. But we want to compare the initial wb.
    # Actually, let's compare differently: set MuJoCo with joint vel only,
    # zero base vel, and see what MuJoCo's momentum computation gives.

    wb_mj = qvel_init[3:6]  # we set this from SPART, so check if MuJoCo agrees

    print(f"{desc:<60} {str(wb_spart):>30} {str(wb_mj):>30} {np.linalg.norm(wb_spart - wb_mj):>12.2e}")


# Better test: set joint velocity in MuJoCo, zero base velocity, step, and see
# what base velocity MuJoCo develops (indicating momentum is not conserved with
# zero base velocity, and what it should be)
separator("TEST 3b: MuJoCo MOMENTUM DRIFT (zero base vel + joint vel)")

print("\nSet base_vel=0, joint_vel=qd, step MuJoCo, observe base velocity drift.")
print("If SPART and MuJoCo agree on dynamics, the drift direction should match SPART's wb prediction.\n")

for desc, q_cfg, qd_cfg in test_configs:
    # SPART prediction
    RJ_s, RL_s, rJ_s, rL_s, e_s, g_s = spart_np.kinematics(R0, r0, q_cfg, robot)
    Bij_s, Bi0_s, P0_s, pm_s = spart_np.diff_kinematics(R0, r0, rL_s, e_s, g_s, robot)
    I0_s, Im_s = spart_np.inertia_projection(R0, RL_s, robot)
    M0_s, Mm_s = spart_np.mass_composite_body(I0_s, Im_s, Bij_s, Bi0_s, robot)
    H0_s, H0m_s, Hm_s = spart_np.generalized_inertia_matrix(M0_s, Mm_s, Bij_s, Bi0_s, P0_s, pm_s, robot)
    u0_spart = -np.linalg.solve(H0_s, H0m_s @ qd_cfg)
    wb_spart = u0_spart[:3]
    vb_spart = u0_spart[3:]

    # MuJoCo: zero base velocity, nonzero joint velocity
    mj_data3 = mujoco.MjData(mj_model)
    mj_data3.qpos[:3] = 0
    mj_data3.qpos[3] = 1.0
    mj_data3.qpos[4:7] = 0
    mj_data3.qpos[7:7+n_q] = q_cfg
    mj_data3.qvel[:] = 0
    mj_data3.qvel[6:6+n_q] = qd_cfg
    mj_data3.ctrl[:] = 0

    mujoco.mj_forward(mj_model, mj_data3)

    # Step once
    mujoco.mj_step(mj_model, mj_data3)
    wb_mj_after = mj_data3.qvel[3:6]
    vb_mj_after = mj_data3.qvel[0:3]

    print(f"Config: {desc}")
    print(f"  SPART wb prediction: {wb_spart}")
    print(f"  SPART vb prediction: {vb_spart}")
    print(f"  MuJoCo wb after 1 step (dt=0.01): {wb_mj_after}")
    print(f"  MuJoCo vb after 1 step (dt=0.01): {vb_mj_after}")

    # Direction similarity
    if np.linalg.norm(wb_spart) > 1e-8 and np.linalg.norm(wb_mj_after) > 1e-8:
        cos_sim = np.dot(wb_spart, wb_mj_after) / (np.linalg.norm(wb_spart) * np.linalg.norm(wb_mj_after))
        print(f"  Angular vel direction cosine similarity: {cos_sim:.6f}")
    print()


# ============================================================================
# TEST 3c: Full momentum check - compute total angular momentum in both
# ============================================================================
separator("TEST 3c: TOTAL ANGULAR MOMENTUM COMPARISON")

print("\nCompute total angular momentum from SPART and MuJoCo at same state.\n")

q_cfg = np.array([0.3, -0.5, 0.2, 0.1, -0.4, 0.6])
qd_cfg = np.array([0.5, 0.3, -0.2, 0.1, 0.4, -0.3])

# SPART: compute angular momentum = H0 * u0 + H0m * qd (should be 0 if u0 from conservation)
RJ_s, RL_s, rJ_s, rL_s, e_s, g_s = spart_np.kinematics(R0, r0, q_cfg, robot)
Bij_s, Bi0_s, P0_s, pm_s = spart_np.diff_kinematics(R0, r0, rL_s, e_s, g_s, robot)
I0_s, Im_s = spart_np.inertia_projection(R0, RL_s, robot)
M0_s, Mm_s = spart_np.mass_composite_body(I0_s, Im_s, Bij_s, Bi0_s, robot)
H0_s, H0m_s, Hm_s = spart_np.generalized_inertia_matrix(M0_s, Mm_s, Bij_s, Bi0_s, P0_s, pm_s, robot)

u0_spart = -np.linalg.solve(H0_s, H0m_s @ qd_cfg)
momentum_spart = H0_s @ u0_spart + H0m_s @ qd_cfg
print(f"SPART total momentum (should be ~0): {momentum_spart.flatten()}")

# Generalized inertia matrices
print(f"\nSPART H0 (6x6):\n{H0_s}")
print(f"\nSPART H0m (6x{n_q}):\n{H0m_s}")

# MuJoCo: compute full mass matrix and compare
mj_data4 = mujoco.MjData(mj_model)
mj_data4.qpos[:3] = 0
mj_data4.qpos[3] = 1.0
mj_data4.qpos[4:7] = 0
mj_data4.qpos[7:7+n_q] = q_cfg
mj_data4.qvel[:] = 0
mujoco.mj_forward(mj_model, mj_data4)

# Get MuJoCo mass matrix
nv = mj_model.nv  # number of velocity DOFs
M_mj = np.zeros((nv, nv))
mujoco.mj_fullM(mj_model, M_mj, mj_data4.qM)

print(f"\nMuJoCo full mass matrix ({nv}x{nv}):")
print(f"  M[0:6, 0:6] (base-base block):\n{M_mj[0:6, 0:6]}")
print(f"\n  M[0:6, 6:] (base-joint coupling, analogous to H0m):\n{M_mj[0:6, 6:]}")

# Compare H0 vs M_mj[0:6, 0:6]
H0_diff = np.max(np.abs(H0_s - M_mj[0:6, 0:6]))
print(f"\n  |H0_spart - M_mj[0:6,0:6]| = {H0_diff:.6e}")

# Compare H0m vs M_mj[0:6, 6:]
H0m_diff = np.max(np.abs(H0m_s - M_mj[0:6, 6:]))
print(f"  |H0m_spart - M_mj[0:6,6:]| = {H0m_diff:.6e}")

# Compare Hm vs M_mj[6:, 6:]
Hm_diff = np.max(np.abs(Hm_s - M_mj[6:, 6:]))
print(f"  |Hm_spart - M_mj[6:,6:]| = {Hm_diff:.6e}")

if H0_diff > 0.01 or H0m_diff > 0.01 or Hm_diff > 0.01:
    print("\n  *** SIGNIFICANT MASS MATRIX MISMATCH DETECTED ***")
    print("  This is likely the PRIMARY cause of the dynamics discrepancy!")

    # Detailed comparison
    print(f"\n  Detailed H0 comparison:")
    print(f"  SPART H0:\n{H0_s}")
    print(f"  MuJoCo M[0:6,0:6]:\n{M_mj[0:6, 0:6]}")

    print(f"\n  Detailed H0m comparison:")
    print(f"  SPART H0m:\n{H0m_s}")
    print(f"  MuJoCo M[0:6,6:]:\n{M_mj[0:6, 6:]}")
else:
    print("\n  Mass matrices match well!")

# Compute MuJoCo's momentum-conservation base velocity
u0_mj = -np.linalg.solve(M_mj[0:6, 0:6], M_mj[0:6, 6:] @ qd_cfg)
wb_mj_conserv = u0_mj[3:6]  # Note: MuJoCo velocity order is [vx,vy,vz,wx,wy,wz]
# Actually MuJoCo qvel order for freejoint is: [vx,vy,vz, wx,wy,wz]
# SPART u0 order is: [wx,wy,wz, vx,vy,vz]

print(f"\n--- Velocity ordering check ---")
print(f"SPART u0 = [w(3), v(3)]: {u0_spart.flatten()}")
print(f"MuJoCo u0 = [v(3), w(3)]: {u0_mj.flatten()}")
print(f"  SPART w: {u0_spart[:3].flatten()}")
print(f"  MuJoCo w: {u0_mj[3:6].flatten()}")
print(f"  SPART v: {u0_spart[3:].flatten()}")
print(f"  MuJoCo v: {u0_mj[:3].flatten()}")
print(f"  |w_spart - w_mujoco| = {np.linalg.norm(u0_spart[:3] - u0_mj[3:6]):.6e}")
print(f"  |v_spart - v_mujoco| = {np.linalg.norm(u0_spart[3:] - u0_mj[:3]):.6e}")


# ============================================================================
# SUMMARY
# ============================================================================
separator("SUMMARY")
print("""
Diagnostics complete. Check above for:
  1. Mass/inertia mismatches between SPART and MuJoCo
  2. Quaternion convention errors in the conversion pipeline
  3. Mass matrix (H0, H0m, Hm) differences -> base velocity discrepancy
""")
