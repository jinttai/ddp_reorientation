"""
Sweep CTC PD gains to minimize joint tracking error in MuJoCo simulation.
"""

import os
import sys
import numpy as np

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import mujoco
from scenario import SCENARIO, get_goal_quaternion

np.set_printoptions(precision=4, suppress=True)


def quat_to_euler_wxyz(q):
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = 2*(w*y - z*x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])


def quat_to_euler_xyzw(q):
    x, y, z, w = q
    return quat_to_euler_wxyz([w, x, y, z])


def run_sim(Kp, Kd, model, X_ddp, U_ddp, q_ddp, qd_ddp, quat_ddp_wxyz, w_ddp, qdd_ddp):
    n_q = SCENARIO['n_q']
    dt_ddp = SCENARIO['dt']
    T = U_ddp.shape[0]
    total_time = T * dt_ddp

    data = mujoco.MjData(model)
    data_ref = mujoco.MjData(model)

    dt_sim = model.opt.timestep
    n_sim_steps = int(total_time / dt_sim)

    # Initial state
    data.qpos[0:3] = 0
    data.qpos[3:7] = quat_ddp_wxyz[0]
    data.qpos[7:] = q_ddp[0]
    data.qvel[0:3] = 0
    data.qvel[3:6] = w_ddp[0]
    data.qvel[6:] = qd_ddp[0]
    mujoco.mj_forward(model, data)

    sim_q_log = []
    sim_qd_log = []
    sim_quat_log = []

    for step in range(n_sim_steps):
        sim_time = step * dt_sim
        idx = int(sim_time / dt_ddp)
        if idx >= T:
            idx = T - 1
        alpha = (sim_time - idx * dt_ddp) / dt_ddp
        alpha = np.clip(alpha, 0.0, 1.0)

        q_ref = (1 - alpha) * q_ddp[idx] + alpha * q_ddp[idx + 1]
        qd_ref = (1 - alpha) * qd_ddp[idx] + alpha * qd_ddp[idx + 1]
        qdd_ref = (1 - alpha) * qdd_ddp[idx] + alpha * qdd_ddp[idx + 1]

        q_err = q_ref - data.qpos[7:]
        qd_err = qd_ref - data.qvel[6:]
        qacc_des = qdd_ref + Kp * q_err + Kd * qd_err

        data_ref.qpos[:] = data.qpos[:]
        data_ref.qvel[:] = data.qvel[:]
        data_ref.qacc[0:6] = 0
        data_ref.qacc[6:] = qacc_des
        mujoco.mj_inverse(model, data_ref)
        data.ctrl[:] = data_ref.qfrc_inverse[6:]

        sim_q_log.append(data.qpos[7:].copy())
        sim_qd_log.append(data.qvel[6:].copy())
        sim_quat_log.append(data.qpos[3:7].copy())

        mujoco.mj_step(model, data)

    sim_q = np.array(sim_q_log)
    sim_qd = np.array(sim_qd_log)
    sim_quat = np.array(sim_quat_log)
    sim_t = np.arange(n_sim_steps) * dt_sim

    # Resample to DDP timesteps
    sim_q_at_ddp = np.zeros_like(q_ddp)
    sim_euler_at_ddp = np.zeros((T + 1, 3))
    euler_ddp_deg = np.zeros((T + 1, 3))

    for k in range(T + 1):
        t_target = k * dt_ddp
        idx_sim = min(int(t_target / dt_sim), len(sim_t) - 1)
        sim_q_at_ddp[k] = sim_q[idx_sim]

        sq = sim_quat[idx_sim]
        sq /= np.linalg.norm(sq) + 1e-8
        sim_euler_at_ddp[k] = np.degrees(quat_to_euler_wxyz(sq))

        dq = X_ddp[k, 2*n_q:]
        dq /= np.linalg.norm(dq) + 1e-8
        euler_ddp_deg[k] = np.degrees(quat_to_euler_xyzw(dq))

    q_err_deg = np.rad2deg(sim_q_at_ddp - q_ddp)
    orient_err = sim_euler_at_ddp - euler_ddp_deg

    rms_q = np.sqrt(np.mean(q_err_deg ** 2, axis=0))
    rms_orient = np.sqrt(np.mean(orient_err ** 2, axis=0))
    total_joint_rms = np.sqrt(np.mean(rms_q ** 2))

    # Final orientation
    final_sim_euler = sim_euler_at_ddp[-1]
    final_ddp_euler = euler_ddp_deg[-1]

    return {
        'rms_q': rms_q,
        'rms_orient': rms_orient,
        'total_joint_rms': total_joint_rms,
        'final_sim_euler': final_sim_euler,
        'final_ddp_euler': final_ddp_euler,
        'max_q_err': np.max(np.abs(q_err_deg)),
        'max_orient_err': np.max(np.abs(orient_err)),
    }


def main():
    results_dir = os.path.join(ROOT_DIR, 'results')
    X_ddp = np.load(os.path.join(results_dir, 'trajectory_casadi_ddp_states.npy'))
    U_ddp = np.load(os.path.join(results_dir, 'trajectory_casadi_ddp_controls.npy'))

    n_q = SCENARIO['n_q']
    dt_ddp = SCENARIO['dt']
    T = U_ddp.shape[0]

    q_ddp = X_ddp[:, :n_q]
    qd_ddp = X_ddp[:, n_q:2*n_q]
    quat_ddp_xyzw = X_ddp[:, 2*n_q:]
    quat_ddp_wxyz = quat_ddp_xyzw[:, [3, 0, 1, 2]]

    # Base angular velocity from quaternion differences
    w_ddp = np.zeros((T + 1, 3))
    for i in range(T):
        q_cur = quat_ddp_wxyz[i]
        q_nxt = quat_ddp_wxyz[i + 1]
        q_inv = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]])
        aw, av = q_inv[0], q_inv[1:]
        bw, bv = q_nxt[0], q_nxt[1:]
        rw = aw * bw - np.dot(av, bv)
        rv = aw * bv + bw * av + np.cross(av, bv)
        rw = np.clip(rw, -1.0, 1.0)
        if rw > 0.999999:
            w_ddp[i] = (2.0 / dt_ddp) * rv
        else:
            theta = 2.0 * np.arccos(rw)
            sin_half = np.sin(theta / 2.0)
            if sin_half < 1e-6:
                w_ddp[i] = np.zeros(3)
            else:
                w_ddp[i] = (theta / dt_ddp) * (rv / sin_half)
    w_ddp[-1] = w_ddp[-2]

    qdd_ddp = np.zeros((T + 1, n_q))
    qdd_ddp[:T] = U_ddp
    qdd_ddp[T] = U_ddp[-1]

    xml_path = os.path.join(ROOT_DIR, 'assets', 'spacerobot_cjt.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)

    q_goal = get_goal_quaternion()
    goal_euler_deg = np.degrees(quat_to_euler_xyzw(q_goal))

    # Gain sweep
    # Kp_values and Kd computed as critical damping: Kd = 2*sqrt(Kp)
    Kp_values = [500, 1000, 2000, 5000, 10000, 20000, 50000]

    print(f"{'Kp':<10} {'Kd':<10} {'Joint RMS(deg)':<16} {'Orient RMS R/P/Y (deg)':<35} {'Final Orient R/P/Y (deg)':<40} {'Max Jerr':<10} {'Max Oerr':<10}")
    print("-" * 145)

    best_score = np.inf
    best_kp = None
    best_kd = None
    best_result = None

    for kp_val in Kp_values:
        kd_val = 2.0 * np.sqrt(kp_val)  # critical damping
        Kp = np.full(n_q, kp_val)
        Kd = np.full(n_q, kd_val)

        result = run_sim(Kp, Kd, model, X_ddp, U_ddp, q_ddp, qd_ddp, quat_ddp_wxyz, w_ddp, qdd_ddp)

        orient_str = f"R={result['rms_orient'][0]:.4f}, P={result['rms_orient'][1]:.4f}, Y={result['rms_orient'][2]:.4f}"
        final_str = f"R={result['final_sim_euler'][0]:.2f}, P={result['final_sim_euler'][1]:.2f}, Y={result['final_sim_euler'][2]:.2f}"

        score = result['total_joint_rms'] + np.sqrt(np.mean(result['rms_orient']**2))

        marker = ""
        if score < best_score:
            best_score = score
            best_kp = kp_val
            best_kd = kd_val
            best_result = result
            marker = " <-- best"

        print(f"{kp_val:<10.0f} {kd_val:<10.1f} {result['total_joint_rms']:<16.4f} {orient_str:<35} {final_str:<40} {result['max_q_err']:<10.4f} {result['max_orient_err']:<10.4f}{marker}")

    print(f"\n{'='*80}")
    print(f"Best: Kp={best_kp}, Kd={best_kd:.1f}")
    print(f"  Joint RMS: {best_result['total_joint_rms']:.4f} deg")
    print(f"  Orient RMS: R={best_result['rms_orient'][0]:.4f}, P={best_result['rms_orient'][1]:.4f}, Y={best_result['rms_orient'][2]:.4f} deg")
    print(f"  Final: R={best_result['final_sim_euler'][0]:.2f}, P={best_result['final_sim_euler'][1]:.2f}, Y={best_result['final_sim_euler'][2]:.2f}")
    print(f"  Goal:  R={goal_euler_deg[0]:.2f}, P={goal_euler_deg[1]:.2f}, Y={goal_euler_deg[2]:.2f}")


if __name__ == '__main__':
    main()
