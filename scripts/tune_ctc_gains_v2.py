"""
Fine-grained CTC gain tuning with varying Kp/Kd ratios.
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
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
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

    data.qpos[0:3] = 0
    data.qpos[3:7] = quat_ddp_wxyz[0]
    data.qpos[7:] = q_ddp[0]
    data.qvel[0:3] = 0
    data.qvel[3:6] = w_ddp[0]
    data.qvel[6:] = qd_ddp[0]
    mujoco.mj_forward(model, data)

    sim_q_log = []
    sim_quat_log = []
    unstable = False

    for step in range(n_sim_steps):
        sim_time = step * dt_sim
        idx = min(int(sim_time / dt_ddp), T - 1)
        alpha = np.clip((sim_time - idx * dt_ddp) / dt_ddp, 0, 1)

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
        sim_quat_log.append(data.qpos[3:7].copy())

        mujoco.mj_step(model, data)

        if np.any(np.isnan(data.qpos)) or np.any(np.abs(data.qpos) > 1e6):
            unstable = True
            break

    if unstable:
        return None

    sim_q = np.array(sim_q_log)
    sim_quat = np.array(sim_quat_log)

    sim_q_at_ddp = np.zeros_like(q_ddp)
    sim_euler_at_ddp = np.zeros((T + 1, 3))
    euler_ddp_deg = np.zeros((T + 1, 3))

    for k in range(T + 1):
        t_target = k * dt_ddp
        idx_sim = min(int(t_target / dt_sim), len(sim_q) - 1)
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

    return {
        'rms_q': rms_q,
        'rms_orient': rms_orient,
        'total_joint_rms': np.sqrt(np.mean(rms_q ** 2)),
        'total_orient_rms': np.sqrt(np.mean(rms_orient ** 2)),
        'final_sim_euler': sim_euler_at_ddp[-1],
        'final_ddp_euler': euler_ddp_deg[-1],
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
            w_ddp[i] = (theta / dt_ddp) * (rv / sin_half) if sin_half > 1e-6 else np.zeros(3)
    w_ddp[-1] = w_ddp[-2]

    qdd_ddp = np.zeros((T + 1, n_q))
    qdd_ddp[:T] = U_ddp
    qdd_ddp[T] = U_ddp[-1]

    xml_path = os.path.join(ROOT_DIR, 'assets', 'spacerobot_cjt.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    q_goal = get_goal_quaternion()
    goal_euler_deg = np.degrees(quat_to_euler_xyzw(q_goal))

    # Fine sweep
    configs = []
    # Uniform gains
    for kp in [300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000]:
        for kd_ratio in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            kd = kd_ratio * np.sqrt(kp)
            configs.append((np.full(n_q, kp), np.full(n_q, kd), f"Kp={kp}, Kd={kd:.1f} (r={kd_ratio})"))

    # Per-joint: higher for big joints (1-3), lower for wrist (4-6)
    for kp_big in [500, 1000, 2000, 3000, 5000]:
        for kp_small_ratio in [0.3, 0.5, 1.0]:
            kp_small = kp_big * kp_small_ratio
            kd_big = 3.0 * np.sqrt(kp_big)
            kd_small = 3.0 * np.sqrt(kp_small)
            Kp = np.array([kp_big]*3 + [kp_small]*3)
            Kd = np.array([kd_big]*3 + [kd_small]*3)
            configs.append((Kp, Kd, f"Kp=[{kp_big}/{kp_small:.0f}], Kd=[{kd_big:.0f}/{kd_small:.0f}]"))

    print(f"Testing {len(configs)} configurations...\n")
    print(f"{'Config':<45} {'JointRMS':<10} {'OrientRMS':<10} {'Score':<10} {'Final R/P/Y':<35}")
    print("-" * 115)

    best_score = np.inf
    best_config = None
    best_result = None
    best_gains = None

    for Kp, Kd, desc in configs:
        result = run_sim(Kp, Kd, model, X_ddp, U_ddp, q_ddp, qd_ddp, quat_ddp_wxyz, w_ddp, qdd_ddp)
        if result is None:
            print(f"{desc:<45} UNSTABLE")
            continue

        score = result['total_joint_rms'] + result['total_orient_rms']
        final = result['final_sim_euler']
        marker = ""
        if score < best_score:
            best_score = score
            best_config = desc
            best_result = result
            best_gains = (Kp.copy(), Kd.copy())
            marker = " ***"

        print(f"{desc:<45} {result['total_joint_rms']:<10.4f} {result['total_orient_rms']:<10.4f} {score:<10.4f} R={final[0]:.2f} P={final[1]:.2f} Y={final[2]:.2f}{marker}")

    print(f"\n{'='*80}")
    print(f"Best config: {best_config}")
    print(f"  Kp: {best_gains[0]}")
    print(f"  Kd: {best_gains[1]}")
    print(f"  Joint RMS: {best_result['total_joint_rms']:.4f} deg")
    print(f"  Orient RMS: R={best_result['rms_orient'][0]:.4f}, P={best_result['rms_orient'][1]:.4f}, Y={best_result['rms_orient'][2]:.4f} deg")
    print(f"  Final:  R={best_result['final_sim_euler'][0]:.2f}, P={best_result['final_sim_euler'][1]:.2f}, Y={best_result['final_sim_euler'][2]:.2f}")
    print(f"  Goal:   R={goal_euler_deg[0]:.2f}, P={goal_euler_deg[1]:.2f}, Y={goal_euler_deg[2]:.2f}")
    print(f"  Per-joint RMS (deg): {best_result['rms_q']}")


if __name__ == '__main__':
    main()
