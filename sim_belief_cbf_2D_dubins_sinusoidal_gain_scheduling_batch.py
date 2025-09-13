import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
# import mplcursors  # for enabling data cursor in matplotlib plots
import numpy as np
from jax import jit
from jaxopt import BoxOSQP as OSQP
from tqdm import tqdm

from cbfs import BeliefCBF
from cbfs import gain_schedule_ctrl, s_trajectory
from dynamics import *
from estimators import *
from sensor import noisy_sensor_mult as sensor
from scipy.stats import chi2
# from sensor import ubiased_noisy_sensor as sensor

# For logging results
import os
from datetime import datetime
import json

def getSimParams():

    lin_vel = 5.0
    Q_scale_factor = 0.001
        
    sim_params = {
        "dt": 0.001,
        "T": 30000,  # EKF BECOMES UNSTABLE > 30000 !!!
        "dynamics": DubinsMultCtrlDynamics(Q = jnp.array([(0.1*Q_scale_factor)**2,
                                                          (0.1*Q_scale_factor)**2,
                                                          (0.005*Q_scale_factor)**2,
                                                          (0.005*Q_scale_factor)**2])),  # or UnicycleDynamics()
        "wall_y": 5.0,
        "lin_vel": lin_vel,
        "x_init": [0.0, 0.0, lin_vel, 0.45],
        "estimator_type": "GEKF",
        "CBF_ON": True,
    }

    sensor_params = {
        "mu_u": 0.1,
        "sigma_u": jnp.sqrt(0.01),   # std deviation
        "mu_v": 0.001,
        "sigma_v": jnp.sqrt(0.0005), # std deviation
        "sensor_update_frequency": 0.1, # Hz
        "obs_fun": lambda x: jnp.array([x[1]])  # Observation Function
    }

    control_params = {
        "U_MAX": np.array([1.0, 1.0]),
        "clf_gain": 20.0,            # CLF linear gain
        "clf_slack_penalty": 50.0,
        "cbf_gain": 50.0,            # CBF linear gain
        "CBF_ON": True,
    }

    belief_cbf_params = {
        "alpha1": jnp.array([0.0, -1.0, 0.0, 0.0]),
        "alpha2": jnp.array([0.0,  1.0, 0.0, 0.0]),
        "beta": jnp.array([-sim_params["wall_y"]]),
        "delta": 0.001
    }

    return sim_params, sensor_params, control_params, belief_cbf_params

def simulate(sim_params, sensor_params, control_params, belief_cbf_params, key=None):
    
    # --- Unpack sim_params ---
    dt = sim_params["dt"]
    T = sim_params["T"]
    dynamics = sim_params["dynamics"]
    wall_y = sim_params["wall_y"]
    lin_vel = sim_params["lin_vel"]
    x_init = sim_params["x_init"]
    estimator_type = sim_params["estimator_type"]

    # --- Unpack sensor_params ---
    mu_u = sensor_params["mu_u"]
    sigma_u = sensor_params["sigma_u"]
    mu_v = sensor_params["mu_v"]
    sigma_v = sensor_params["sigma_v"]
    sensor_update_frequency = sensor_params["sensor_update_frequency"]
    obs_fun = sensor_params["obs_fun"]

    # --- Unpack control_params ---
    U_MAX = control_params["U_MAX"]
    clf_gain = control_params["clf_gain"]
    clf_slack_penalty = control_params["clf_slack_penalty"]
    cbf_gain = control_params["cbf_gain"]
    CBF_ON = control_params["CBF_ON"]

    # --- Unpack belief_cbf_params ---
    alpha1 = belief_cbf_params["alpha1"]
    alpha2 = belief_cbf_params["alpha2"]
    beta   = belief_cbf_params["beta"]
    delta  = belief_cbf_params["delta"]

    cbf = BeliefCBF(alpha1, beta, delta, dynamics.state_dim)
    cbf2 = BeliefCBF(alpha2, beta, delta, dynamics.state_dim)

    # Initial state (truth)
    x_true = jnp.array(x_init)  # Start position

    if sensor.__name__ == "noisy_sensor_mult":
         x_initial_measurement = sensor(x_true, 0, mu_u, sigma_u, mu_v, sigma_v, key=key) # mult_noise
    elif sensor.__name__ == "unbiased_noisy_sensor":
        x_initial_measurement = sensor(x_true, t=0, std=sigma_v, key=key) # unbiased_fixed_noise

    if estimator_type == "EKF":
        estimator = EKF(dynamics, dt, x_init=x_initial_measurement, R=jnp.square(sigma_v)*jnp.eye(dynamics.state_dim), h=obs_fun)
    elif estimator_type == "GEKF":
        estimator = GEKF(dynamics, dt, mu_u, sigma_u, mu_v, sigma_v, x_init=x_initial_measurement, h=obs_fun)

    # OSQP solver instance
    solver = OSQP()

    m = len(U_MAX) # control dim
    var_dim = m + 1 # ctrl dim + slack variable

    # @jit
    def solve_qp(b, goal_loc):

        x_estimated, _ = cbf.extract_mu_sigma(b)

        u_nom = gain_schedule_ctrl(v_r=lin_vel,
                               x = x_estimated,
                               x_d = jnp.insert(goal_loc, 2, lin_vel),
                               lambda1=1.0, a1=16.0, a2=100.0)

        u_nom = jnp.clip(u_nom, -U_MAX, U_MAX)

        # Compute CBF components
        h = cbf.h_b(b)
        L_f_hb, _, L_f_2_h, Lg_Lf_h, _, _ = cbf.h_dot_b(b, dynamics) # ∇h(x)

        L_f_h = L_f_hb

        rhs, L_f_h, _ = cbf.h_b_r2_RHS(h, L_f_h, L_f_2_h, cbf_gain)

        # Compute CBF2 components
        h_2 = cbf2.h_b(b)
        L_f_hb_2, _, L_f_2_h_2, Lg_Lf_h_2, _, _ = cbf2.h_dot_b(b, dynamics) # ∇h(x)

        L_f_h_2 = L_f_hb_2

        rhs2, _, _ = cbf2.h_b_r2_RHS(h_2, L_f_h_2, L_f_2_h_2, cbf_gain)

        A = jnp.vstack([
            jnp.concatenate([-Lg_Lf_h, jnp.array([0.0])]), # -LgLfh u       <= -[alpha1 alpha2].T @ [Lfh h] + Lf^2h
            jnp.concatenate([-Lg_Lf_h_2, jnp.array([0.0])]), # 2nd CBF
            jnp.eye(var_dim)
        ])

        u = jnp.hstack([
            (rhs).squeeze(),                            # CBF constraint: rhs = -[alpha1 alpha2].T [Lfh h] + Lf^2h
            (rhs2).squeeze(),                           # 2nd CBF constraint
            U_MAX, 
            jnp.inf # no upper limit on slack
        ])

        l = jnp.hstack([
            -jnp.inf, # No lower limit on CBF condition
            -jnp.inf, # 2nd CBF
            -U_MAX,
            0.0 # slack can't be negative
        ])

        if CBF_ON:
            A, u, l = A, u, l
        else:
            A, u, l = A[2:], u[2:], l[2:]

        # Define Q matrix: Minimize ||u||^2 and slack (penalty*delta^2)
        Q = jnp.eye(var_dim)
        Q = Q.at[-1, -1].set(2*clf_slack_penalty)

        c = jnp.append(-2.0*u_nom.flatten(), 0.0)

        # Solve the QP using jaxopt OSQP
        sol = solver.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
        
        return sol, h, h_2, u_nom

    solve_qp_cpu = jit(solve_qp, backend='cpu')

    x_traj = []  # Store trajectory
    x_meas = [] # Measurements
    x_est = [] # Estimates
    u_traj = []  # Store controls
    u_nom_list = []
    cbf_values = []
    kalman_gains = []
    covariances = []
    in_covariances = [] # Innovation Covariance of EKF
    prob_leave = [] # Probability of leaving safe set
    x_nom = [] # Store nominal trajectory
    NEES_list = []
    NIS_list = []

    x_estimated, p_estimated = estimator.get_belief()
    x_measured = x_initial_measurement

    # Generate goal trajectory
    t_vec = jnp.arange(0.0, T + 1.0, 1.0)*dt
    goal_x_nom = s_trajectory(t_vec, A=5.0, omega=0.25, v=lin_vel).T

    traj_idx = 0
    goal_loc = goal_x_nom[traj_idx]

    # Simulation loop
    for t in tqdm(range(T), desc="Simulation Progress"):

        x_traj.append(x_true)

        belief = cbf.get_b_vector(x_estimated, p_estimated)

        # target_goal_loc = sinusoidal_trajectory(t*dt, A=goal[1], omega=1.0, v=lin_vel)

        sol, h, h_2, u_nom = solve_qp_cpu(belief, goal_loc)
        u_nom_list.append(u_nom)
        # sol, h, h_2 = solve_qp(belief, goal_loc)

        # clf_values.append(V)
        cbf_values.append([h, h_2])

        u_sol = jnp.array([sol.primal[0][:2]]).reshape(-1, 1)

        # u_sol = sol
        u_opt = jnp.clip(u_sol, -U_MAX.reshape(-1,1), U_MAX.reshape(-1,1))
        # u_opt = u_sol

        # Apply control to the true state (x_true)
        x_true = x_true + dt * dynamics.x_dot(x_true, u_opt)

        estimator.predict(u_opt)

        # update measurement and estimator belief
        if t > 0 and t%(1/sensor_update_frequency) == 0:
            # obtain current measurement
            if sensor.__name__ == "noisy_sensor_mult":
                x_measured = sensor(x_true, t, mu_u, sigma_u, mu_v, sigma_v, key=key)        
            elif sensor.__name__ == "unbiased_noisy_sensor":
                x_measured = sensor(x_true, t, sigma_v, key=key) # unbiased_fixed_noise

            if estimator.name == "GEKF":
                estimator.update(x_measured)
                prob_leave.append((t, estimator.prob_leaving_CVaR_GEKF(alpha1, delta, beta)))

            if estimator.name == "EKF":
                estimator.update(x_measured)
                prob_leave.append((t, estimator.prob_leaving_VaR_EKF(alpha1, delta, beta)))

        NEES_list.append(estimator.NEES(x_true))
        NIS_list.append(estimator.NIS())
        x_estimated, p_estimated = estimator.get_belief()

        goal_loc = goal_x_nom[t]

        # Store for plotting
        u_traj.append(u_opt)
        x_meas.append(x_true.at[1].set(obs_fun(x_measured)[0]))
        x_est.append(x_estimated)
        kalman_gains.append(estimator.K)
        covariances.append(p_estimated)
        in_covariances.append(estimator.in_cov)
        x_nom.append(goal_loc[:2])

    # Convert to numpy arrays for plotting
    x_traj = np.array(x_traj).squeeze()
    x_meas = np.array(x_meas).squeeze()
    x_est = np.array(x_est).squeeze()
    u_traj = np.array(u_traj)
    u_nom_list = jnp.array(u_nom_list)
    cbf_values = jnp.array(cbf_values) 
    covariance_traces = [jnp.trace(P) for P in covariances]

    x_nom = np.array(x_nom).squeeze()

    h_vals   = cbf_values[:, 0]
    h2_vals  = cbf_values[:, 1]
    time = dt*np.arange(T)  # assuming x_meas.shape[0] == N

    fig, axs = plt.subplots(4, 2, figsize=(14, 18))  # 3 rows, 2 columns
    plt.subplots_adjust(
        left=0.08,   # reduce left margin
        right=0.95,  # reduce right margin
        top=0.95,    # reduce top margin
        bottom=0.08, # reduce bottom margin
        hspace=0.2   # spacing between rows
    )
    axs = axs.ravel()  # flatten to 1D for easy indexing

    # 1. Trajectories
    axs[0].scatter(x_meas[:, 0], x_meas[:, 1], color="green", marker="o", s=1.0, alpha=0.5, label="Measurements")
    axs[0].plot(x_traj[:, 0], x_traj[:, 1], "b-", label="True trajectory")
    axs[0].plot(x_est[:, 0], x_est[:, 1], color="orange", label="Estimated trajectory")
    axs[0].plot(x_nom[:, 0], x_nom[:, 1], "black", label="Nominal trajectory")
    axs[0].axhline(y=wall_y, color="red", linestyle="dashed", linewidth=1, label="Obstacle")
    axs[0].axhline(y=-wall_y, color="red", linestyle="dashed", linewidth=1)
    axs[0].set_xlabel("x [m]"); axs[0].set_ylabel("y [m]")
    axs[0].set_title(f"2D Trajectory ({estimator.name})")
    axs[0].legend(); axs[0].grid()

    # 2. Longitudinal acceleration
    axs[1].plot(time, u_traj[:, 0], label="a")
    axs[1].plot(time, u_nom_list[:, 0], label="a_nom")
    axs[1].set_xlabel("Time [s]"); axs[1].set_ylabel("Acceleration [m/s²]")
    axs[1].set_title(f"Longitudinal Acceleration ({estimator.name})")
    axs[1].legend(); axs[1].grid()

    # 3. Yaw rate
    axs[2].plot(time, u_traj[:, 1], label="ω")
    axs[2].plot(time, u_nom_list[:, 1], label="ω_nom")
    axs[2].set_xlabel("Time [s]"); axs[2].set_ylabel("Yaw rate [rad/s]")
    axs[2].set_title(f"Yaw Rate ({estimator.name})")
    axs[2].legend(); axs[2].grid()

    # 4. Barrier functions
    h_vals = cbf_values[:, 0]
    h2_vals = cbf_values[:, 1]
    axs[3].plot(time, h_vals, color='red', label=f"CBF: y < {wall_y}")
    axs[3].plot(time, h2_vals, color='purple', label=f"CBF: y > -{wall_y}")
    axs[3].set_xlabel("Time [s]"); axs[3].set_ylabel("Barrier value [m]")
    axs[3].set_title(f"Barrier Function Values ({estimator.name})")
    axs[3].legend(); axs[3].grid()

    # 5. Trace of covariance
    covariance_traces = [jnp.trace(P) for P in covariances]
    axs[4].plot(time, np.array(covariance_traces), color="red", linestyle="-", label="Trace of Covariance")
    axs[4].set_xlabel("Time [s]"); axs[4].set_ylabel("Trace value")
    axs[4].set_title(f"Trace of Covariance Over Time ({estimator.name})")
    axs[4].legend(); axs[4].grid()

    # 6. NEES with chi-square bounds
    state_dim = dynamics.state_dim
    confidence = 0.95
    alpha = 1 - confidence
    lower = chi2.ppf(alpha/2, df=state_dim)
    upper = chi2.ppf(1 - alpha/2, df=state_dim)
    axs[5].axhline(lower, color="red", linestyle="--", label=f"Lower {confidence*100:.1f}% bound")
    axs[5].axhline(upper, color="green", linestyle="--", label=f"Upper {confidence*100:.1f}% bound")
    axs[5].axhline(state_dim, color="purple", linestyle="--", label="Expected value")
    axs[5].plot(time, np.array(NEES_list), color="green", linestyle="-", label="NEES")
    axs[5].set_xlabel("Time [s]"); axs[5].set_ylabel("NEES value")
    axs[5].set_title(f"NEES over Time ({estimator.name})")
    axs[5].legend(); axs[5].grid()

    # 7. Probability of leaving safe set
    times = np.array([t for t, _ in prob_leave])
    probs = np.array([p for _, p in prob_leave])
    axs[6].plot(times/1000, probs, color="blue", linestyle="dashed", label="Probs leaving (CBF 1)")
    axs[6].set_title(f"Probability of Leaving/ Staying ({estimator.name})")
    axs[6].set_xlabel("Time [s]"); axs[6].set_ylabel("Probability")
    axs[6].legend(); axs[6].grid()

    # plt.tight_layout()
    fig.savefig("Sin Gain Scheduling.png", dpi=300) 

    nees_arr = np.array(NEES_list)
    nees_coverage = 100 * np.mean((nees_arr >= lower) & (nees_arr <= upper))
    cov_np    = np.asarray(covariance_traces)
    probs_np  = np.asarray([p for _, p in prob_leave])  # if not already array

    # Define Metrics
    metrics = {
        # Control/Estimation
        "num_est_exceed":  np.sum(np.abs(x_est[:, 1]) > wall_y),
        "num_true_exceed": np.sum(np.abs(x_traj[:, 1]) > wall_y),
        "max_true":            np.max(x_traj[:, 1]),
        "min_true":        float(np.min(x_traj[:, 1])),
        "max_est":             np.max(x_est[:, 1]),
        "min_est":         float(np.min(x_est[:, 1])),
        "tracking_rmse":       np.sqrt(np.mean((x_traj - x_est) ** 2)),
        "avg_acc": jnp.mean(u_traj[:, 0]),
        "avg_w": jnp.mean(u_traj[:, 1]),
        "pct_h_vals_lt0":  float(100.0 * np.mean(h_vals < 0)),
        "pct_h2_vals_lt0": float(100.0 * np.mean(h2_vals < 0)),

        # NEES / covariance
        "nees_coverage_%": float(nees_coverage),
        "mean_trace_P":    float(np.mean(cov_np)),
        "final_trace_P":   float(cov_np[-1]),

        # probability bounds
        "mean_prob_bound": float(np.mean(probs_np)),
        "max_prob_bound":  float(np.max(probs_np)),

        # Add max prob of leaving CBF 2
        # Add max prob of staying in CBF 1
        # Add max prob of staying in CBF 2
        # Add mean prob of leaving CBF 2
        # Add mean prob of staying in CBF 1
        # Add mean prob of staying in CBF 2
    }

    return metrics, fig

def printMetrics(metrics): 
    print("\n--- Results ---")
    for k, v in metrics.items():
        print(f"{k:20s}: {v}")

def printSimParams(sim_params, sensor_params, control_params, belief_cbf_params):
    def print_dict(d, name="dict"):
        print(f"\n{name}:")
        for key, value in d.items():
            print(f"  {key}: {value}")

    print("\n--- Simulation Parameters ---")
    print_dict(sim_params, "Sim Params")

    print("\n-- Sensor Params ---")
    print_dict(sensor_params, "Sensor Params")
    
    print("\n--- Environment Setup ---")
    print_dict(control_params, "Env Setup")
 
    print("\n--- Control Parameters ---")
    print_dict(belief_cbf_params, "Control Params")

    if sim_params["CBF_ON"]:
        print("CBF: ON")
    else:
        print("CBF: OFF")

    print("\n--- Belief CBF Parameters ---")

start_idx = 1
end_idx = 50

save_freq = 5

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_results", "GEKF_acc", timestamp)
os.makedirs(base_path, exist_ok=True)

save_path = os.path.join(base_path, "summary.json")

if __name__ == "__main__":

    sim_params, sensor_params, control_params, belief_cbf_params = getSimParams()

    printSimParams(sim_params, sensor_params, control_params, belief_cbf_params)

    indices = list(range(start_idx, end_idx + 1))

    metrics_sum = {}

    for i in tqdm(indices, desc="Running simulations"):

        key = jax.random.PRNGKey(i)

        metrics, fig = simulate(sim_params, sensor_params, control_params, belief_cbf_params, key=key)
        
        # printMetrics(metrics) # For a single simuation

        # Save results for this iteration
        with open(os.path.join(base_path, f"{i}_result.json"), "w") as f:
                 json.dump({"metrics": {k: str(v) for k, v in metrics.items()}}, f, indent=2)
        fig.savefig(os.path.join(base_path, f"{i}_result.png"), dpi=300) 

        if not metrics_sum:
            metrics_sum = {k: float(v) for k, v in metrics.items()}
        else:
            for k in metrics:
                metrics_sum[k] += float(metrics[k])

        # Save every nth run
        if (i - start_idx + 1) % save_freq == 0:
            avg_metrics = {k: v / (i - start_idx + 1) for k, v in metrics_sum.items()}
            with open(save_path, "w") as f:
                json.dump({
                    "start_idx": start_idx,
                    "end_idx": i,
                    "sim_params": {k: str(v) for k, v in sim_params.items()},
                    "sensor_params": {k: str(v) for k, v in sensor_params.items()},
                    "control_params": {k: str(v) for k, v in control_params.items()},
                    "belief_cbf_params": {k: str(v) for k, v in belief_cbf_params.items()},
                    "avg_metrics": avg_metrics
                }, f, indent=2)

    # Final average
    avg_metrics = {k: v / len(indices) for k, v in metrics_sum.items()}

    print(f"\n --- Results for runs {start_idx} to {end_idx} ({end_idx - start_idx + 1} runs) ---")
    printMetrics(avg_metrics)