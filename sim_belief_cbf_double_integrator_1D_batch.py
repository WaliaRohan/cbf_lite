import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mplcursors  # for enabling data cursor in matplotlib plots
import numpy as np
from jax import grad, jit, random
from jaxopt import BoxOSQP as OSQP
from tqdm import tqdm

from cbfs import BeliefCBF
from cbfs import clf_1D_doubleint as clf
from dynamics import *
from estimators import *
from sensor import noisy_sensor_mult as sensor

# For logging results
import os
from datetime import datetime
import json

def getSimParams():
    sim_params = {
    "dt": 0.001,
    "T": 10000,
    "dynamics": LinearDoubleIntegrator1D(),
    "wall_x": 6.0,
    "goal_x": 7.0,
    "x_init": [1.0, 0.0],
    "estimator_type": "EKF",
    "jax_device": "cpu" # gpu, or None (if you don't want to jit)
    }

    sensor_params = {
        "mu_u": 0.1,
        "sigma_u": jnp.sqrt(0.01), # std deviation
        "mu_v": 0.001,
        "sigma_v": jnp.sqrt(0.0005), # std deviation
        "sensor_update_frequency": 10,
    }

    control_params = {
        "u_max": 5.0,
        "clf_gain": 20.0,
        "clf_slack_penalty": 50.0,
        "cbf_gain": 2.5,
        "CBF_ON": True
    }

    belief_cbf_params = {
        "alpha": jnp.array([-1.0, 0.0]),
        "beta": -sim_params["wall_x"],
        "delta": 0.001  # Probability of failure threshold
    }

    return sim_params, sensor_params, control_params, belief_cbf_params

def simulate(sim_params, sensor_params, control_params, belief_cbf_params, key=None):
    
    # Recover sim params
    dt = sim_params["dt"]
    T = sim_params["T"]
    dynamics = sim_params["dynamics"]
    wall_x = sim_params["wall_x"]
    goal_x = sim_params["goal_x"]
    x_init = sim_params["x_init"]
    estimator_type = sim_params["estimator_type"]
    jax_device = sim_params["jax_device"]

    mu_u = sensor_params["mu_u"]
    sigma_u = sensor_params["sigma_u"]
    mu_v = sensor_params["mu_v"]
    sigma_v = sensor_params["sigma_v"]
    sensor_update_frequency = sensor_params["sensor_update_frequency"]

    u_max = control_params["u_max"]
    clf_gain = control_params["clf_gain"]
    clf_slack_penalty = control_params["clf_slack_penalty"]
    cbf_gain = control_params["cbf_gain"]
    CBF_ON = control_params["CBF_ON"]

    alpha = belief_cbf_params["alpha"]
    beta = belief_cbf_params["beta"]
    delta = belief_cbf_params["delta"]

    # Initial state (truth)
    x_true = jnp.array([x_init])  # Start position
    goal = jnp.array([goal_x])  # Goal position

    if sensor.__name__ == "noisy_sensor_mult":
        x_initial_measurement = sensor(x_true, 0, mu_u, sigma_u, mu_v, sigma_v, key=key) # mult_noise
    elif sensor.__name__ == "unbiased_noisy_sensor":
        x_initial_measurement = sensor(x_true, t=0, std=sigma_v, key=key) # unbiased_fixed_noise

    if estimator_type == "EKF":
        estimator = EKF(dynamics, dt, x_init=x_initial_measurement, R=jnp.square(sigma_v)*jnp.eye(dynamics.state_dim))
    elif estimator_type == "GEKF":
        estimator = GEKF(dynamics, dt, mu_u, sigma_u, mu_v, sigma_v, x_init=x_initial_measurement)

    cbf = BeliefCBF(alpha, beta, delta, n=dynamics.state_dim)

    # Autodiff: Compute Gradients for CLF
    grad_V = grad(clf, argnums=0)  # ∇V(x)

    # OSQP solver instance
    solver = OSQP()

    list_lgv = []
    list_Lg_Lf_h = []
    list_rhs = []
    list_L_f_h = []
    list_L_f_2_h = []
    list_grad_h_b = []
    list_f_b = []

    def solve_qp(b):
        x_estimated, _ = cbf.extract_mu_sigma(b)

        """Solve the CLF-CBF-QP using JAX & OSQP"""
        # Compute CLF components
        V = clf(x_estimated, goal)
        grad_V_x = grad_V(x_estimated, goal)  # ∇V(x)

        L_f_V = jnp.dot(grad_V_x.T, dynamics.f(x_estimated))
        L_g_V = jnp.dot(grad_V_x.T, dynamics.g(x_estimated))
        
        # Compute CBF components
        h = cbf.h_b(b)
        L_f_hb, _, L_f_2_h, Lg_Lf_h, grad_h_b, f_b = cbf.h_dot_b(b, dynamics) # ∇h(x)

        L_f_h = L_f_hb

        rhs, L_f_h, _ = cbf.h_b_r2_RHS(h, L_f_h, L_f_2_h, cbf_gain)

        # Define Q matrix: Minimize ||u||^2 and slack (penalty*delta^2)
        Q = jnp.array([
            [1, 0],
            [0, 2*clf_slack_penalty]
        ])
        
        # c = jnp.zeros(2)  # No linear cost term
        c = jnp.array([-1.0, 0.0])

        A = jnp.array([
            [L_g_V.flatten()[0].astype(float), -1.0], #  LgV u - delta <= -LfV - gamma(V) 
            [-Lg_Lf_h.flatten()[0].astype(float), 0.0], # -LgLfh u       <= -[alpha1 alpha2].T @ [Lfh h] + Lf^2h
            [1, 0],
            [0, 1]
        ])

        u = jnp.hstack([
            (-L_f_V - clf_gain * V).squeeze(),          # CLF constraint
            (rhs).squeeze(),                            # CBF constraint: rhs = -[alpha1 alpha2].T [Lfh h] + Lf^2h
            u_max, 
            jnp.inf # no upper limit on slack
        ])

        l = jnp.hstack([
            -jnp.inf, # No lower limit on CLF condition
            -jnp.inf, # No lower limit on CBF condition
            -u_max,
            0.0 # slack can't be negative
        ])

        # Remove CBF conditions if CBF is not ON
        if not CBF_ON:
            A = jnp.delete(A, 1, axis=0)  # Remove 2nd row
            u = jnp.delete(u, 1)          # Remove corresponding element in u
            l = jnp.delete(l, 1)          # Remove corresponding element in l

        # Solve the QP using jaxopt OSQP
        sol = solver.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
        return sol, V, h, L_g_V, Lg_Lf_h, rhs, L_f_h, L_f_2_h, grad_h_b, f_b

    x_traj = []  # Store trajectory
    x_meas = [] # Measurements
    x_est = [] # Estimates
    u_traj = []  # Store controls
    clf_values = []
    cbf_values = []
    kalman_gains = []
    covariances = []
    in_covariances = [] # Innovation Covariance of EKF
    prob_leave = [] # Probability of leaving safe set

    x_estimated, p_estimated = estimator.get_belief()

    x_measured = x_initial_measurement

    solve_qp_cpu = jit(solve_qp, backend='cpu')
    solve_qp_gpu = jit(solve_qp, backend='gpu')

    # Simulation loop
    for t in tqdm(range(T), desc="Simulation Progress"):

        x_traj.append(x_true)

        belief = cbf.get_b_vector(x_estimated, p_estimated)

        # Solve QP

        if jax_device is None:
             sol, V, h, LgV, Lg_Lf_h, rhs, L_f_h, L_f_2_h, grad_h_b, f_b = solve_qp(belief)
        elif jax_device == "cpu":
            sol, V, h, LgV, Lg_Lf_h, rhs, L_f_h, L_f_2_h, grad_h_b, f_b = solve_qp_cpu(belief)
        elif jax_device == "gpu":
            sol, V, h, LgV, Lg_Lf_h, rhs, L_f_h, L_f_2_h, grad_h_b, f_b = solve_qp_gpu(belief)

        # DEBUGGING
        list_lgv.append(LgV)
        list_Lg_Lf_h.append(Lg_Lf_h)
        list_rhs.append(rhs)
        list_L_f_h.append(L_f_h)
        list_L_f_2_h.append(L_f_2_h)
        list_grad_h_b.append(grad_h_b)
        list_f_b.append(f_b)

        clf_values.append(V)
        cbf_values.append(h)

        u_sol = jnp.array([sol.primal[0][0]])
        u_opt = jnp.clip(u_sol, -u_max, u_max)

        # Apply control to the true state (x_true)
        x_true = x_true + dt * dynamics.x_dot(x_true, u_opt)

        estimator.predict(u_opt)

        # update measurement and estimator belief
        if t > 0 and t%(sensor_update_frequency) == 0:
            # obtain current measurement
            if sensor.__name__ == "noisy_sensor_mult":
                x_measured = sensor(x_true, t, mu_u, sigma_u, mu_v, sigma_v, key=key)        
            elif sensor.__name__ == "unbiased_noisy_sensor":
                x_measured = sensor(x_true, t, sigma_v, key=key) # unbiased_fixed_noise

            if estimator.name == "GEKF":
                estimator.update(x_measured)

            if estimator.name == "EKF":
                estimator.update(x_measured)

            prob_leave.append(estimator.compute_probability_bound(alpha, delta))
        else:
            if len(prob_leave) > 0:
                prob_leave.append(prob_leave[-1])
            else:
                prob_leave.append(-1)

        x_estimated, p_estimated = estimator.get_belief()

        # Store for plotting
        u_traj.append(u_opt)
        x_meas.append(x_measured)
        x_est.append(x_estimated)
        kalman_gains.append(estimator.K)
        covariances.append(p_estimated)
        in_covariances.append(estimator.in_cov)

    # Convert to numpy arrays for analyzing
    x_traj = np.array(x_traj).squeeze()
    x_meas = np.array(x_meas).squeeze()
    x_est = np.array(x_est).squeeze()

    # time = dt*np.arange(T)  # assuming x_meas.shape[0] == N

    # # Second figure: X component comparison
    # plt.figure(figsize=(10, 10))
    # # Position x
    # plt.plot(time, x_meas[:, 0], color="green", label="Measured x", linestyle="dashed", linewidth=2, alpha=0.5)
    # plt.plot(time, x_est[:, 0], color="orange", label="Estimated x", linestyle="dotted", linewidth=1.75)
    # plt.plot(time, x_traj[:, 0], color="blue", label="True x", linewidth=2)
    # # Velocity v
    # plt.plot(time, x_meas[:, 1], color="green", label="Measured v", linestyle="dashdot", linewidth=2, alpha=0.5)
    # plt.plot(time, x_est[:, 1], color="orange", label="Estimated v", linestyle="dotted", linewidth=1.5)
    # plt.plot(time, x_traj[:, 1], color="blue", label="True v", linestyle="solid", linewidth=1)
    # # Add horizontal lines
    # plt.axhline(y=wall_x, color="red", linestyle="dashed", linewidth=1, label="Obstacle")
    # plt.axhline(y=goal_x, color="purple", linestyle="dashed", linewidth=1, label="Goal")
    # plt.xlabel("Time step (s)", fontsize=16)
    # plt.ylabel("State value", fontsize=16)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.legend(fontsize=14)
    # plt.title("State Components Over Time: Position and Velocity", fontsize=18)
    # plt.show()

    # # # Plot controls
    # plt.figure(figsize=(10, 10))
    # plt.plot(time, np.array([u[0] for u in u_traj]), color='blue', label="u_x")
    # plt.plot(time, np.array(cbf_values), color='red', label="CBF")
    # plt.plot(time, np.array(clf_values), color='green', label="CLF")
    # plt.xlabel("Time step (s)")
    # plt.ylabel("Value")
    # plt.title(f"CBF, CLF, and Control Values ({estimator.name})")
    # # Tick labels font size
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # # Legend font size
    # plt.legend(fontsize=14)
    # plt.show()

    # kalman_gain_traces = [jnp.trace(K) for K in kalman_gains]
    # covariance_traces = [jnp.trace(P) for P in covariances]

    # # # Plot trace of Kalman gains and covariances
    # plt.figure(figsize=(10, 10))
    # plt.plot(time, np.array(kalman_gain_traces), "b-", label="Trace of Kalman Gain")
    # plt.plot(time, np.array(covariance_traces), "r-", label="Trace of Covariance")
    # plt.xlabel("Time Step (s)")
    # plt.ylabel("Trace Value")
    # plt.title(f"Trace of Kalman Gain and Covariance Over Time ({estimator.name})")
    # plt.legend()
    # plt.grid()
    # plt.show()

    # Define Metrics
    metrics = {
        "num_est_exceed":      np.sum(x_est > wall_x),
        "num_true_exceed":     np.sum(x_traj > wall_x),
        "max_est":             np.max(x_est),
        "max_true":            np.max(x_traj),
        "mean_dist_est_obs":   np.mean(wall_x - x_est),
        "mean_dist_true_obs":  np.mean(wall_x - x_traj),
        "mean_dist_est_goal":  np.mean(goal_x - x_est),
        "mean_dist_true_goal": np.mean(goal_x - x_traj),
        "controller_effort":   np.linalg.norm(u_traj, ord=2),
        "tracking_rmse":       np.sqrt(np.mean((x_traj - x_est) ** 2)),
    }

    return metrics

def printMetrics(metrics):

    print("\n--- Results ---")
    print("Number of estimate exceedances:",     metrics["num_est_exceed"])
    print("Number of true exceedences:",         metrics["num_true_exceed"])
    print("Max estimate value:",                 metrics["max_est"])
    print("Max true value:",                     metrics["max_true"])
    print("Mean est distance from obstacle:",    metrics["mean_dist_est_obs"])
    print("Mean true distance from obstacle:",   metrics["mean_dist_true_obs"])
    print("Mean est distance from goal:",        metrics["mean_dist_est_goal"])
    print("Mean true distance from goal:",       metrics["mean_dist_true_goal"])
    print("Average controller effort:",          metrics["controller_effort"])
    print("Tracking RMSE:",                      metrics["tracking_rmse"])


def printSimParams(sim_params, sensor_params, control_params, belief_cbf_params):
    
    # Sim params
    dt = sim_params["dt"]
    T = sim_params["T"]
    dynamics = sim_params["dynamics"]
    wall_x = sim_params["wall_x"]
    goal_x = sim_params["goal_x"]
    x_init = sim_params["x_init"]
    estimator_type = sim_params["estimator_type"]

    # Sensor params
    mu_u = sensor_params["mu_u"]
    sigma_u = sensor_params["sigma_u"]
    mu_v = sensor_params["mu_v"]
    sigma_v = sensor_params["sigma_v"]
    sensor_update_frequency = sensor_params["sensor_update_frequency"]

    # Control params
    u_max = control_params["u_max"]
    clf_gain = control_params["clf_gain"]
    clf_slack_penalty = control_params["clf_slack_penalty"]
    cbf_gain = control_params["cbf_gain"]
    CBF_ON = control_params["CBF_ON"]

    # Belief CBF Params
    alpha = belief_cbf_params["alpha"]
    beta = belief_cbf_params["beta"]
    delta = belief_cbf_params["delta"]

    print("\n--- Simulation Parameters ---")
    print(f"Dynamics: {dynamics.name}")
    print(f"Estimator type: {estimator_type}")
    print(f"Time Step (dt): {dt}")
    print(f"Number of Steps (T): {T}")
    print(f"Default jax backend device: {jax.default_backend()}")
    print(f"Requested jax device:  {sim_params['jax_device']}")

    print("\n-- Sensor Params ---")
    print(f"Sensor Type: {sensor.__name__}")
    print(f"mu_u       : {mu_u}")
    print(f"sigma_u    : {sigma_u}")
    print(f"mu_v       : {mu_v}")
    print(f"sigma_v    : {sigma_v}")
    print(f"Sensor Update Frequency (number of iterations): {sensor_update_frequency}")
    
    print("\n--- Environment Setup ---")
    print(f"Obstacle Position (wall_x): {wall_x}")
    print(f"Goal Position (goal_x): {goal_x}")
    print(f"Initial Position (x_init): {x_init}")
 
    print("\n--- Control Parameters ---")
    print(f"u_max: {u_max}")
    print(f"CLF Linear Gain (clf_gain): {clf_gain}")
    print(f"CLF Slack (clf_slack): {clf_slack_penalty}")
    print(f"CBF Linear Gain (cbf_gain): {cbf_gain}")
    if CBF_ON:
        print("CBF: ON")
    else:
        print("CBF: OFF")

    print("\n--- Belief CBF Parameters ---")
    print(f"alpha: {alpha}")
    print(f"beta: {beta}")
    print(f"Failure Probability Threshold (delta): {delta}")

start_idx = 1
end_idx = 100

save_freq = 10
base_path = "/home/speedracer1702/Projects/automata_lab/cbf_lite/Results/Summer 2025/1D Double Integrator Tuning/Fix post Feedback 1/Batch"
os.makedirs(base_path, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = os.path.join(base_path, f"run_EKF_nominal_{timestamp}.json")

if __name__ == "__main__":
    
    sim_params, sensor_params, control_params, belief_cbf_params = getSimParams()
    printSimParams(sim_params, sensor_params, control_params, belief_cbf_params)

    indices = list(range(start_idx, end_idx + 1))

    metrics_sum = {}

    for i in tqdm(indices, desc="Running simulations"):

        key = jax.random.PRNGKey(i)

        metrics = simulate(sim_params, sensor_params, control_params, belief_cbf_params, key=key)
        # printMetrics(metrics) # For a single simuation

        # Accumulate sums for running average
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





