import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, jit, random
from jaxopt import BoxOSQP as OSQP
from tqdm import tqdm

from cbfs import BeliefCBF
from cbfs import vanilla_clf_x as clf
from dynamics import *
from estimators import *
from sensor import noisy_sensor_mult as sensor


def simulate_run(run, estimator_name):

    # Define simulation parameters
    dt = 0.001 # Time step
    T = 5000 # Number of steps
    u_max = 10.0

    # Obstacle
    wall_x = 5.0
    goal_x = 6.0
    x_init = 1.0

    # Initial state (truth)
    x_true = jnp.array([x_init])  # Start position
    goal = jnp.array([goal_x])  # Goal position
    obstacle = jnp.array([wall_x])  # Wall

    dynamics = NonLinearSingleIntegrator1D() 

    # High noise
    mu_u = 0.1
    sigma_u = jnp.sqrt(0.001)

    mu_v = 0.01
    sigma_v = jnp.sqrt(0.0005)

    # Low noise
    # mu_u = 0.0174
    # sigma_u = jnp.sqrt(2.916e-4) # 10 times more than what was shown in GEKF paper

    # # Additive noise
    # mu_v = -0.0386
    # sigma_v = jnp.sqrt(7.97e-5)

    x_initial_measurement = sensor(x_true, 0, mu_u, sigma_u, mu_v, sigma_v, random.PRNGKey(run))

    if estimator_name == "GEKF":
        estimator = GEKF(dynamics, dt, mu_u, sigma_u, mu_v, sigma_v, x_init=x_initial_measurement)
    elif estimator_name == "EKF":
        estimator = EKF(dynamics, dt, x_init=x_initial_measurement, R=sigma_v*jnp.eye(dynamics.state_dim))

    print("Using ", estimator_name)

    # Define belief CBF parameters
    n = dynamics.state_dim
    alpha = jnp.array([-1.0])  # Example matrix
    beta = jnp.array([-wall_x])  # Example vector
    delta = 0.001  # Probability of failure threshold
    cbf = BeliefCBF(alpha, beta, delta, n)

    # Control params
    clf_gain = 20.0  # CLF linear gain
    clf_slack_penalty = 0.00025
    cbf_gain = 200.0 # CBF linear gain

    # Autodiff: Compute Gradients for CLF and CBF
    grad_V = grad(clf, argnums=0)  # ∇V(x)

    # OSQP solver instance
    solver = OSQP()

    print(jax.default_backend())

    # @jit
    def solve_qp(b):
        x_estimated, sigma = cbf.extract_mu_sigma(b)

        """Solve the CLF-CBF-QP using JAX & OSQP"""
        # Compute CLF components
        V = clf(x_estimated, goal)
        grad_V_x = grad_V(x_estimated, goal)  # ∇V(x)

        L_f_V = jnp.dot(grad_V_x.T, dynamics.f(x_estimated))
        L_g_V = jnp.dot(grad_V_x.T, dynamics.g(x_estimated))
        
        # Compute CBF components
        h = cbf.h_b(b)
        L_f_hb, L_g_hb = cbf.h_dot_b(b, dynamics) # ∇h(x)

        L_f_h = L_f_hb
        L_g_h = L_g_hb

        # Define QP matrices
        Q = jnp.array([
            [1, 0],
            [0, 2*clf_slack_penalty]
        ])
        
        # Minimize ||u||^2 and slack
        c = jnp.zeros(2)  # No linear cost term

        A = jnp.array([
            [L_g_V.flatten()[0].astype(float), -1.0],       # -Lgh u <= Lfh + alpha(h)
            [-L_g_h.flatten()[0].astype(float), 0.0],       # LgV u - delta <= -LfV - gamma
            [1, 0],
            [0, 1]
        ])

        u = jnp.hstack([
            (-L_f_V - clf_gain * V).squeeze(),   # CLF constraint
            (L_f_h.squeeze() + cbf_gain * h).squeeze(),     # CBF constraint
            u_max, 
            jnp.inf
        ])

        l = jnp.hstack([
            -jnp.inf,
            -jnp.inf,
            -u_max,
            0.0
        ])

        # Solve the QP using jaxopt OSQP
        sol = solver.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
        return sol, clf_gain*V, cbf_gain*h

    x_traj = []  # Store trajectory
    x_meas = [] # Measurements
    x_est = [] # Estimates
    u_traj = []  # Store controls
    clf_values = []
    cbf_values = []
    kalman_gains = []
    covariances = []

    x_estimated, p_estimated = estimator.get_belief()

    # @jit
    def get_b_vector(mu, sigma):

        # Extract the upper triangular elements of a matrix as a 1D array
        upper_triangular_indices = jnp.triu_indices(sigma.shape[0])
        vec_sigma = sigma[upper_triangular_indices]

        b = jnp.concatenate([mu, vec_sigma])

        return b

    x_measured = x_initial_measurement

    solve_qp_cpu = jit(solve_qp, backend='cpu')


    # Simulation loop
    for t in tqdm(range(T), desc="Simulation Progress"):

        x_traj.append(x_true)

        belief = get_b_vector(x_estimated, p_estimated)

        # Solve QP
        sol, V, h = solve_qp_cpu(belief)

        clf_values.append(V)
        cbf_values.append(h)

        u_opt = jnp.array([sol.primal[0][0]])

        # Apply control to the true state (x_true)
        x_true = x_true + dt * (dynamics.f(x_true) + dynamics.g(x_true) @ u_opt)

        estimator.predict(u_opt)

        # update measurement and estimator belief
        if t > 0 and t%10 == 0:
            # obtain current measurement
            x_measured =  sensor(x_true, t, mu_u, sigma_u, mu_v, sigma_v, random.PRNGKey(run))

            if estimator.name == "GEKF":
                estimator.update_1D(x_measured)

            if estimator.name == "EKF":
                estimator.update(x_measured)

        x_estimated, p_estimated = estimator.get_belief()

        # Store for plotting
        u_traj.append(u_opt)
        x_meas.append(x_measured)
        x_est.append(x_estimated)
        kalman_gains.append(estimator.K)
        covariances.append(p_estimated)

    # Convert to JAX arrays
    x_traj = jnp.array(x_traj)

    # Conver to numpy arrays for plotting
    x_traj = np.array(x_traj)
    x_meas = np.array(x_meas)
    x_est = np.array(x_est)

    time = dt*np.arange(T)  # assuming x_meas.shape[0] == N

    kalman_gain_traces = np.array([jnp.trace(K) for K in kalman_gains])
    covariance_traces = np.array([jnp.trace(P) for P in covariances])

    dist = wall_x - x_est


    metrics = {
    "num_est_exceedances": np.sum(x_est > wall_x),
    "num_true_exceedances": np.sum(x_traj > wall_x),
    "max_est_value": np.max(x_est),
    "max_true_value": np.max(x_traj),
    "mean_est_dist_from_obstacle": np.mean(wall_x - x_est),
    "avg_controller_effort": np.linalg.norm(u_traj, ord=2),
    "cumulative_dist_to_goal": np.sum(np.abs(x_traj - wall_x)),
    "tracking_rmse": np.sqrt(np.mean((x_traj - x_est) ** 2))
    }

    # Print Metrics

    # print("\n--- Results ---")

    # print("Number of estimate exceedances: ", np.sum(x_est > wall_x))
    # print("Number of true exceedences", np.sum(x_traj > wall_x))
    # print("Max estimate value: ", np.max(x_est))
    # print("Max true value: ", np.max(x_traj))
    # print("Mean true distance from obstacle: ", np.mean(wall_x - x_est))
    # print("Average controller effort: ", np.linalg.norm(u_traj, ord=2))
    # print("Cummulative distance to goal: ", np.sum(np.abs(x_traj - wall_x)))
    # print(f"{estimator.name} Tracking RMSE: ", np.sqrt(np.mean((x_traj - x_est) ** 2)))

    return metrics

def print_metrics(avg_metrics, estimator_name, runs):
    print("\nAverage Metrics over", runs, "simulations:")
    print("Number of estimate exceedances:", avg_metrics["num_est_exceedances"])
    print("Number of true exceedances:", avg_metrics["num_true_exceedances"])
    print("Max estimate value:", avg_metrics["max_est_value"])
    print("Max true value:", avg_metrics["max_true_value"])
    print("Mean est distance from obstacle:", avg_metrics["mean_est_dist_from_obstacle"])
    print("Average controller effort:", avg_metrics["avg_controller_effort"])
    print("Cumulative distance to goal:", avg_metrics["cumulative_dist_to_goal"])
    print(f"{estimator_name} Tracking RMSE:", avg_metrics["tracking_rmse"])

all_metrics = []

runs = 25

estimator_name = "EKF"

for run in tqdm(range(runs), desc="Running Simulations"):
    all_metrics.append(simulate_run(run, estimator_name))
    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
    print_metrics(avg_metrics, estimator_name, run)

avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}


