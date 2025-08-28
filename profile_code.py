import time, jax
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jaxopt import BoxOSQP as OSQP
from tqdm import tqdm

from cbfs import BeliefCBF
from cbfs import vanilla_clf_dubins as clf
from dynamics import *
from estimators import *
from sensor import noisy_sensor_mult as sensor

PROFILING = True  # flip to False to disable timing without editing the loop

def _block(o):
    # Ensure JAX work finishes before stopping timers
    try:
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else None, o
        )
    except Exception:
        pass

@contextmanager
def timer(name, acc):
    if not PROFILING:
        yield
        return
    t0 = time.perf_counter()
    yield
    acc[name] = acc.get(name, 0.0) + (time.perf_counter() - t0)

def print_profile_report(acc, iters):
    if not acc:
        print("\n(No profiling data collected)")
        return
    items = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(acc.values())
    print("\n=== PROFILE ({} iters) ===".format(iters))
    for k, v in items:
        print(f"{k:>18}: {v:8.4f}s  ({100*v/total:5.1f}%)  avg {1e3*v/iters:7.2f} ms/iter")
    print(f"{'TOTAL (timed)':>18}: {total:8.4f}s\n")


# Sim Params
dt = 0.001
T = 200
dynamics = DubinsDynamics()
report_results = False

# Sensor Params
mu_u = 0.1
sigma_u = jnp.sqrt(0.01) # Standard deviation
mu_v = 0.001
sigma_v = jnp.sqrt(0.0005) # Standard deviation
sensor_update_frequency = 0.1 # Hz

# Obstacle
wall_y = 2.5
x_init = [0.0, 0.0, 1.0, 0.0] # x, y, v, theta

# Initial state (truth)
x_true = jnp.array(x_init)  # Start position
goal = 3.0*jnp.array([1.0, 1.0])  # Goal position
obstacle = jnp.array([wall_y])  # Wall

# Mean and covariance
x_initial_measurement = sensor(x_true, 0, mu_u, sigma_u, mu_v, sigma_v) # mult_noise
h = lambda x: jnp.array([x[1]])
estimator = GEKF(dynamics, dt, mu_u, sigma_u, mu_v, sigma_v, h=h, x_init=x_initial_measurement)

# Define belief CBF parameters
n = dynamics.state_dim
alpha = jnp.array([0.0, -1.0, 0.0, 0.0])
beta = jnp.array([-wall_y])
delta = 0.001  # Probability of failure threshold
cbf = BeliefCBF(alpha, beta, delta, n)

# Control params
u_max = 5.0
clf_gain = 20.0 # CLF linear gain
clf_slack_penalty = 100.0
cbf_gain = 10.0  # CBF linear gain
CBF_ON = True

# Autodiff: Compute Gradients for CLF
grad_V = grad(clf, argnums=0)  # ∇V(x)

# OSQP solver instance
solver = OSQP()

def solve_qp(b):
    x_estimated, sigma = cbf.extract_mu_sigma(b)

    """Solve the CLF-CBF-QP using JAX & OSQP"""
    # Compute CLF components
    V = clf(x_estimated, goal)
    grad_V_x = grad_V(x_estimated, goal)  # ∇V(x)

    L_f_V = jnp.dot(grad_V_x.T, dynamics.f(x_estimated))
    L_g_V = jnp.dot(grad_V_x.T, dynamics.g(x_estimated))
    
    # # Compute CBF components
    h = cbf.h_b(b)
    L_f_hb, L_g_hb, L_f_2_h, Lg_Lf_h, grad_h_b, f_b = cbf.h_dot_b(b, dynamics) # ∇h(x)

    L_f_h = L_f_hb
    L_g_h = L_g_hb

    rhs, L_f_h, h_gain = cbf.h_b_r2_RHS(h, L_f_h, L_f_2_h, cbf_gain)

    # Define Q matrix: Minimize ||u||^2 and slack (penalty*delta^2)
    Q = jnp.array([
        [1, 0],
        [0, 2*clf_slack_penalty]
    ])
    
    c = jnp.zeros(2)  # No linear cost term

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

    if not CBF_ON:
        A = jnp.delete(A, 1, axis=0)  # Remove 2nd row
        u = jnp.delete(u, 1)          # Remove corresponding element in u
        l = jnp.delete(l, 1)          # Remove corresponding element in l

    # Solve the QP using jaxopt OSQP
    sol = solver.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params

    return sol, V

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
x_meas.append(x_measured)

solve_qp_cpu = jit(solve_qp, backend='cpu')
solve_qp_gpu = jit(solve_qp, backend='gpu')


# Warm-up JITs once (compilation happens here)
belief_warmup = cbf.get_b_vector(x_estimated, p_estimated)
_ = solve_qp_cpu(belief_warmup)
_block(_)

acc = {"solve_qp_cpu":0.0, "dynamics_step":0.0, "estimator_predict":0.0,
       "estimator_update":0.0, "sensor":0.0, "logging":0.0, "belief":0.0, "other":0.0}

for t in range(T):
    t_iter0 = time.perf_counter()
    t_sub = 0.0

    # Belief build
    t0 = time.perf_counter()
    x_traj.append(x_true)
    belief = cbf.get_b_vector(x_estimated, p_estimated)
    dt = time.perf_counter() - t0
    acc["belief"] += dt; t_sub += dt

    # QP
    t0 = time.perf_counter()
    sol, V = solve_qp_cpu(belief)
    _block((sol, V))
    dt = time.perf_counter() - t0
    acc["solve_qp_cpu"] += dt; t_sub += dt

    clf_values.append(V)
    u_sol = jnp.array([sol.primal[0][0]])
    u_opt = jnp.clip(u_sol, -u_max, u_max)

    # Dynamics
    t0 = time.perf_counter()
    x_true = x_true + dt * dynamics.x_dot(x_true, u_opt)
    _block(x_true)
    dt2 = time.perf_counter() - t0
    acc["dynamics_step"] += dt2; t_sub += dt2

    # Estimator predict
    t0 = time.perf_counter()
    estimator.predict(u_opt)
    dt = time.perf_counter() - t0
    acc["estimator_predict"] += dt; t_sub += dt

    # Sensor + update (conditional)
    if t > 0 and t % (1 / sensor_update_frequency) == 0:
        t0 = time.perf_counter()
        x_measured = sensor(x_true, t, mu_u, sigma_u, mu_v, sigma_v)
        dt = time.perf_counter() - t0
        acc["sensor"] += dt; t_sub += dt

        t0 = time.perf_counter()
        if estimator.name == "GEKF" or estimator.name == "EKF":
            estimator.update(x_measured)
        dt = time.perf_counter() - t0
        acc["estimator_update"] += dt; t_sub += dt

        x_meas.append(x_measured)

    x_estimated, p_estimated = estimator.get_belief()

    # Logging
    t0 = time.perf_counter()
    u_traj.append(u_opt)
    x_est.append(x_estimated)
    kalman_gains.append(estimator.K)
    covariances.append(p_estimated)
    in_covariances.append(estimator.in_cov)
    dt = time.perf_counter() - t0
    acc["logging"] += dt; t_sub += dt

    # Other (unaccounted)
    acc["other"] += (time.perf_counter() - t_iter0) - t_sub

print_profile_report(acc, T)