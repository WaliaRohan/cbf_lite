import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from cbfs import vanilla_cbf_circle as cbf
from cbfs import vanilla_clf as clf
from dynamics import SimpleDynamics
from sensor import noisy_sensor as sensor

# Define simulation parameters
dt = 0.1  # Time step
T = 30  # Number of steps
x_traj = []  # Store trajectory
u_traj = []  # Store controls

# Initial state (truth)
x_true = np.array([-1.5, -1.5])  # Start position
goal = np.array([2.0, 2.0])  # Goal position

obstacle = np.array([1.0, 1.0])  # Obstacle position
safe_radius = 0.5  # Safety radius around the obstacle

dynamics = SimpleDynamics()

# Simulation loop
for _ in range(T):
    # Get reading from sensor
    x_measured = sensor(x_true)

    x_estimated = x_measured

    # Optimization variables
    u = cp.Variable(2)
    delta = cp.Variable()

    # Compute CLF components based on belief (x_estimated)
    V = clf(x_estimated, goal)
    L_f_V = 2 * (x_estimated - goal) @ dynamics.f(x_estimated)
    L_g_V = 2 * (x_estimated - goal) @ dynamics.g(x_estimated)
    gamma = 1.0  # CLF gain

    # Compute CBF components based on belief (x_estimated)
    h = cbf(x_estimated, obstacle, safe_radius)
    L_f_h = 2 * (x_estimated - obstacle) @ dynamics.f(x_estimated)
    L_g_h = 2 * (x_estimated - obstacle) @ dynamics.g(x_estimated)
    alpha = 1.0  # CBF gain

    # Constraints
    constraints = [
        L_f_V + L_g_V @ u + gamma * V <= delta,  # CLF constraint (relaxed)
        L_f_h + L_g_h @ u + alpha * h >= 0,  # CBF constraint (strict)
        cp.norm(u, "inf") <= 1.0  # Control limit
    ]

    # Objective function (minimize control + CLF relaxation)
    lambda_delta = 10.0
    objective = cp.Minimize(cp.norm(u) ** 2 + lambda_delta * delta ** 2)

    # Solve QP
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Apply control to the true state (x_true)
    u_opt = np.array(u.value).flatten()
    x_true = x_true + (dynamics.f(x_true) + dynamics.g(x_true) @ u_opt)

    # Store for plotting
    x_traj.append(x_true.copy())
    u_traj.append(u_opt)

# Convert to numpy arrays
x_traj = np.array(x_traj)

# Plot trajectory
plt.figure(figsize=(6, 6))
plt.plot(x_traj[:, 0], x_traj[:, 1], "b-", label="Trajectory (True state)")
plt.scatter(goal[0], goal[1], c="g", marker="*", s=200, label="Goal")
plt.scatter(obstacle[0], obstacle[1], c="r", marker="o", s=200, label="Obstacle")
circle = plt.Circle(obstacle, safe_radius, color="r", fill=False, linestyle="--")
plt.gca().add_patch(circle)

plt.xlim(-2, 3)
plt.ylim(-2, 3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("CLF-CBF QP-Controlled Trajectory (Belief-based)")
plt.legend()
plt.grid()
plt.show()
