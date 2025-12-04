

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD DATA
# ============================================================

data = np.load("sim_sinusoidal_EKF.npz", allow_pickle=True)

# --- Core state & control arrays ---
x_traj      = data["x_traj"]
x_meas      = data["x_meas"]
x_est       = data["x_est"]
u_traj      = data["u_traj"]
u_nom       = data["u_nom"]
cbf_values  = data["cbf_values"]
x_nom       = data["x_nom"]
time        = data["time"]

left_lglfh         = data["left_lglfh"]
right_lglfh        = data["right_lglfh"]
left_rhs           = data["left_rhs"]
right_rhs          = data["right_rhs"]

right_l_f_h  = data["right_l_f_h_full"]
right_l_f_2_h = data["right_l_f_2_h_full"]
left_l_f_h    = data["left_l_f_h_full"]
left_l_f_2_h  = data["left_l_f_2_h_full"]

h2_vals = cbf_values[:, 1]

# ============================================================
#                     FIGURE 7
#       Compare ω = u_opt[:,1]  vs  -RHS / LgLfh[:,1]
# ============================================================

T = len(right_lglfh)
time = np.arange(T)

u_opt_list = np.array(u_traj)

# Extract the second control input (ω)
u2 = u_opt_list[:, 1]

# Extract the second coefficient of Lg_Lf_h for the right CBF
LgLfh_right_ang = right_lglfh[:, 1]

# Compute the HOCBF-implied upper bound on u2
hocbf_bound = -right_rhs.squeeze() / LgLfh_right_ang   # elementwise division

# Create 3-subplot figure
fig7, axes7 = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

# =====================================
# Subplot 1 — ω vs HOCBF Bound
# =====================================
ax7 = axes7[0]

MAX_ANGULAR = 10.0
u2_clipped = np.clip(u2, -MAX_ANGULAR, MAX_ANGULAR)

ax7.plot(time, u2, label="u_opt ω (heading)", color="red")
ax7.plot(time, hocbf_bound, label="-RHS / LgLf_h[1] (bound)", linestyle="--", color="gray", alpha=0.4)

ax7.set_title("Figure 7: Heading Control vs HOCBF-implied Bound")
ax7.set_ylabel("ω (rad/s)")
ax7.grid(True)
ax7.legend()


# =====================================
# Subplot 2 — CBF higher-order terms
# =====================================
ax_terms = axes7[1]

alpha = 50.0
roots = np.array([-0.75])  # select stable root
coeff = alpha * np.poly(roots)

h_ddot  = right_l_f_2_h.squeeze()

term_c1 = -coeff[0] * right_l_f_h
term_c2 = -coeff[1] * h2_vals

ax_terms.plot(time, term_c1, label="-α c₁ ḣ", color="blue")
ax_terms.plot(time, term_c2, label="-α c₂ h",  color="green")
# ax_terms.plot(time, -right_l_f_2_h, label="-L_f² h", color="black")

ax_terms.set_title("Right CBF Higher-Order Terms")
ax_terms.set_ylabel("Value")
ax_terms.grid(True)
ax_terms.legend()


# =====================================
# Subplot 3 — New: LgLf_h Right Angular Component
# =====================================
ax_ang = axes7[2]

ax_ang.plot(time, LgLfh_right_ang, label="LgLfₕ right angular term", color="purple")

ax_ang.set_title("LgLfₕ Angular Component (Right CBF)")
ax_ang.set_xlabel("Time Step")
ax_ang.set_ylabel("Value")
ax_ang.grid(True)
ax_ang.legend()


# =====================================
# Subplot 4 — New: LgLf_h Right Angular Component
# =====================================
ax_lf2h = axes7[3]

ax_lf2h.plot(time, right_l_f_2_h, label="Lf2h term")

ax_lf2h.set_title("Lf2h")
ax_lf2h.set_xlabel("Time Step")
ax_lf2h.set_ylabel("Value")
ax_lf2h.grid(True)
ax_lf2h.legend()

plt.show()