import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

# --- Load both EKF and GEKF datasets ---
skip = 25  # adjust to skip frames for speed

data_ekf = np.load("sim_data_EKF.npz")
data_gekf = np.load("sim_data_GEKF.npz")

def downsample(d):
    return {
        "x_traj": d["x_traj"][::skip],
        "x_est": d["x_est"][::skip],
        "x_meas": d["x_meas"][::skip],
        "cbf_values": d["cbf_values"][::skip],
        "x_nom": d["x_nom"][::skip],
        "time": d["time"][::skip]
    }

ekf = downsample(data_ekf)
gekf = downsample(data_gekf)

wall_y = 5.0
n_frames = min(len(ekf["time"]), len(gekf["time"]))

# --- Create figure ---
fig, (ax_traj, ax_cbf) = plt.subplots(1, 2, figsize=(12, 6))
plt.tight_layout(pad=3)

# ---- subplot 1: trajectories ----
# EKF
traj_nom_ekf,  = ax_traj.plot([], [], 'black', label="Reference")
traj_true_ekf, = ax_traj.plot([], [], 'blue', label="True (EKF)")
# traj_est_ekf,  = ax_traj.plot([], [], color='blue', label="Est (EKF)")
meas_scatter_ekf = ax_traj.scatter([], [], s=5, color='skyblue', alpha=0.4, label="Meas (EKF)")

# GEKF
# traj_nom_gekf,  = ax_traj.plot([], [], 'orange', alpha=0.6, label="Nominal (GEKF)")
traj_true_gekf, = ax_traj.plot([], [], 'orange', linestyle='-', alpha=1.0, label="True (GEKF)")
# traj_est_gekf,  = ax_traj.plot([], [], color='orange', linestyle='-', alpha=0.6, label="Est (GEKF)")
meas_scatter_gekf = ax_traj.scatter([], [], s=5, color='green', alpha=0.4, label="Meas (GEKF)")

ax_traj.axhline(y=wall_y, color='red', linestyle='--')
ax_traj.axhline(y=-wall_y, color='red', linestyle='--')
ax_traj.set_xlim(min(np.min(ekf["x_traj"][:,0]), np.min(gekf["x_traj"][:,0])) - 1,
                 max(np.max(ekf["x_traj"][:,0]), np.max(gekf["x_traj"][:,0])) + 1)
ax_traj.set_ylim(-1.2*wall_y, 1.2*wall_y)
ax_traj.set_title("2D Trajectory Comparison (EKF vs GEKF)")
ax_traj.set_xlabel("x [m]")
ax_traj.set_ylabel("y [m]")
leg = ax_traj.legend(
    fontsize=7,
    loc='center',            # anchor corner of legend box
    bbox_to_anchor=(0.25, 0.5)  # (x, y) inside axes fraction: (0,0)=bottom-left, (1,1)=top-right
)
for handle in leg.legend_handles:
    handle.set_alpha(1.0)
ax_traj.grid(True)

# ---- subplot 2: CBF values ----
h_line_ekf,  = ax_cbf.plot([], [], color='cyan', label="BCBF-EKF Upper")
h2_line_ekf, = ax_cbf.plot([], [], color='grey', label="BCBF-EKF Lower")
h_line_gekf,  = ax_cbf.plot([], [], color='black', linestyle='--', label="BCBF-GEKF Upper")
h2_line_gekf, = ax_cbf.plot([], [], color='magenta', linestyle='--', label="BCBF-GEKF Lower")

ax_cbf.axhline(0, color='black', linestyle=':')
ax_cbf.set_xlim(0, min(ekf["time"][-1], gekf["time"][-1]))
ax_cbf.set_ylim(
    min(np.min(ekf["cbf_values"]), np.min(gekf["cbf_values"])) - 0.1,
    max(np.max(ekf["cbf_values"]), np.max(gekf["cbf_values"])) + 0.1
)
ax_cbf.set_xlabel("Time [s]")
ax_cbf.set_ylabel("Barrier value [m]")
ax_cbf.set_title("CBF Evolution (EKF vs GEKF)")
ax_cbf.legend(
    fontsize=7,
    loc='center',            # anchor corner of legend box
    bbox_to_anchor=(0.25, 0.5)  # (x, y) inside axes fraction: (0,0)=bottom-left, (1,1)=top-right
)
ax_cbf.grid(True)

# --- Update function ---
def update(i):
    # EKF
    traj_true_ekf.set_data(ekf["x_traj"][:i,0], ekf["x_traj"][:i,1])
    # traj_est_ekf.set_data(ekf["x_est"][:i,0], ekf["x_est"][:i,1])
    traj_nom_ekf.set_data(ekf["x_nom"][:i,0], ekf["x_nom"][:i,1])
    meas_scatter_ekf.set_offsets(ekf["x_meas"][:i,:2])
    h_line_ekf.set_data(ekf["time"][:i], ekf["cbf_values"][:i,0])
    h2_line_ekf.set_data(ekf["time"][:i], ekf["cbf_values"][:i,1])

    # GEKF
    traj_true_gekf.set_data(gekf["x_traj"][:i,0], gekf["x_traj"][:i,1])
    # traj_est_gekf.set_data(gekf["x_est"][:i,0], gekf["x_est"][:i,1])
    # traj_nom_gekf.set_data(gekf["x_nom"][:i,0], gekf["x_nom"][:i,1])
    meas_scatter_gekf.set_offsets(gekf["x_meas"][:i,:2])
    h_line_gekf.set_data(gekf["time"][:i], gekf["cbf_values"][:i,0])
    h2_line_gekf.set_data(gekf["time"][:i], gekf["cbf_values"][:i,1])

    # ax_traj.set_title(f"Frame {i}/{n_frames}")
    return (
            traj_true_ekf,
            # traj_est_ekf,
            traj_nom_ekf,
            meas_scatter_ekf,
            traj_true_gekf,
            # traj_est_gekf,
            # traj_nom_gekf,
            meas_scatter_gekf,
            h_line_ekf,
            h2_line_ekf,
            h_line_gekf,
            h2_line_gekf)

# --- Animate ---
ani = FuncAnimation(fig, update, frames=n_frames, interval=1, blit=True)

# --- Add progress bar ---
pbar = tqdm(total=n_frames, desc="Saving animation", dynamic_ncols=True)

def progress_callback(frame, total):
    pbar.update(1)
    if frame == total - 1:
        pbar.close()

# --- Fast ffmpeg writer ---
writer = FFMpegWriter(
    fps=30,
    codec="libx264",
    extra_args=[
        "-preset", "ultrafast",
        "-crf", "25",
        "-threads", str(os.cpu_count()),
        "-tune", "animation",
        "-pix_fmt", "yuv420p"
    ]
)

ani.save(
    "trajectory_cbf_comparison.mp4",
    writer=writer,
    dpi=300,
    progress_callback=progress_callback
)
