import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

data = np.load("sim_data.npz")
skip = 10  # change to 2, 5, etc. for different skip rates
x_traj = data["x_traj"][::skip]
x_est = data["x_est"][::skip]
x_meas = data["x_meas"][::skip]
cbf_values = data["cbf_values"][::skip]
x_nom = data["x_nom"][::skip]
time = data["time"][::skip]

wall_y = 5.0
n_frames = len(time)

# --- Create figure
fig, (ax_traj, ax_cbf) = plt.subplots(1, 2, figsize=(12, 6))
plt.tight_layout(pad=3)

# ---- subplot 1: trajectories ----
traj_true, = ax_traj.plot([], [], 'b-', label="True traj")
traj_est,  = ax_traj.plot([], [], color='orange', label="Estimate")
traj_nom,  = ax_traj.plot([], [], 'k--', label="Nominal")
meas_scatter = ax_traj.scatter([], [], s=5, color='green', alpha=0.4, label="Meas")
ax_traj.axhline(y=wall_y, color='red', linestyle='--')
ax_traj.axhline(y=-wall_y, color='red', linestyle='--')
ax_traj.set_xlim(np.min(x_traj[:,0])-1, np.max(x_traj[:,0])+1)
ax_traj.set_ylim(-1.2*wall_y, 1.2*wall_y)
ax_traj.set_title("2D Trajectory Evolution")
ax_traj.set_xlabel("x [m]")
ax_traj.set_ylabel("y [m]")
ax_traj.legend(); ax_traj.grid(True)

# ---- subplot 4: CBF values ----
h_line,  = ax_cbf.plot([], [], color='red', label=f"CBF: y<{wall_y}")
h2_line, = ax_cbf.plot([], [], color='purple', label=f"CBF: y>-{wall_y}")
ax_cbf.axhline(0, color='black', linestyle=':')
ax_cbf.set_xlim(0, time[-1])
ax_cbf.set_ylim(np.min(cbf_values)-0.1, np.max(cbf_values)+0.1)
ax_cbf.set_xlabel("Time [s]")
ax_cbf.set_ylabel("Barrier value [m]")
ax_cbf.set_title("CBF evolution over time")
ax_cbf.legend(); ax_cbf.grid(True)

# --- Update function
def update(i):
    traj_true.set_data(x_traj[:i,0], x_traj[:i,1])
    traj_est.set_data(x_est[:i,0], x_est[:i,1])
    traj_nom.set_data(x_nom[:i,0], x_nom[:i,1])
    meas_scatter.set_offsets(x_meas[:i,:2])
    h_line.set_data(time[:i], cbf_values[:i,0])
    h2_line.set_data(time[:i], cbf_values[:i,1])
    ax_traj.set_title(f"Frame {i}/{n_frames}")
    return traj_true, traj_est, traj_nom, meas_scatter, h_line, h2_line

# --- Animate
ani = FuncAnimation(fig, update,
                    frames=n_frames,
                    interval=1, blit=True)

# --- Add progress bar
pbar = tqdm(total=n_frames, desc="Saving animation", dynamic_ncols=True)

def progress_callback(frame, total):
    pbar.update(1)
    if frame == total - 1:
        pbar.close()

# --- Save fast with ffmpeg
writer = FFMpegWriter(
    fps=30,  # higher frame rate = smoother motion, same total frames
    codec="libx264",
    extra_args=[
        "-preset", "ultrafast",       # fastest possible preset
        "-crf", "25",                 # slightly higher CRF = smaller, faster
        "-threads", str(os.cpu_count()),
        "-tune", "animation",         # optimize encoder for flat colors/plots
        "-pix_fmt", "yuv420p"
    ]
)

ani.save(
    "trajectory_cbf_animation.mp4",
    writer=writer,
    dpi=300,
    progress_callback=progress_callback
)
