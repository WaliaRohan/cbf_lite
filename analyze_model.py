import pandas as pd
import matplotlib.pyplot as plt
import mplcursors

# Load data
df = pd.read_csv("input_vs_pred_analyzed.csv")
x  = df["input_pos"].to_numpy()
y1 = df["Err (Pred - True)"].to_numpy()
y2 = df["Max Bound Err (Pred - True)"].to_numpy() + 0.437  # offset applied

# Plot
fig, ax = plt.subplots(figsize=(8,5))
s1 = ax.scatter(x, y1, label="Error (Pred - True)", marker="o", alpha=0.8, s=10)
(line2,) = ax.plot(x, y2, label="Max Bound Error (Pred - True)",
                   linestyle="--", color="tab:orange", alpha=0.8)

ax.set_xlabel("Training Ground Truth")
ax.set_ylabel("Error (Prediction - Ground Truth)")
ax.legend()
ax.grid(True)
plt.tight_layout()

# # Data cursor
# cursor = mplcursors.cursor([s1, line2], hover=True)

# @cursor.connect("add")
# def on_add(sel):
#     i = sel.index
#     if sel.artist is s1:
#         sel.annotation.set_text(f"input_pos={x[i]:.3f}\nErr={y1[i]:.3f}")
#     else:  # line2
#         sel.annotation.set_text(f"input_pos={x[i]:.3f}\nMaxBoundErr={y2[i]:.3f}")
#     sel.annotation.get_bbox_patch().set(alpha=0.9)


# New plot: Abs Err vs input_pos
y3 = df["Abs Err"].to_numpy()

fig2, ax2 = plt.subplots(figsize=(8,5))
s3 = ax2.scatter(x, y3, marker="o", alpha=0.8, s=10)

ax2.set_xlabel("Training Ground Truth")
ax2.set_ylabel("Absolute Error (Prediction - GT)")
ax2.legend()
ax2.grid(True)
plt.tight_layout()

# # Data cursor
# cursor2 = mplcursors.cursor(s3, hover=True)

# @cursor2.connect("add")
# def on_add_abs(sel):
#     i = sel.index
#     sel.annotation.set_text(f"input_pos={x[i]:.3f}\nAbsErr={y3[i]:.3f}")
#     sel.annotation.get_bbox_patch().set(alpha=0.9)

plt.show()
