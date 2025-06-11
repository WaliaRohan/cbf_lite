import jax.numpy as jnp

# Create a 2x1x2 array with distinguishable values
A_g = jnp.array([
    [[1.0, 2.0]],   # A_g[0, 0, 0] = 1.0, A_g[0, 0, 1] = 2.0
    [[3.0, 4.0]]    # A_g[1, 0, 0] = 3.0, A_g[1, 0, 1] = 4.0
])

print("Original A_g shape:", A_g.shape)
print(A_g)

# Attempting to transpose: swap the last two axes
A_g_T = A_g.T
print("Transposed A_g shape:", A_g_T.shape)
print(A_g_T)

sigma = jnp.eye(2) * 0.1

print(A_g@sigma)