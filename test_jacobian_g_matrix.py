import jax
import jax.numpy as jnp
import sympy as sp

# b1, b2 = sp.symbols('b1 b2')
# # g = sp.Matrix([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9])

# g11, g12, g13, g21, g22, g23, g31, g32, g33 = sp.symbols('g11 g12 g13 g21 g22 g23 g31 g32 g33')
# G = sp.Matrix([[g11, g12, g13], [g21, g22, g23], [g31, g32, g33]])
# b = sp.Matrix([b1, b2])

# g = sp.Matrix

# A = sp.Matrix([G[:, j].jacobian(b) for j in range(G.cols)])

# sp.pprint(A)
# print(A.shape)

def g(b):
    """Compute a 2x3 matrix where each element is a constant times b[0] * b[1]."""
    # c = jnp.array([
    #     [11, 12, 13],  # Constants for row 1
    #     [21, 22, 23]   # Constants for row 2
    # ])

    c = jnp.array([
        [11, 12],  # Constants for row 1
        [21, 22]   # Constants for row 2
    ])
    
    return c * b[0] * b[1]  # Element-wise multiplication

b = jnp.array([1.0, 1.0]) 
A = jax.jacfwd(g)(b)

print(A)

print("--------------------")

first_section = A[:1].T
print(first_section.shape)

sigma = jnp.array([[0.1, -0.1], [-0.1, 0.1]])
# print(A, A.shape)
# print(A.T)

# print("--------------------------------")

# print(A@sigma)
# print(sigma@A.T)

# print("--------------------------------")
# Function to apply Sigma @ A_T for each batch
# print(A@sigma + sigma@A.T)
# def matmul_sigma(a_slice):
#     return sigma @ a_slice  # Shape (2,2) @ (2,3) → (2,3)

# # Vectorized multiplication
# Sigma_AT = jax.vmap(matmul_sigma, in_axes=0)(A.transpose(0, 2, 1))  # (2,3,2) → (2,2,3)

# print(Sigma_AT)
