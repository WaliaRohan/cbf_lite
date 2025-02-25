import sympy as sp

from dynamics import SimpleDynamics

case = "Dubins" # "Simple", "Dubins"

if case == "Simple":
    dynamics = SimpleDynamics()

    x1, x2, u1, u2 = sp.symbols('x1 x2 u1 u2')

    x = sp.Matrix([x1, x2])
    u = sp.Matrix([u1, u2])

    f_matrix = sp.Matrix(dynamics.f_matrix)
    g_matrix = sp.Matrix(dynamics.g_matrix)

    x_dot = f_matrix*x + g_matrix*u

elif case == "Dubins":

    x, y, v, theta, u = sp.symbols('x y v theta u')

    u = sp.Matrix([u])

    x_vector = sp.Matrix([x, y, v, theta])

    f_matrix = sp.Matrix([
                v * sp.cos(theta),  # x_dot
                v * sp.sin(theta),  # y_dot
                0,                   # theta_dot (no drift)
                0                    # v_dot (velocity is constant)
            ])
    # g_matrix = sp.Matrix([
    #             0,  # No control influence on x
    #             0,  # No control influence on y
    #             0,  # Control directly affects theta (heading)
    #             1   # No control influence on velocity
    #         ])
    g_matrix = sp.Matrix([
                0,  # No control influence on x
                theta,  
                0,  # Control directly affects theta (heading)
                x   
            ])
    
    x_dot = f_matrix + g_matrix*u

# Print results
print("x_dot")

sp.pprint(x_dot)

J_x = x_dot.jacobian(x_vector)

print("Jacobian of x_dot with respect to x: ")

sp.pprint(J_x)

J_u = x_dot.jacobian(u)

print("Jacobian of x_dot with respect to u: ")

sp.pprint(J_u)

# Compute the joint Jacobian with respect to both x and u
z = sp.Matrix.vstack(x_vector, u)  # Combine x and u into a single vector
joint_jacobian = x_dot.jacobian(z)

# Print the result
print("Joint Jacobian [J_x | J_u]:")
sp.pprint(joint_jacobian)

