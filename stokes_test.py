import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def formfunction(x, dimY, shape):
    h1 = 5
    hm = 4
    h2 = 1.5

    if shape == "linear" or shape == "half_fin":
        return (1 - x) * h1 / 2 + x * h2 / 2

    elif shape == "rectangular":
        return 0.5 * dimY

    elif shape == "quadratic":
        c1 = h2 + 2 * h1 / 2 - 2 * hm
        c2 = 2 * hm - 3 * h1 / 2 - h2 / 2
        c3 = h1 / 2
        return c1 * x**2 + c2 * x + c3

    elif shape == "crazy":
        d1 = 3
        d2 = 4
        return (
            (1 - x) * h1 / 2
            + x * h2 / 2
            + np.dot((np.sin(2 * np.pi * d1 * x)), (1 - (1 - 1 / d2) * x))
        )

    else:
        raise ValueError("Unknown shape: %s" % shape)


def stokes_flow(dimX, dimY, nx, ny, form_shape):
    dx = dimX / (nx - 1)
    dy = dimY / (ny - 1)
    mu = 1.0  # Dynamic viscosity

    # Grid points
    x = np.linspace(0, dimX, nx)
    y = np.linspace(0, dimY, ny)

    # Create the mesh grid
    X, Y = np.meshgrid(x, y)

    # Apply the form function to the upper boundary
    upper_boundary = formfunction(x / dimX, dimY, form_shape)

    # Initialize sparse matrix and RHS vector
    A = sp.lil_matrix((2 * nx * ny, 2 * nx * ny))
    b = np.zeros(2 * nx * ny)

    def index(i, j, var):
        return (i * ny + j) * 2 + var

    # Fill the matrix A and vector b
    for i in range(nx):
        for j in range(ny):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                # Boundary conditions (u = 0, v = 0)
                A[index(i, j, 0), index(i, j, 0)] = 1
                A[index(i, j, 1), index(i, j, 1)] = 1
            else:
                # Stokes equations
                # u-momentum
                A[index(i, j, 0), index(i, j, 0)] = -2 * (dx**2 + dy**2)
                A[index(i, j, 0), index(i - 1, j, 0)] = dy**2
                A[index(i, j, 0), index(i + 1, j, 0)] = dy**2
                A[index(i, j, 0), index(i, j - 1, 0)] = dx**2
                A[index(i, j, 0), index(i, j + 1, 0)] = dx**2
                A[index(i, j, 0), index(i, j, 1)] = -mu * dx * dy

                # v-momentum
                A[index(i, j, 1), index(i, j, 1)] = -2 * (dx**2 + dy**2)
                A[index(i, j, 1), index(i - 1, j, 1)] = dy**2
                A[index(i, j, 1), index(i + 1, j, 1)] = dy**2
                A[index(i, j, 1), index(i, j - 1, 1)] = dx**2
                A[index(i, j, 1), index(i, j + 1, 1)] = dx**2
                A[index(i, j, 1), index(i, j, 0)] = -mu * dx * dy

    # Convert to CSR format for efficient solving
    A = A.tocsr()

    # Solve the linear system
    U = spla.spsolve(A, b)

    # Extract velocity components
    u = U[0::2].reshape(nx, ny)
    v = U[1::2].reshape(nx, ny)

    # Replace NaNs and infs with zeros
    u[np.isnan(u) | np.isinf(u)] = 0
    v[np.isnan(v) | np.isinf(v)] = 0

    return X, Y, u, v


# Parameters
dimX = 10
dimY = 5
nx = 50
ny = 25
form_shape = "linear"

# Solve the Stokes flow
X, Y, u, v = stokes_flow(dimX, dimY, nx, ny, form_shape)

# Plot the results
plt.figure(figsize=(10, 6))
plt.quiver(X, Y, u, v)
plt.title("Velocity Field")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
