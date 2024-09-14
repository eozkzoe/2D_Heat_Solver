import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class StokesSolver:
    def __init__(self, Lx, Ly, Nx, Ny, mu, U_in, U_out):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx + 1
        self.Ny = Ny + 1
        self.mu = mu
        self.U_in = U_in
        self.U_out = U_out

        self.dx = Lx / self.Nx
        self.dy = Ly / self.Ny

        self.u = np.zeros((self.Nx, self.Ny))
        self.v = np.zeros((self.Nx, self.Ny))
        self.p = np.zeros((self.Nx, self.Ny))

    def set_boundary_conditions(self, u_bc_type, v_bc_type, p_bc_type):
        # Inlet (left)
        if u_bc_type["left"] == "D":
            self.u[0, :] = self.U_in
        if v_bc_type["left"] == "D":
            self.v[0, :] = 0
        if p_bc_type["left"] == "N":
            self.p[0, :] = self.p[1, :]  # Neumann: dp/dx = 0

        # Outlet (right)
        if u_bc_type["right"] == "D":
            self.u[-1, :] = self.U_out
        if v_bc_type["right"] == "D":
            self.v[-1, :] = 0
        if u_bc_type["right"] == "N":
            self.u[-1, :] = self.u[-2, :]
        if v_bc_type["right"] == "N":
            self.v[-1, :] = self.v[-2, :]
        if p_bc_type["right"] == "D":
            self.p[-1, :] = 0
        if p_bc_type["right"] == "N":
            self.p[-1, :] = self.p[-2, :]  # Neumann: dp/dx = 0

        # Top, bottom (No slip)
        if u_bc_type["top"] == "D":
            self.u[:, -1] = 0
        if u_bc_type["bottom"] == "D":
            self.u[:, 0] = 0
        if v_bc_type["top"] == "D":
            self.v[:, -1] = 0
        if v_bc_type["bottom"] == "D":
            self.v[:, 0] = 0

        if p_bc_type["top"] == "N":
            self.p[:, -1] = self.p[:, -2]  # Neumann: dp/dy = 0
        if p_bc_type["bottom"] == "N":
            self.p[:, 0] = self.p[:, 1]  # Neumann: dp/dy = 0

    def solve_stokes(self, u_bc_type, v_bc_type, p_bc_type, max_iter=500, tol=1e-5):
        for it in range(max_iter):
            u_old = self.u.copy()
            v_old = self.v.copy()
            p_old = self.p.copy()

            # Velocity update (explicit)
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    self.u[i, j] = (
                        (u_old[i + 1, j] + u_old[i - 1, j]) * self.dy**2
                        + (u_old[i, j + 1] + u_old[i, j - 1]) * self.dx**2
                    ) / (2 * (self.dx**2 + self.dy**2)) - (
                        self.dx**2 * self.dy**2
                    ) / (
                        2 * self.mu * (self.dx**2 + self.dy**2)
                    ) * (
                        p_old[i + 1, j] - p_old[i - 1, j]
                    ) / (
                        2 * self.dx
                    )
                    self.v[i, j] = (
                        (v_old[i + 1, j] + v_old[i - 1, j]) * self.dy**2
                        + (v_old[i, j + 1] + v_old[i, j - 1]) * self.dx**2
                    ) / (2 * (self.dx**2 + self.dy**2)) - (
                        self.dx**2 * self.dy**2
                    ) / (
                        2 * self.mu * (self.dx**2 + self.dy**2)
                    ) * (
                        p_old[i, j + 1] - p_old[i, j - 1]
                    ) / (
                        2 * self.dy
                    )

            # Pressure update (explicit)
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    self.p[i, j] = (
                        (p_old[i + 1, j] + p_old[i - 1, j]) * self.dy**2
                        + (p_old[i, j + 1] + p_old[i, j - 1]) * self.dx**2
                    ) / (2 * (self.dx**2 + self.dy**2)) - (
                        self.dx**2 * self.dy**2
                    ) / (
                        2 * (self.dx**2 + self.dy**2)
                    ) * (
                        (self.u[i + 1, j] - self.u[i - 1, j]) / (2 * self.dx)
                        + (self.v[i, j + 1] - self.v[i, j - 1]) / (2 * self.dy)
                    )

            self.set_boundary_conditions(u_bc_type, v_bc_type, p_bc_type)

            # Calculate convergence
            u_diff = np.linalg.norm(self.u - u_old)
            v_diff = np.linalg.norm(self.v - v_old)
            p_diff = np.linalg.norm(self.p - p_old)

            if u_diff < tol and v_diff < tol and p_diff < tol:
                print(f"Converged in {it} iterations.")
                break

    def plot_results(self):
        X, Y = np.meshgrid(
            np.linspace(0, self.Lx, self.Nx), np.linspace(0, self.Ly, self.Ny)
        )

        # 2D plot
        plt.figure(figsize=(10, 6))
        plt.contourf(X, Y, self.u.T, levels=20, cmap="jet")
        plt.colorbar(label="u velocity")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Velocity field (u component)")
        plt.show()

        # 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, self.u.T, cmap="jet")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u velocity")
        ax.set_title("Velocity field (u component) - 3D view")
        plt.show()


# Main
if __name__ == "__main__":
    Lx = 20
    Ly = 10
    Nx = 20
    Ny = 20
    mu = 5.0
    U_in = 5.0
    U_out = 2.0

    solver = StokesSolver(Lx, Ly, Nx, Ny, mu, U_in, U_out)

    u_bc_type = {"left": "D", "right": "D", "top": "D", "bottom": "D"}
    v_bc_type = {"left": "D", "right": "D", "top": "D", "bottom": "D"}
    p_bc_type = {"left": "N", "right": "D", "top": "N", "bottom": "N"}

    solver.set_boundary_conditions(u_bc_type, v_bc_type, p_bc_type)
    solver.solve_stokes(u_bc_type, v_bc_type, p_bc_type, max_iter=1000, tol=1e-5)
    solver.plot_results()
