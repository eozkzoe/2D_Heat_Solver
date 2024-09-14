import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import code


class Greens:
    def __init__(self, L, cond, N, M, bc_x, bc_y, alphas=[]):
        self.alphas = alphas
        self.L = L
        self.cond = cond
        self.N = N  # degree of accuracy of green function in X-direction
        self.M = M  # degree of accuracy of green function in Y-direction
        self.bc_x = bc_x
        self.bc_y = bc_y

    def get_B(self, i):
        b = self.alphas[i] * self.L / self.cond
        return b

    def get_eigenvalue(self, m, bc_numeric):
        match bc_numeric:
            case 11:
                return (m * np.pi) / self.L
            case 12:
                return ((2 * m - 1) * np.pi) / (2 * self.L)
            case 13:

                def equation(beta):
                    return beta * self.L * (1 / np.tan(beta * self.L)) + self.get_B(2)

                beta_m = fsolve(equation, (m * np.pi) / self.L)[0]
                return beta_m
            case 23:

                def equation(beta):
                    return beta * self.L * np.tan(beta * self.L) - self.get_B(2)

                beta_m = fsolve(equation, (m * np.pi) / self.L)[0]
                return beta_m
            case 33:

                def equation(beta):
                    return np.tan(beta * self.L) + (
                        beta * self.L * (self.alphas[0] + self.alphas[1]) / self.cond
                    ) / (
                        beta**2 * self.L**2
                        - (self.alphas[0] * self.alphas[1]) / self.cond**2
                    )

                beta_m = fsolve(equation, (m * np.pi) / self.L)[0]
                return beta_m

    def get_eigenfunction(self, m, bc_numeric):
        beta_m = self.get_eigenvalue(m, bc_numeric)
        match bc_numeric:
            case 11 | 12 | 13:
                return lambda x: np.sin(beta_m * x)

            case 23:
                return lambda x: np.cos(beta_m * x)

            case 33:
                return lambda x: beta_m * self.L * np.cos(beta_m * x) + self.get_B(
                    1
                ) * np.sin(beta_m * x)

    def get_norm(self, m, bc_numeric):
        match bc_numeric:
            case 11 | 12:
                return self.L / 2
            case 13 | 23 | 33:
                return self.L / (2 * self.get_phi(2, m, bc_numeric))

    def get_phi(self, i, m, bc_numeric):
        beta_m = self.get_eigenvalue(m, bc_numeric)
        b = self.get_B(i)
        return ((beta_m**2) * (self.L**2)) / ((beta_m**2) * (self.L**2) + b**2 + b)

    def get_PHI(self, m):
        phi = self.get_phi(2, m)
        beta_m = self.get_eigenvalue(m)
        b = self.get_B(1)
        return phi / (beta_m**2 * self.L**2 + b**2 + b * phi)

    def construct_G2D(self, x, y, xs, ys):
        summation = 0
        for m in range(1, self.M + 1):
            for n in range(1, self.N + 1):
                beta_m = self.get_eigenvalue(m, self.bc_x)
                theta_n = self.get_eigenvalue(n, self.bc_y)
                Xm = self.get_eigenfunction(m, self.bc_x)
                Yn = self.get_eigenfunction(n, self.bc_y)
                Nx = self.get_norm(m, self.bc_x)
                Ny = self.get_norm(n, self.bc_y)
                _first = 1 / (beta_m**2 + theta_n**2)
                _second = (Xm(x) * Xm(xs)) / Nx
                _third = (Yn(y) * Yn(ys)) / Ny
                overall = _first * _second * _third
                summation += overall
        return summation


class AnalyticalTemp(Greens):
    def __init__(
        self,
        L,
        cond,
        N,
        M,
        bc_x,
        bc_y,
        sources: list[tuple],
        observers: list[tuple],
        alphas=[],
    ):
        super().__init__(L, cond, N, M, bc_x, bc_y, alphas)
        self.sources = sources  # source vector
        self.observers = observers  # points of observation

    def integrate(self, curve, delta):
        return np.trapz(curve, dx=delta)

    def source_integral(self, x, y):
        delta = self.L / len(self.sources)
        values = []
        for xs, ys in self.sources:
            G_2D = self.construct_G2D(x, y, xs, ys)
            values.append(G_2D * self.source_distro(xs, ys))
        return 1 / self.cond * np.trapz(values, dx=delta)

    def finite_difference(self, func, x, y, xs, ys, delta=1e-5):
        # Central difference approximation for partial derivatives
        G_xs_plus = func(x, xs + delta, y, ys)
        G_xs_minus = func(x, xs - delta, y, ys)
        G_ys_plus = func(x, xs, y, ys + delta)
        G_ys_minus = func(x, xs, y, ys - delta)

        dGdxs = (G_xs_plus - G_xs_minus) / (2 * delta)
        dGdys = (G_ys_plus - G_ys_minus) / (2 * delta)

        return dGdxs, dGdys

    def boundary_integral(self, x, y):
        delta = 1e-5  # small perturbation for numerical derivative
        integral_values = []

        for direction in ["E", "W", "N", "S"]:
            values = []
            for xs, ys in self.sources:
                G_2D = self.construct_G2D(x, xs, y, ys)
                dGdxs, dGdys = self.finite_difference(
                    self.construct_G2D, x, y, xs, ys, delta
                )
                dTdxs = (
                    0  # Assuming T(xs, ys) = 0 due to homogeneous Dirichlet conditions
                )
                dTdys = (
                    0  # Assuming T(xs, ys) = 0 due to homogeneous Dirichlet conditions
                )

                if direction == "E":
                    dy = 1  # Step in y-direction
                    n = np.array(
                        [1, 0]
                    )  # Normal vector pointing outwards on the east boundary
                    values.append((G_2D * dTdxs - 0 * dGdxs) * np.dot(n, [1, 1]) * dy)
                elif direction == "W":
                    dy = 1  # Step in y-direction
                    n = np.array(
                        [-1, 0]
                    )  # Normal vector pointing outwards on the west boundary
                    values.append((G_2D * dTdxs - 0 * dGdxs) * np.dot(n, [1, 1]) * dy)
                elif direction == "N":
                    dx = 1  # Step in x-direction
                    n = np.array(
                        [0, 1]
                    )  # Normal vector pointing outwards on the north boundary
                    values.append((G_2D * dTdys - 0 * dGdys) * np.dot(n, [1, 1]) * dx)
                elif direction == "S":
                    dx = 1  # Step in x-direction
                    n = np.array(
                        [0, -1]
                    )  # Normal vector pointing outwards on the south boundary
                    values.append((G_2D * dTdys - 0 * dGdys) * np.dot(n, [1, 1]) * dx)

            integral_values.append(self.integrate(values, delta))

        return sum(integral_values)

    def compute_contributions(self):
        temperatures = []
        for x, y in self.observers:
            Txy = self.source_integral(x, y) + self.boundary_integral(x, y)
            temperatures.append(Txy)
        greens_field = np.asarray(temperatures).reshape(self.L - 1, self.L - 1)
        return np.flipud(greens_field.T)

    def source_distro(self, xs, ys):
        return 0.001  # uniform source assumption for now...


def combine_finite_diff_and_green(T_map, greens_field, fill=False):
    if not fill:
        assert T_map.shape[1] == greens_field.shape[1], (
            f"Incompatible shapes Temp: {T_map.shape}, Greens: {greens_field.shape}\n"
            f"Either reshape Greens field or change observer values to match T_map shape"
        )
    else:
        padding_rows = T_map.shape[0] - greens_field.shape[0]
        padding = np.zeros((padding_rows, greens_field.shape[1]))
        greens_field = np.vstack((padding, greens_field))
    T_map_with_source = np.zeros_like(T_map)
    T_map_with_source = T_map + greens_field
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.matshow(greens_field, cmap=plt.cm.Greys)
    # plt.show()

    return T_map_with_source
