import numpy as np


def calculate_area(ur, br, bl, ul):
    # calculate the area of the cell
    # ul (upper left), bl (bottom left), br (bottom right), ur (upper right) are the coordinates of the four vertices of the cell
    # apply Gaussian trapezoidal formula to calculate the areas
    _surface_area = 0.5 * abs(
        (ur.x * br.y - br.x * ur.y)
        + (br.x * bl.y - bl.x * br.y)
        + (bl.x * ul.y - ul.x * bl.y)
        + (ul.x * ur.y - ur.x * ul.y)
    )

    return _surface_area


def dy(a, b):
    # Calculate distance between 'a' and 'b' along the y axis
    return abs(a.y - b.y)


def dx(a, b):
    # Calculate distance between 'a' and 'b' along the x axis
    return abs(a.x - b.x)


def dist(a, b):
    # Calculate the euclidean distance between 'a' and 'b'
    return abs((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def combine_meshes(bot_x, bot_y, top_x, top_y):
    compensation = (
        np.max(bot_y[0]) + np.min(bot_y[0]) - np.min(bot_y[1]) - np.max(bot_y[1])
    ) / 2
    c_x = np.concatenate((bot_x, bot_x))
    c_y = np.concatenate(
        (top_y + 15, bot_y)
    )  # 15.75 for linear or 10.5 for rect
    return c_x[1:-1, 1:-1], c_y[1:-1, 1:-1]


class Coordicate2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class SparseDense:
    def dense_to_sparse(self, dense_matrix):
        sparse_matrix = {}
        rows, cols = dense_matrix.shape

        for i in range(rows):
            for j in range(cols):
                if dense_matrix[i, j] != 0:
                    sparse_matrix[(i, j)] = dense_matrix[i, j]

        return sparse_matrix

    def sparse_to_dense(self, sparse_matrix):
        dense_matrix = np.zeros((len(sparse_matrix), len(sparse_matrix)))
        for (i, j), value in sparse_matrix.items():
            dense_matrix[i, j] = value

        return dense_matrix


unsteady_animation = None  # declare as global for animation to work


class BasicSettings:
    def __init__(
        self,
        dimX,
        dimY,
        N,
        M,
        shape,
        boundary,
        therm_con=1,
        TD=[],
        Tinf=25.0,
        alpha=-1,
        q=1,
        rho=1,
        c=1,
        time_steps=0,
        theta=1,
        problem_type="steady",
        mirrored=False,
    ):
        self.dimX = dimX
        self.dimY = dimY
        assert N == M, "Only equal N, M nodes supported"
        self.N = N
        self.M = M
        self.shape = shape
        self.boundary = boundary
        self.therm_con = therm_con
        self.TD = TD
        self.Tinf = Tinf
        self.alpha = alpha
        self.q = q
        self.time_steps = time_steps
        self.theta = theta
        self.problem_type = problem_type
        self.mirrored = mirrored

    def formfunction(self, x):
        self.h1 = 30
        self.hm = 4
        self.h2 = 20

        if self.shape == "linear" or self.shape == "half_fin":
            return (1 - x) * self.h1 / 2 + x * self.h2 / 2

        elif self.shape == "rectangular":
            return self.dimY

        elif self.shape == "quadratic":
            c1 = self.h2 + 2 * self.h1 / 2 - 2 * self.hm
            c2 = 2 * self.hm - 3 * self.h1 / 2 - self.h2 / 2
            c3 = self.h1 / 2
            return c1 * x**2 + c2 * x + c3

        elif self.shape == "crazy":
            d1 = 3
            d2 = 4
            return (
                (1 - x) * self.h1 / 2
                + x * self.h2 / 2
                + np.dot((np.sin(2 * np.pi * d1 * x)), (1 - (1 - 1 / d2) * x))
            )

        else:
            raise ValueError("Unknown shape: %s" % self.shape)

    def setUpMesh(self, N, M):
        # Initialize arrays to store node coordinates
        # X = np.zeros((M+1, M+1))
        # Y = np.zeros((M+1, M+1))

        # return X, Y
        x_nodes = N + 1
        y_nodes = M + 1

        x = np.linspace(0, self.dimX, x_nodes)
        self.X, _ = np.meshgrid(x, np.linspace(0, 1, y_nodes))

        self.Y = np.zeros((y_nodes, x_nodes))
        for i in range(x_nodes):
            self.Y[:, i] = np.linspace(self.formfunction(x[i] / self.dimX), 0, y_nodes)
        if self.mirrored:
            self.Y = -self.Y + self.h1 / 2
            self.Y = np.flip(self.Y, axis=0)
        self.dX = x_nodes / N


class SourceSettings(BasicSettings):
    def __init__(
        self,
        dimX,
        dimY,
        N,
        M,
        shape,
        boundary,
        source_region,
        observer_region,
        bc_x,
        bc_y,
        therm_con=1,
        TD=[],
        Tinf=0,
        alpha=-1,
        q=1,
        rho=1,
        c=1,
        time_steps=0,
        theta=1,
        problem_type="steady",
        G_N=5,
        G_M=5,
        alphas=[5, 5, 5, 5],
        mirrored=False,
    ):
        super().__init__(
            dimX,
            dimY,
            N,
            M,
            shape,
            boundary,
            therm_con,
            TD,
            Tinf,
            alpha,
            q,
            rho,
            c,
            time_steps,
            theta,
            problem_type,
            mirrored,
        )
        for b in boundary:
            assert b in ["N", "D", "R"], f"{b} is not a valid boundary!"
        self.source_field = self.get_field(source_region)
        self.observer_field = self.get_field(observer_region)
        self.bc_x = bc_x
        self.bc_y = bc_y
        self.G_N = G_N
        self.G_M = G_M
        self.alphas = alphas
        self.D = therm_con / (rho * c)

    def get_field(self, xyxy):
        points = [
            (x, y)
            for x in range(xyxy[0], xyxy[2] + 1)
            for y in range(xyxy[1], xyxy[3] + 1)
        ]
        return points


class FlowSettings(BasicSettings):
    def __init__(
        self,
        dimX,
        dimY,
        N,
        M,
        shape,
        boundary,
        u_bc_type,
        v_bc_type,
        p_bc_type,
        therm_con=1,
        TD=[],
        Tinf=0,
        alpha=-1,
        q=1,
        rho=1,
        c=1,
        time_steps=0,
        theta=1,
        problem_type="steady",
        mu=5.0,
        U_in=5.0,
        U_out=5.0,
        mirrored=False,
    ):
        super().__init__(
            dimX,
            dimY,
            N,
            M,
            shape,
            boundary,
            therm_con,
            TD,
            Tinf,
            alpha,
            q,
            rho,
            c,
            time_steps,
            theta,
            problem_type,
            mirrored,
        )
        self.u_bc_type = u_bc_type
        self.v_bc_type = v_bc_type
        self.p_bc_type = p_bc_type
        self.mu = mu
        self.U_in = U_in
        self.U_out = U_out
        self.D = therm_con / (rho * c)
