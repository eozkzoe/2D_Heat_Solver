import numpy as np
from utils import calculate_area, dx, dy, dist, Coordicate2D
from scipy import sparse
import matplotlib.pyplot as plt
from greens import combine_finite_diff_and_green


class Heat2D_FVM:
    def __init__(
        self, N, M, X, Y, D, dX, boundary=[], TD=[], q=0.0, alpha=0.0, Tinf=0.0
    ):
        # i, j is the index of the cell
        # X, Y is the mesh
        # boundary is the boundary condition: 'TD', 'q', 'alpha', 'Tinf'
        # TD is the temperature difference
        # q is the heat flux
        # alpha is the heat transfer coefficient
        # Tinf is the temperature of the surrounding

        self.X = X
        self.Y = Y
        self.boundary = boundary
        self.TD = TD
        self.q = q
        self.alpha = alpha
        self.Tinf = Tinf
        self.dX = dX

        # n is the number of points in the first direction
        # m is the number of points in the second direction
        self.n = N + 1
        self.m = M + 1
        self.nm = self.n * self.m

        self.A = np.zeros((self.nm, self.nm))
        self.B = np.zeros(self.nm)

        self.D = D
        self.iters = 0

    def index(self, i, j):
        # Return the index in the computational vector based on the physical indices 'i' and 'j'
        return i * (self.n) + j

    def second_order_stencil(self, i, j, use_inner_at):
        # Based on 'i','j' decide if the node is inner or boundary (which boundary?)
        if i == 0:
            if j == 0:
                # NW corner
                # print(f"building NW at {i}, {j}")
                _nw, _nwb = self.build_NW(i, j)
                # if (use_inner_at is not None) and use_inner_at == "N":
                #     _nw = -_nw
                #     _nw[self.index(i, j)] = _nw[self.index(i, j)] * 0.33
                return _nw, _nwb
            elif j == self.m - 1:
                # NE corner
                # print(f"building NE at {i}, {j}")
                _ne, _neb = self.build_NE(i, j)
                # if (use_inner_at is not None) and use_inner_at == "N":
                # _ne = -_ne
                # _ne[self.index(i, j)] = _ne[self.index(i, j)] * 0.33
                return _ne, _neb
            else:
                # North boundary
                # print(f"building N at {i}, {j}")
                _north, _nb = self.build_north(i, j)
                if (use_inner_at is not None) and use_inner_at == "N":
                    _north[self.index(i, j)] = -_north[self.index(i, j)]
                    _north[self.index(i + 1, j)] = _north[self.index(i + 1, j)]
                return _north, _nb
        elif i == self.n - 1:
            if j == 0:
                # SW corner
                # print(f"building SW at {i}, {j}")
                _sw, _swb = self.build_SW(i, j)
                # if (use_inner_at is not None) and use_inner_at == "S":
                # _sw = -_sw
                # _sw[self.index(i, j)] = _sw[self.index(i, j)] * 0.33
                return _sw, _swb
            elif j == self.m - 1:
                # SE corner
                # print(f"building SE at {i}, {j}")
                _se, _seb = self.build_SE(i, j)
                # if (use_inner_at is not None) and use_inner_at == "S":
                # _se = -_se
                # _se[self.index(i, j)] = _se[self.index(i, j)] * 0.33
                return _se, _seb
            else:
                # North boundary
                # print(f"building N at {i}, {j}")
                _south, _sb = self.build_south(i, j)
                if (use_inner_at is not None) and use_inner_at == "S":
                    _south[self.index(i, j)] = -_south[self.index(i, j)]
                    _south[self.index(i - 1, j)] = _south[self.index(i - 1, j)]
                return _south, _sb
        elif j == 0:
            # West boundary
            # print(f"building W at {i}, {j}")
            return self.build_west(i, j)
        elif j == self.m - 1:
            # East boundary
            # print(f"building E at {i}, {j}")
            return self.build_east(i, j)
        else:
            # print(f"building I at {i}, {j}")
            return self.build_inner(i, j)

    def build_inner(self, i, j):
        stencil = np.zeros(self.nm)
        b = 0
        # % Nomenclature:
        # %
        # %    NW(i-1,j-1)   Nw -  N(i-1,j) -  Ne     NE(i-1,j+1)
        # %
        # %                 |                 |
        # %
        # %       nW - - - - nw ------ n ------ ne - - - nE
        # %                 |                 |
        # %       |         |        |        |       |
        # %                 |                 |
        # %   W(i, j-1) - - w - - P (i,j) - - e - -  E (i,j+1)
        # %                 |                 |
        # %       |         |        |        |       |
        # %                 |                 |
        # %      sW - - - - sw ------ s ------ se - - - sE
        # %
        # %                 |                 |
        # %
        # %   SW(i+1,j-1)   Sw  -  S(i+1,j)  - Se      SE(i+1,j+1)
        # %
        # % Indexing of stencil:

        # %    D_4 - D_1 - D2
        # %     |     |     |
        # %    D_3 - D_0 - D3
        # %     |     |     |
        # %    D_2 -  D1 - D4

        # principle node coordinate
        P = Coordicate2D(self.X[i, j], self.Y[i, j])
        N = Coordicate2D(self.X[i - 1, j], self.Y[i - 1, j])
        S = Coordicate2D(self.X[i + 1, j], self.Y[i + 1, j])
        W = Coordicate2D(self.X[i, j - 1], self.Y[i, j - 1])
        E = Coordicate2D(self.X[i, j + 1], self.Y[i, j + 1])
        NW = Coordicate2D(self.X[i - 1, j - 1], self.Y[i - 1, j - 1])
        NE = Coordicate2D(self.X[i - 1, j + 1], self.Y[i - 1, j + 1])
        SW = Coordicate2D(self.X[i + 1, j - 1], self.Y[i + 1, j - 1])
        SE = Coordicate2D(self.X[i + 1, j + 1], self.Y[i + 1, j + 1])

        # auxiliary node coordinate
        Nw = Coordicate2D((N.x + NW.x) / 2, (N.y + NW.y) / 2)
        Ne = Coordicate2D((N.x + NE.x) / 2, (N.y + NE.y) / 2)
        Sw = Coordicate2D((S.x + SW.x) / 2, (S.y + SW.y) / 2)
        Se = Coordicate2D((S.x + SE.x) / 2, (S.y + SE.y) / 2)
        nW = Coordicate2D((W.x + NW.x) / 2, (W.y + NW.y) / 2)
        nE = Coordicate2D((E.x + NE.x) / 2, (E.y + NE.y) / 2)
        sW = Coordicate2D((W.x + SW.x) / 2, (W.y + SW.y) / 2)
        sE = Coordicate2D((E.x + SE.x) / 2, (E.y + SE.y) / 2)

        n = Coordicate2D((N.x + P.x) / 2, (N.y + P.y) / 2)
        s = Coordicate2D((S.x + P.x) / 2, (S.y + P.y) / 2)
        w = Coordicate2D((W.x + P.x) / 2, (W.y + P.y) / 2)
        e = Coordicate2D((E.x + P.x) / 2, (E.y + P.y) / 2)

        se = Coordicate2D((Se.x + e.x) / 2, (Se.y + e.y) / 2)
        sw = Coordicate2D((Sw.x + w.x) / 2, (Sw.y + w.y) / 2)
        ne = Coordicate2D((Ne.x + e.x) / 2, (Ne.y + e.y) / 2)
        nw = Coordicate2D((Nw.x + w.x) / 2, (Nw.y + w.y) / 2)

        # calculate the area of the cell
        S_P = calculate_area(ne, se, sw, nw)
        S_n = calculate_area(Ne, e, w, Nw)
        S_s = calculate_area(e, Se, Sw, w)
        S_w = calculate_area(n, s, sW, nW)
        S_e = calculate_area(nE, sE, s, n)
        # print(S_P, S_n, S_s, S_w, S_e)

        D3 = (
            (dx(se, ne) * (dx(nE, n) / 4 + dx(s, sE) / 4 + dx(sE, nE))) / S_e
            + (dy(se, ne) * (dy(nE, n) / 4 + dy(s, sE) / 4 + dy(sE, nE))) / S_e
            + (dx(e, Ne) * dx(ne, nw)) / (4 * S_n)
            + (dx(Se, e) * dx(sw, se)) / (4 * S_s)
            + (dy(e, Ne) * dy(ne, nw)) / (4 * S_n)
            + (dy(Se, e) * dy(sw, se)) / (4 * S_s)
        ) / S_P
        D_3 = (
            (dx(nw, sw) * (dx(n, nW) / 4 + dx(sW, s) / 4 + dx(nW, sW))) / S_w
            + (dy(nw, sw) * (dy(n, nW) / 4 + dy(sW, s) / 4 + dy(nW, sW))) / S_w
            + (dx(Nw, w) * dx(ne, nw)) / (4 * S_n)
            + (dx(w, Sw) * dx(sw, se)) / (4 * S_s)
            + (dy(Nw, w) * dy(ne, nw)) / (4 * S_n)
            + (dy(w, Sw) * dy(sw, se)) / (4 * S_s)
        ) / S_P
        D1 = (
            (dx(sw, se) * (dx(Se, e) / 4 + dx(w, Sw) / 4 + dx(Sw, Se))) / S_s
            + (dy(sw, se) * (dy(Se, e) / 4 + dy(w, Sw) / 4 + dy(Sw, Se))) / S_s
            + (dx(s, sE) * dx(se, ne)) / (4 * S_e)
            + (dx(sW, s) * dx(nw, sw)) / (4 * S_w)
            + (dy(s, sE) * dy(se, ne)) / (4 * S_e)
            + (dy(sW, s) * dy(nw, sw)) / (4 * S_w)
        ) / S_P
        # North
        D_1 = (
            (dx(ne, nw) * (dx(e, Ne) / 4 + dx(Nw, w) / 4 + dx(Ne, Nw))) / S_n
            + (dy(ne, nw) * (dy(e, Ne) / 4 + dy(Nw, w) / 4 + dy(Ne, Nw))) / S_n
            + (dx(nE, n) * dx(se, ne)) / (4 * S_e)
            + (dx(n, nW) * dx(nw, sw)) / (4 * S_w)
            + (dy(nE, n) * dy(se, ne)) / (4 * S_e)
            + (dy(n, nW) * dy(nw, sw)) / (4 * S_w)
        ) / S_P

        # NW
        D_4 = (
            (dx(Nw, w) * dx(ne, nw)) / (4 * S_n)
            + (dx(n, nW) * dx(nw, sw)) / (4 * S_w)
            + (dy(Nw, w) * dy(ne, nw)) / (4 * S_n)
            + (dy(n, nW) * dy(nw, sw)) / (4 * S_w)
        ) / S_P

        # NE
        D2 = (
            (dx(nE, n) * dx(se, ne)) / (4 * S_e)
            + (dx(e, Ne) * dx(ne, nw)) / (4 * S_n)
            + (dy(nE, n) * dy(se, ne)) / (4 * S_e)
            + (dy(e, Ne) * dy(ne, nw)) / (4 * S_n)
        ) / S_P

        # SW
        D_2 = (
            (dx(w, Sw) * dx(sw, se)) / (4 * S_s)
            + (dx(sW, s) * dx(nw, sw)) / (4 * S_w)
            + (dy(w, Sw) * dy(sw, se)) / (4 * S_s)
            + (dy(sW, s) * dy(nw, sw)) / (4 * S_w)
        ) / S_P

        # SE
        D4 = (
            (dx(s, sE) * dx(se, ne)) / (4 * S_e)
            + (dx(Se, e) * dx(sw, se)) / (4 * S_s)
            + (dy(s, sE) * dy(se, ne)) / (4 * S_e)
            + (dy(Se, e) * dy(sw, se)) / (4 * S_s)
        ) / S_P

        # Center (P)
        D0 = (
            (dx(se, ne) * (dx(n, s) + dx(nE, n) / 4 + dx(s, sE) / 4)) / S_e
            + (dx(ne, nw) * (dx(w, e) + dx(e, Ne) / 4 + dx(Nw, w) / 4)) / S_n
            + (dx(sw, se) * (dx(e, w) + dx(Se, e) / 4 + dx(w, Sw) / 4)) / S_s
            + (dx(nw, sw) * (dx(s, n) + dx(n, nW) / 4 + dx(sW, s) / 4)) / S_w
            + (dy(se, ne) * (dy(n, s) + dy(nE, n) / 4 + dy(s, sE) / 4)) / S_e
            + (dy(ne, nw) * (dy(w, e) + dy(e, Ne) / 4 + dy(Nw, w) / 4)) / S_n
            + (dy(sw, se) * (dy(e, w) + dy(Se, e) / 4 + dy(w, Sw) / 4)) / S_s
            + (dy(nw, sw) * (dy(s, n) + dy(n, nW) / 4 + dy(sW, s) / 4)) / S_w
        ) / S_P

        stencil[self.index(i, j)] = -D0
        stencil[self.index(i - 1, j)] = D_1
        stencil[self.index(i + 1, j)] = D1
        stencil[self.index(i, j - 1)] = D_3
        stencil[self.index(i, j + 1)] = D3
        stencil[self.index(i - 1, j - 1)] = D_4
        stencil[self.index(i - 1, j + 1)] = D2
        stencil[self.index(i + 1, j - 1)] = D_2
        stencil[self.index(i + 1, j + 1)] = D4

        return stencil, b

    def build_north(self, i, j):
        stencil = np.zeros(self.nm)
        b = 0
        if self.boundary[0] == "D":
            stencil[self.index(i, j)] = 1.0
            b = self.TD[0]
        else:
            # principle node coordinate
            P = Coordicate2D(self.X[i, j], self.Y[i, j])
            S = Coordicate2D(self.X[i + 1, j], self.Y[i + 1, j])
            W = Coordicate2D(self.X[i, j - 1], self.Y[i, j - 1])
            E = Coordicate2D(self.X[i, j + 1], self.Y[i, j + 1])
            SW = Coordicate2D(self.X[i + 1, j - 1], self.Y[i + 1, j - 1])
            SE = Coordicate2D(self.X[i + 1, j + 1], self.Y[i + 1, j + 1])

            # auxiliary node coordinate
            Sw = Coordicate2D((S.x + SW.x) / 2, (S.y + SW.y) / 2)
            Se = Coordicate2D((S.x + SE.x) / 2, (S.y + SE.y) / 2)
            sW = Coordicate2D((W.x + SW.x) / 2, (W.y + SW.y) / 2)
            sE = Coordicate2D((E.x + SE.x) / 2, (E.y + SE.y) / 2)

            s = Coordicate2D((S.x + P.x) / 2, (S.y + P.y) / 2)
            w = Coordicate2D((W.x + P.x) / 2, (W.y + P.y) / 2)
            e = Coordicate2D((E.x + P.x) / 2, (E.y + P.y) / 2)

            se = Coordicate2D((Se.x + e.x) / 2, (Se.y + e.y) / 2)
            sw = Coordicate2D((Sw.x + w.x) / 2, (Sw.y + w.y) / 2)

            # calculate the area of the cell
            S_ss = calculate_area(e, se, sw, w)
            S_s = calculate_area(e, Se, Sw, w)
            S_ssw = calculate_area(P, s, sW, W)
            S_sse = calculate_area(E, sE, s, P)

            # East
            D3 = (
                dy(sw, se) * (dy(Se, e) / 4) / S_s
                + dx(sw, se) * (dx(Se, e) / 4) / S_s
                + dy(se, e) * (dy(s, sE) / 4 + 3 * dy(sE, E) / 4 + dy(E, P) / 2) / S_sse
                + dx(se, e) * (dx(s, sE) / 4 + 3 * dx(sE, E) / 4 + dx(E, P) / 2) / S_sse
            ) / S_ss

            # West
            D_3 = (
                dy(w, sw) * (3 * dy(W, sW) / 4 + dy(sW, s) / 4 + dy(P, W) / 2) / S_ssw
                + dx(w, sw) * (3 * dx(W, sW) / 4 + dx(sW, s) / 4 + dx(P, W) / 2) / S_ssw
                + dy(sw, se) * (dy(w, Sw) / 4) / S_s
                + dx(sw, se) * (dx(w, Sw) / 4) / S_s
            ) / S_ss

            # South
            D1 = (
                dy(w, sw) * (dy(sW, s) / 4 + dy(s, P) / 4) / S_ssw
                + dx(w, sw) * (dx(sW, s) / 4 + dx(s, P) / 4) / S_ssw
                + dy(sw, se) * (dy(w, Sw) / 4 + dy(Sw, Se) + dy(Se, e) / 4) / S_s
                + dx(sw, se) * (dx(w, Sw) / 4 + dx(Sw, Se) + dx(Se, e) / 4) / S_s
                + dy(se, e) * (dy(P, s) / 4 + dy(s, sE) / 4) / S_sse
                + dx(se, e) * (dx(P, s) / 4 + dx(s, sE) / 4) / S_sse
            ) / S_ss

            # SW
            D_2 = (
                dy(w, sw) * (dy(W, sW) / 4 + dy(sW, s) / 4) / S_ssw
                + dx(w, sw) * (dx(W, sW) / 4 + dx(sW, s) / 4) / S_ssw
                + dy(sw, se) * (dy(w, Sw) / 4) / S_s
                + dx(sw, se) * (dx(w, Sw) / 4) / S_s
            ) / S_ss

            # SE
            D4 = (
                dy(sw, se) * (dy(Se, e) / 4) / S_s
                + dx(sw, se) * (dx(Se, e) / 4) / S_s
                + dy(se, e) * (dy(s, sE) / 4 + dy(sE, E) / 4) / S_sse
                + dx(se, e) * (dx(s, sE) / 4 + dx(sE, E) / 4) / S_sse
            ) / S_ss

            coefficient = 0.0
            if self.boundary[0] == "N":
                coefficient = 0.0
                b = self.q * dist(e, w) / S_ss
            elif self.boundary[0] == "R":
                coefficient = self.alpha
                b = self.alpha * self.Tinf * dist(e, w) / S_ss

            D0 = (
                coefficient * dist(e, w)
                + dy(w, sw) * (dy(sW, s) / 4 + 3 * dy(s, P) / 4 + dy(P, W) / 2) / S_ssw
                + dx(w, sw) * (dx(sW, s) / 4 + 3 * dx(s, P) / 4 + dx(P, W) / 2) / S_ssw
                + dy(sw, se) * (dy(w, Sw) / 4 + dy(Se, e) / 4 + dy(e, w)) / S_s
                + dx(sw, se) * (dx(w, Sw) / 4 + dx(Se, e) / 4 + dx(e, w)) / S_s
                + dy(se, e) * (3 * dy(P, s) / 4 + dy(s, sE) / 4 + dy(E, P) / 2) / S_sse
                + dx(se, e) * (3 * dx(P, s) / 4 + dx(s, sE) / 4 + dx(E, P) / 2) / S_sse
            ) / S_ss

            stencil[self.index(i, j)] = D0
            stencil[self.index(i + 1, j)] = D1
            stencil[self.index(i, j - 1)] = D_3
            stencil[self.index(i, j + 1)] = D3
            stencil[self.index(i + 1, j - 1)] = D_2
            stencil[self.index(i + 1, j + 1)] = D4

        return stencil, b

    def build_south(self, i, j):
        stencil = np.zeros(self.nm)
        b = 0
        if self.boundary[1] == "D":
            stencil[self.index(i, j)] = 1.0
            b = self.TD[1]

        else:
            # principle node coordinate
            P = Coordicate2D(self.X[i, j], self.Y[i, j])
            N = Coordicate2D(self.X[i - 1, j], self.Y[i - 1, j])
            W = Coordicate2D(self.X[i, j - 1], self.Y[i, j - 1])
            E = Coordicate2D(self.X[i, j + 1], self.Y[i, j + 1])
            NW = Coordicate2D(self.X[i - 1, j - 1], self.Y[i - 1, j - 1])
            NE = Coordicate2D(self.X[i - 1, j + 1], self.Y[i - 1, j + 1])

            # auxiliary node coordinate
            Nw = Coordicate2D((N.x + NW.x) / 2, (N.y + NW.y) / 2)
            Ne = Coordicate2D((N.x + NE.x) / 2, (N.y + NE.y) / 2)
            nW = Coordicate2D((W.x + NW.x) / 2, (W.y + NW.y) / 2)
            nE = Coordicate2D((E.x + NE.x) / 2, (E.y + NE.y) / 2)

            n = Coordicate2D((N.x + P.x) / 2, (N.y + P.y) / 2)
            w = Coordicate2D((W.x + P.x) / 2, (W.y + P.y) / 2)
            e = Coordicate2D((E.x + P.x) / 2, (E.y + P.y) / 2)

            ne = Coordicate2D((Ne.x + e.x) / 2, (Ne.y + e.y) / 2)
            nw = Coordicate2D((Nw.x + w.x) / 2, (Nw.y + w.y) / 2)

            # calculate the area of the cell
            S_nn = calculate_area(ne, e, w, nw)
            S_n = calculate_area(Ne, e, w, Nw)
            S_nnw = calculate_area(n, P, W, nW)
            S_nne = calculate_area(nE, E, P, n)

            # East
            D3 = (
                dy(nw, ne) * (dy(Ne, e) / 4) / S_n
                + dx(nw, ne) * (dx(Ne, e) / 4) / S_n
                + dy(ne, e) * (dy(n, nE) / 4 + 3 * dy(nE, E) / 4 + dy(E, P) / 2) / S_nne
                + dx(ne, e) * (dx(n, nE) / 4 + 3 * dx(nE, E) / 4 + dx(E, P) / 2) / S_nne
            ) / S_nn

            # West
            D_3 = (
                dy(w, nw) * (3 * dy(W, nW) / 4 + dy(nW, n) / 4 + dy(P, W) / 2) / S_nnw
                + dx(w, nw) * (3 * dx(W, nW) / 4 + dx(nW, n) / 4 + dx(P, W) / 2) / S_nnw
                + dy(nw, ne) * (dy(w, Nw) / 4) / S_n
                + dx(nw, ne) * (dx(w, Nw) / 4) / S_n
            ) / S_nn

            # North
            D_1 = (
                dy(w, nw) * (dy(nW, n) / 4 + dy(n, P) / 4) / S_nnw
                + dx(w, nw) * (dx(nW, n) / 4 + dx(n, P) / 4) / S_nnw
                + dy(nw, ne) * (dy(w, Nw) / 4 + dy(Nw, Ne) + dy(Ne, e) / 4) / S_n
                + dx(nw, ne) * (dx(w, Nw) / 4 + dx(Nw, Ne) + dx(Ne, e) / 4) / S_n
                + dy(ne, e) * (dy(P, n) / 4 + dy(n, nE) / 4) / S_nne
                + dx(ne, e) * (dx(P, n) / 4 + dx(n, nE) / 4) / S_nne
            ) / S_nn

            # NW
            D_4 = (
                dy(w, nw) * (dy(W, nW) / 4 + dy(nW, n) / 4) / S_nnw
                + dx(w, nw) * (dx(W, nW) / 4 + dx(nW, n) / 4) / S_nnw
                + dy(nw, ne) * (dy(w, Nw) / 4) / S_n
                + dx(nw, ne) * (dx(w, Nw) / 4) / S_n
            ) / S_nn

            # NE
            D2 = (
                dy(nw, ne) * (dy(Ne, e) / 4) / S_n
                + dx(nw, ne) * (dx(Ne, e) / 4) / S_n
                + dy(ne, e) * (dy(n, nE) / 4 + dy(nE, E) / 4) / S_nne
                + dx(ne, e) * (dx(n, nE) / 4 + dx(nE, E) / 4) / S_nne
            ) / S_nn

            coefficient = 0.0
            if self.boundary[1] == "N":
                coefficient = 0.0
                b = self.q * dist(e, w) / S_nn
            elif self.boundary[1] == "R":
                coefficient = self.alpha
                b = self.alpha * self.Tinf * dist(e, w) / S_nn

            D0 = (
                coefficient * dist(e, w)
                + dy(w, nw) * (dy(nW, n) / 4 + 3 * dy(n, P) / 4 + dy(P, W) / 2) / S_nnw
                + dx(w, nw) * (dx(nW, n) / 4 + 3 * dx(n, P) / 4 + dx(P, W) / 2) / S_nnw
                + dy(nw, ne) * (dy(w, Nw) / 4 + dy(Ne, e) / 4 + dy(e, w)) / S_n
                + dx(nw, ne) * (dx(w, Nw) / 4 + dx(Ne, e) / 4 + dx(e, w)) / S_n
                + dy(ne, e) * (3 * dy(P, n) / 4 + dy(n, nE) / 4 + dy(E, P) / 2) / S_nne
                + dx(ne, e) * (3 * dx(P, n) / 4 + dx(n, nE) / 4 + dx(E, P) / 2) / S_nne
            ) / S_nn

            stencil[self.index(i, j)] = D0
            stencil[self.index(i - 1, j)] = D_1
            stencil[self.index(i, j - 1)] = D_3
            stencil[self.index(i, j + 1)] = D3
            stencil[self.index(i - 1, j - 1)] = D_4
            stencil[self.index(i - 1, j + 1)] = D2

        return stencil, b

    def build_west(self, i, j):
        stencil = np.zeros(self.nm)
        b = 0
        if self.boundary[2] == "D":
            stencil[self.index(i, j)] = 1.0
            b = self.TD[2]

        else:
            # principle node coordinate
            P = Coordicate2D(self.X[i, j], self.Y[i, j])
            N = Coordicate2D(self.X[i - 1, j], self.Y[i - 1, j])
            S = Coordicate2D(self.X[i + 1, j], self.Y[i + 1, j])
            E = Coordicate2D(self.X[i, j + 1], self.Y[i, j + 1])
            NE = Coordicate2D(self.X[i - 1, j + 1], self.Y[i - 1, j + 1])
            SE = Coordicate2D(self.X[i + 1, j + 1], self.Y[i + 1, j + 1])

            # auxiliary node coordinate
            Ne = Coordicate2D((N.x + NE.x) / 2, (N.y + NE.y) / 2)
            nE = Coordicate2D((E.x + NE.x) / 2, (E.y + NE.y) / 2)
            Se = Coordicate2D((S.x + SE.x) / 2, (S.y + SE.y) / 2)
            sE = Coordicate2D((E.x + SE.x) / 2, (E.y + SE.y) / 2)

            n = Coordicate2D((N.x + P.x) / 2, (N.y + P.y) / 2)
            s = Coordicate2D((S.x + P.x) / 2, (S.y + P.y) / 2)
            e = Coordicate2D((E.x + P.x) / 2, (E.y + P.y) / 2)

            ne = Coordicate2D((Ne.x + e.x) / 2, (Ne.y + e.y) / 2)
            se = Coordicate2D((Se.x + e.x) / 2, (Se.y + e.y) / 2)

            # calculate the area of the cell
            S_e = calculate_area(nE, sE, s, n)
            S_ee = calculate_area(ne, se, s, n)
            S_ees = calculate_area(e, Se, S, P)
            S_een = calculate_area(Ne, e, P, N)

            # East
            D3 = (
                dy(s, se) * (dy(Se, e) / 4 + dy(e, P) / 4) / S_ees
                + dx(s, se) * (dx(Se, e) / 4 + dx(e, P) / 4) / S_ees
                + dy(se, ne) * (dy(s, sE) / 4 + dy(sE, nE) + dy(nE, n) / 4) / S_e
                + dx(se, ne) * (dx(s, sE) / 4 + dx(sE, nE) + dx(nE, n) / 4) / S_e
                + dy(ne, n) * (dy(P, e) / 4 + dy(e, Ne) / 4) / S_een
                + dx(ne, n) * (dx(P, e) / 4 + dx(e, Ne) / 4) / S_een
            ) / S_ee

            # North
            D_1 = (
                dy(n, ne) * (3 * dy(N, Ne) / 4 + dy(Ne, e) / 4 + dy(P, N) / 2) / S_een
                + dx(n, ne) * (3 * dx(N, Ne) / 4 + dx(Ne, e) / 4 + dx(P, N) / 2) / S_een
                + dy(se, ne) * (dy(n, nE) / 4) / S_e
                + dx(se, ne) * (dx(n, nE) / 4) / S_e
            ) / S_ee

            # South
            D1 = (
                dy(s, se) * (3 * dy(S, Se) / 4 + dy(Se, e) / 4 + dy(P, S) / 2) / S_ees
                + dx(s, se) * (3 * dx(S, Se) / 4 + dx(Se, e) / 4 + dx(P, S) / 2) / S_ees
                + dy(se, ne) * (dy(s, sE) / 4) / S_e
                + dx(se, ne) * (dx(s, sE) / 4) / S_e
            ) / S_ee

            # SE
            D4 = (
                dy(s, se) * (dy(e, Se) / 4 + dy(Se, S) / 4) / S_ees
                + dx(s, se) * (dx(e, Se) / 4 + dx(Se, S) / 4) / S_ees
                + dy(se, ne) * (dy(s, sE) / 4) / S_e
                + dx(se, ne) * (dx(s, sE) / 4) / S_e
            ) / S_ee

            # NE
            D2 = (
                dy(se, ne) * (dy(nE, n) / 4) / S_e
                + dx(se, ne) * (dx(nE, n) / 4) / S_e
                + dy(n, ne) * (dy(e, Ne) / 4 + dy(Ne, N) / 4) / S_een
                + dx(n, ne) * (dx(e, Ne) / 4 + dx(Ne, N) / 4) / S_een
            ) / S_ee

            coefficient = 0.0
            if self.boundary[2] == "N":
                coefficient = 0.0
                b = self.q * dist(n, s) / S_ee
            elif self.boundary[2] == "R":
                coefficient = self.alpha
                b = self.alpha * self.Tinf * dist(n, s) / S_ee

            D0 = (
                coefficient * dist(n, s)
                + dy(n, ne) * (dy(Ne, e) / 4 + 3 * dy(e, P) / 4 + dy(P, N) / 2) / S_een
                + dx(n, ne) * (dx(Ne, e) / 4 + 3 * dx(e, P) / 4 + dx(P, N) / 2) / S_een
                + dy(se, ne) * (dy(n, nE) / 4 + dy(s, sE) / 4 + dy(n, s)) / S_e
                + dx(se, ne) * (dx(n, nE) / 4 + dx(s, sE) / 4 + dx(n, s)) / S_e
                + dy(s, se) * (3 * dy(P, e) / 4 + dy(e, Se) / 4 + dy(S, P) / 2) / S_ees
                + dx(s, se) * (3 * dx(P, e) / 4 + dx(e, Se) / 4 + dx(S, P) / 2) / S_ees
            ) / S_ee

            stencil[self.index(i, j)] = D0
            stencil[self.index(i - 1, j)] = D_1
            stencil[self.index(i + 1, j)] = D1
            stencil[self.index(i, j + 1)] = D3
            stencil[self.index(i + 1, j + 1)] = D4
            stencil[self.index(i - 1, j + 1)] = D2

        return stencil, b

    def build_east(self, i, j):
        stencil = np.zeros(self.nm)
        b = 0
        if self.boundary[3] == "D":
            stencil[self.index(i, j)] = 1.0
            b = self.TD[3]

        else:
            # principle node coordinate
            P = Coordicate2D(self.X[i, j], self.Y[i, j])
            N = Coordicate2D(self.X[i - 1, j], self.Y[i - 1, j])
            S = Coordicate2D(self.X[i + 1, j], self.Y[i + 1, j])
            W = Coordicate2D(self.X[i, j - 1], self.Y[i, j - 1])
            NW = Coordicate2D(self.X[i - 1, j - 1], self.Y[i - 1, j - 1])
            SW = Coordicate2D(self.X[i + 1, j - 1], self.Y[i + 1, j - 1])

            # auxiliary node coordinate
            Nw = Coordicate2D((N.x + NW.x) / 2, (N.y + NW.y) / 2)
            Sw = Coordicate2D((S.x + SW.x) / 2, (S.y + SW.y) / 2)
            nW = Coordicate2D((W.x + NW.x) / 2, (W.y + NW.y) / 2)
            sW = Coordicate2D((W.x + SW.x) / 2, (W.y + SW.y) / 2)

            n = Coordicate2D((N.x + P.x) / 2, (N.y + P.y) / 2)
            s = Coordicate2D((S.x + P.x) / 2, (S.y + P.y) / 2)
            w = Coordicate2D((W.x + P.x) / 2, (W.y + P.y) / 2)

            nw = Coordicate2D((Nw.x + w.x) / 2, (Nw.y + w.y) / 2)
            sw = Coordicate2D((Sw.x + w.x) / 2, (Sw.y + w.y) / 2)

            # calculate the area of the cell
            S_w = calculate_area(n, s, sW, nW)
            S_ww = calculate_area(n, s, sw, nw)
            S_wws = calculate_area(P, S, Sw, w)
            S_wwn = calculate_area(N, P, w, Nw)

            # West
            D_3 = (
                dy(s, sw) * (dy(Sw, w) / 4 + dy(w, P) / 4) / S_wws
                + dx(s, sw) * (dx(Sw, w) / 4 + dx(w, P) / 4) / S_wws
                + dy(sw, nw) * (dy(s, sW) / 4 + dy(sW, nW) + dy(nW, n) / 4) / S_w
                + dx(sw, nw) * (dx(s, sW) / 4 + dx(sW, nW) + dx(nW, n) / 4) / S_w
                + dy(nw, n) * (dy(P, w) / 4 + dy(w, Nw) / 4) / S_wwn
                + dx(nw, n) * (dx(P, w) / 4 + dx(w, Nw) / 4) / S_wwn
            ) / S_ww

            # North
            D_1 = (
                dy(n, nw) * (3 * dy(N, Nw) / 4 + dy(Nw, w) / 4 + dy(P, N) / 2) / S_wwn
                + dx(n, nw) * (3 * dx(N, Nw) / 4 + dx(Nw, w) / 4 + dx(P, N) / 2) / S_wwn
                + dy(sw, nw) * (dy(n, nW) / 4) / S_w
                + dx(sw, nw) * (dx(n, nW) / 4) / S_w
            ) / S_ww

            # South
            D1 = (
                dy(s, sw) * (3 * dy(S, Sw) / 4 + dy(Sw, w) / 4 + dy(P, S) / 2) / S_wws
                + dx(s, sw) * (3 * dx(S, Sw) / 4 + dx(Sw, w) / 4 + dx(P, S) / 2) / S_wws
                + dy(sw, nw) * (dy(s, sW) / 4) / S_w
                + dx(sw, nw) * (dx(s, sW) / 4) / S_w
            ) / S_ww

            # SW
            D_2 = (
                dy(s, sw) * (dy(w, Sw) / 4 + dy(Sw, S) / 4) / S_wws
                + dx(s, sw) * (dx(w, Sw) / 4 + dx(Sw, S) / 4) / S_wws
                + dy(sw, nw) * (dy(s, sW) / 4) / S_w
                + dx(sw, nw) * (dx(s, sW) / 4) / S_w
            ) / S_ww

            # NW
            D_4 = (
                dy(sw, nw) * (dy(nW, n) / 4) / S_w
                + dx(sw, nw) * (dx(nW, n) / 4) / S_w
                + dy(n, nw) * (dy(w, Nw) / 4 + dy(Nw, N) / 4) / S_wwn
                + dx(n, nw) * (dx(w, Nw) / 4 + dx(Nw, N) / 4) / S_wwn
            ) / S_ww

            coefficient = 0.0
            if self.boundary[3] == "N":
                coefficient = 0.0
                b = self.q * dist(n, s) / S_ww
            elif self.boundary[3] == "R":
                coefficient = self.alpha
                b = self.alpha * self.Tinf * dist(n, s) / S_ww

            D0 = (
                coefficient * dist(n, s)
                + dy(n, nw) * (dy(Nw, w) / 4 + 3 * dy(w, P) / 4 + dy(P, N) / 2) / S_wwn
                + dx(n, nw) * (dx(Nw, w) / 4 + 3 * dx(w, P) / 4 + dx(P, N) / 2) / S_wwn
                + dy(sw, nw) * (dy(n, nW) / 4 + dy(s, sW) / 4 + dy(n, s)) / S_w
                + dx(sw, nw) * (dx(n, nW) / 4 + dx(s, sW) / 4 + dx(n, s)) / S_w
                + dy(s, sw) * (3 * dy(P, w) / 4 + dy(w, Sw) / 4 + dy(S, P) / 2) / S_wws
                + dx(s, sw) * (3 * dx(P, w) / 4 + dx(w, Sw) / 4 + dx(S, P) / 2) / S_wws
            ) / S_ww

            stencil[self.index(i, j)] = D0
            stencil[self.index(i - 1, j)] = D_1
            stencil[self.index(i + 1, j)] = D1
            stencil[self.index(i, j - 1)] = D_3
            stencil[self.index(i - 1, j - 1)] = D_4
            stencil[self.index(i + 1, j - 1)] = D_2

        return stencil, b

    def build_NW(self, i, j):
        stencil = np.zeros(self.nm)
        b = 0
        if self.boundary[0] == "D" and self.boundary[2] == "D":
            stencil[self.index(i, j)] = 1.0
            b = (self.TD[0] + self.TD[2]) / 2
        elif self.boundary[0] == "D":
            stencil[self.index(i, j)] = 1.0
            b = self.TD[0]
        elif self.boundary[2] == "D":
            stencil[self.index(i, j)] = 1.0
            b = self.TD[2]
        else:
            # principle node coordinate
            P = Coordicate2D(self.X[i, j], self.Y[i, j])
            S = Coordicate2D(self.X[i + 1, j], self.Y[i + 1, j])
            E = Coordicate2D(self.X[i, j + 1], self.Y[i, j + 1])
            SE = Coordicate2D(self.X[i + 1, j + 1], self.Y[i + 1, j + 1])

            # auxiliary node coordinate
            Se = Coordicate2D((S.x + SE.x) / 2, (S.y + SE.y) / 2)
            sE = Coordicate2D((E.x + SE.x) / 2, (E.y + SE.y) / 2)

            s = Coordicate2D((S.x + P.x) / 2, (S.y + P.y) / 2)
            e = Coordicate2D((E.x + P.x) / 2, (E.y + P.y) / 2)

            se = Coordicate2D((Se.x + e.x) / 2, (Se.y + e.y) / 2)

            # calculate the area of the cell
            S_sse = calculate_area(E, sE, s, P)
            S_ees = calculate_area(e, Se, S, P)
            S_se = calculate_area(E, SE, S, P)
            S_sese = calculate_area(e, se, s, P)

            # East
            D3 = (
                dy(s, sE) * (dy(SE, E) / 4) / S_se
                + dx(s, sE) * (dx(SE, E) / 4) / S_se
                + dy(se, e) * (dy(s, sE) / 4 + 3 * dy(sE, E) / 4 + dy(E, P) / 2) / S_sse
                + dx(se, e) * (dx(s, sE) / 4 + 3 * dx(sE, E) / 4 + dx(E, P) / 2) / S_sse
            ) / (2 * S_sese)

            # South
            D1 = (
                dy(e, Se) * (dy(SE, S) / 4) / S_se
                + dx(e, Se) * (dx(SE, S) / 4) / S_se
                + dy(se, s) * (dy(e, Se) / 4 + 3 * dy(Se, S) / 4 + dy(S, P) / 2) / S_ees
                + dx(se, s) * (dx(e, Se) / 4 + 3 * dx(Se, S) / 4 + dx(S, P) / 2) / S_ees
            ) / (2 * S_sese)

            # SE
            D4 = (
                (dy(se, Se) + dy(se, Se))
                * (dy(P, E) / 4 + dy(E, sE) + dy(Se, S) + dy(P, S) / 4)
                / S_se
                + (dy(se, Se) + dy(se, Se))
                * (dx(P, E) / 4 + dx(E, sE) + dx(Se, S) + dx(P, S) / 4)
                / S_se
                + dy(se, e) * (dy(s, sE) / 4 + dy(s, P) / 4) / S_sse
                + dx(se, e) * (dx(s, sE) / 4 + dx(s, P) / 4) / S_sse
                + dy(se, s) * (dy(e, Se) / 4 + dy(e, P) / 4) / S_ees
                + dx(se, s) * (dx(e, Se) / 4 + dx(e, P) / 4) / S_ees
            ) / (2 * S_sese)

            coefficient = 0.0
            if self.boundary[0] == "N" or self.boundary[2] == "N":
                coefficient = 0.0
                b = self.q * (dist(P, e) + dist(P, s)) / S_sese
            elif self.boundary[0] == "R" and self.boundary[2] == "R":
                coefficient = self.alpha
                b = self.alpha * self.Tinf * (dist(P, e) + dist(P, s)) / S_sese

            D0 = (
                coefficient * (dist(P, e) + dist(P, s))
                + dy(se, s) * (dy(e, Se) / 4 + 6 * dy(Se, S) / 4 + dy(P, S) / 2) / S_ees
                + dx(se, s) * (dx(e, Se) / 4 + 6 * dx(Se, S) / 4 + dx(P, S) / 2) / S_ees
                + (dy(se, sE) + dy(se, Se))
                * (dy(e, Se) / 4 + dy(s, sE) / 4 + dy(P, e) + dy(P, s))
                / S_se
                + (dx(se, sE) + dx(se, Se))
                * (dx(e, Se) / 4 + dx(s, sE) / 4 + dx(P, e) + dx(P, s))
                / S_se
                + dy(se, e) * (6 * dy(E, sE) / 4 + dy(s, sE) / 4 + dy(E, P) / 2) / S_sse
                + dx(se, e) * (6 * dx(E, sE) / 4 + dx(s, sE) / 4 + dx(E, P) / 2) / S_sse
            ) / (2 * S_sese)

            stencil[self.index(i, j)] = D0
            stencil[self.index(i + 1, j)] = D1
            stencil[self.index(i, j + 1)] = D3
            stencil[self.index(i + 1, j + 1)] = D4

        return stencil, b

    def build_NE(self, i, j):
        stencil = np.zeros(self.nm)
        b = 0
        if self.boundary[0] == "D" and self.boundary[3] == "D":
            stencil[self.index(i, j)] = 1.0
            b = (self.TD[0] + self.TD[3]) / 2
        elif self.boundary[0] == "D":
            stencil[self.index(i, j)] = 1.0
            b = self.TD[0]
        elif self.boundary[3] == "D":
            stencil[self.index(i, j)] = 1.0
            b = self.TD[3]
        else:
            # principle node coordinate
            P = Coordicate2D(self.X[i, j], self.Y[i, j])
            S = Coordicate2D(self.X[i + 1, j], self.Y[i + 1, j])
            W = Coordicate2D(self.X[i, j - 1], self.Y[i, j - 1])
            SW = Coordicate2D(self.X[i + 1, j - 1], self.Y[i + 1, j - 1])

            # auxiliary node coordinate
            Sw = Coordicate2D((S.x + SW.x) / 2, (S.y + SW.y) / 2)
            sW = Coordicate2D((W.x + SW.x) / 2, (W.y + SW.y) / 2)

            s = Coordicate2D((S.x + P.x) / 2, (S.y + P.y) / 2)
            w = Coordicate2D((W.x + P.x) / 2, (W.y + P.y) / 2)

            sw = Coordicate2D((Sw.x + w.x) / 2, (Sw.y + w.y) / 2)

            # calculate the area of the cell
            S_ssw = calculate_area(P, s, sW, W)
            S_wws = calculate_area(P, S, Sw, w)
            S_sw = calculate_area(P, S, SW, W)
            S_swsw = calculate_area(P, s, sw, w)

            # West
            D_3 = (
                dy(s, sW) * (dy(SW, W) / 4) / S_sw
                + dx(s, sW) * (dx(SW, W) / 4) / S_sw
                + dy(sw, w) * (dy(s, sW) / 4 + 3 * dy(sW, W) / 4 + dy(W, P) / 2) / S_ssw
                + dx(sw, w) * (dx(s, sW) / 4 + 3 * dx(sW, W) / 4 + dx(W, P) / 2) / S_ssw
            ) / (2 * S_swsw)

            # South
            D1 = (
                dy(w, Sw) * (dy(SW, S) / 4) / S_sw
                + dx(w, Sw) * (dx(SW, S) / 4) / S_sw
                + dy(sw, s) * (dy(w, Sw) / 4 + 3 * dy(Sw, S) / 4 + dy(S, P) / 2) / S_wws
                + dx(sw, s) * (dx(w, Sw) / 4 + 3 * dx(Sw, S) / 4 + dx(S, P) / 2) / S_wws
            ) / (2 * S_swsw)

            # SW
            D_2 = (
                (dy(sw, Sw) + dy(sw, Sw))
                * (dy(P, W) / 4 + dy(W, sW) + dy(Sw, S) + dy(P, S) / 4)
                / S_sw
                + (dy(sw, Sw) + dy(sw, Sw))
                * (dx(P, W) / 4 + dx(W, sW) + dx(Sw, S) + dx(P, S) / 4)
                / S_sw
                + dy(sw, w) * (dy(s, sW) / 4 + dy(s, P) / 4) / S_ssw
                + dx(sw, w) * (dx(s, sW) / 4 + dx(s, P) / 4) / S_ssw
                + dy(sw, s) * (dy(w, Sw) / 4 + dy(w, P) / 4) / S_wws
                + dx(sw, s) * (dx(w, Sw) / 4 + dx(w, P) / 4) / S_wws
            ) / (2 * S_swsw)

            coefficient = 0.0
            if self.boundary[0] == "N" or self.boundary[3] == "N":
                coefficient = 0.0
                b = self.q * (dist(P, w) + dist(P, s)) / S_swsw
            elif self.boundary[0] == "R" and self.boundary[3] == "R":
                coefficient = self.alpha
                b = self.alpha * self.Tinf * (dist(P, w) + dist(P, s)) / S_swsw

            D0 = (
                coefficient * (dist(P, w) + dist(P, s))
                + dy(sw, s) * (dy(w, Sw) / 4 + 6 * dy(Sw, S) / 4 + dy(P, S) / 2) / S_wws
                + dx(sw, s) * (dx(w, Sw) / 4 + 6 * dx(Sw, S) / 4 + dx(P, S) / 2) / S_wws
                + (dy(sw, sW) + dy(sw, Sw))
                * (dy(w, Sw) / 4 + dy(s, sW) / 4 + dy(P, w) + dy(P, s))
                / S_sw
                + (dx(sw, sW) + dx(sw, Sw))
                * (dx(w, Sw) / 4 + dx(s, sW) / 4 + dx(P, w) + dx(P, s))
                / S_sw
                + dy(sw, w) * (6 * dy(W, sW) / 4 + dy(s, sW) / 4 + dy(W, P) / 2) / S_ssw
                + dx(sw, w) * (6 * dx(W, sW) / 4 + dx(s, sW) / 4 + dx(W, P) / 2) / S_ssw
            ) / (2 * S_swsw)

            stencil[self.index(i, j)] = D0
            stencil[self.index(i + 1, j)] = D1
            stencil[self.index(i, j - 1)] = D_3
            stencil[self.index(i + 1, j - 1)] = D_2

        return stencil, b

    def build_SW(self, i, j):
        stencil = np.zeros(self.nm)
        b = 0
        if self.boundary[1] == "D" and self.boundary[2] == "D":
            stencil[self.index(i, j)] = 1.0
            b = (self.TD[1] + self.TD[2]) / 2
        elif self.boundary[1] == "D":
            stencil[self.index(i, j)] = 1.0
            b = self.TD[1]
        elif self.boundary[2] == "D":
            stencil[self.index(i, j)] = 1.0
            b = self.TD[2]
        else:
            # principle node coordinate
            P = Coordicate2D(self.X[i, j], self.Y[i, j])
            N = Coordicate2D(self.X[i - 1, j], self.Y[i - 1, j])
            E = Coordicate2D(self.X[i, j + 1], self.Y[i, j + 1])
            NE = Coordicate2D(self.X[i - 1, j + 1], self.Y[i - 1, j + 1])

            # auxiliary node coordinate
            Ne = Coordicate2D((N.x + NE.x) / 2, (N.y + NE.y) / 2)
            nE = Coordicate2D((N.x + NE.x) / 2, (E.y + NE.y) / 2)

            n = Coordicate2D((N.x + P.x) / 2, (N.y + P.y) / 2)
            e = Coordicate2D((E.x + P.x) / 2, (E.y + P.y) / 2)

            ne = Coordicate2D((Ne.x + e.x) / 2, (Ne.y + e.y) / 2)

            # calculate the area of the cell
            S_nne = calculate_area(nE, E, P, n)
            S_een = calculate_area(Ne, e, P, N)
            S_ne = calculate_area(NE, E, P, N)
            S_nene = calculate_area(ne, e, P, n)

            # East
            D3 = (
                dy(n, nE) * (dy(NE, E) / 4) / S_ne
                + dx(n, nE) * (dx(NE, E) / 4) / S_ne
                + dy(ne, e) * (dy(n, nE) / 4 + 3 * dy(nE, E) / 4 + dy(E, P) / 2) / S_nne
                + dx(ne, e) * (dx(n, nE) / 4 + 3 * dx(nE, E) / 4 + dx(E, P) / 2) / S_nne
            ) / (2 * S_nene)

            # North
            D_1 = (
                dy(e, Ne) * (dy(NE, N) / 4) / S_ne
                + dx(e, Ne) * (dx(NE, N) / 4) / S_ne
                + dy(ne, n) * (dy(e, Ne) / 4 + 3 * dy(Ne, N) / 4 + dy(N, P) / 2) / S_een
                + dx(ne, n) * (dx(e, Ne) / 4 + 3 * dx(Ne, N) / 4 + dx(N, P) / 2) / S_een
            ) / (2 * S_nene)

            # NE
            D2 = (
                (dy(ne, Ne) + dy(ne, Ne))
                * (dy(P, E) / 4 + dy(E, nE) + dy(Ne, N) + dy(P, N) / 4)
                / S_ne
                + (dy(ne, Ne) + dy(ne, Ne))
                * (dx(P, E) / 4 + dx(E, nE) + dx(Ne, N) + dx(P, N) / 4)
                / S_ne
                + dy(ne, e) * (dy(n, nE) / 4 + dy(n, P) / 4) / S_nne
                + dx(ne, e) * (dx(n, nE) / 4 + dx(n, P) / 4) / S_nne
                + dy(ne, n) * (dy(e, Ne) / 4 + dy(e, P) / 4) / S_een
                + dx(ne, n) * (dx(e, Ne) / 4 + dx(e, P) / 4) / S_een
            ) / (2 * S_nene)

            coefficient = 0.0
            if self.boundary[1] == "N" or self.boundary[2] == "N":
                coefficient = 0.0
                b = self.q * (dist(P, e) + dist(P, n)) / S_nene
            elif self.boundary[1] == "R" and self.boundary[2] == "R":
                coefficient = self.alpha
                b = self.alpha * self.Tinf * (dist(P, e) + dist(P, n)) / S_nene

            D0 = (
                coefficient * (dist(P, e) + dist(P, n))
                + dy(ne, n) * (dy(e, Ne) / 4 + 6 * dy(Ne, N) / 4 + dy(P, N) / 2) / S_een
                + dx(ne, n) * (dx(e, Ne) / 4 + 6 * dx(Ne, N) / 4 + dx(P, N) / 2) / S_een
                + (dy(ne, nE) + dy(ne, Ne))
                * (dy(e, Ne) / 4 + dy(n, nE) / 4 + dy(P, e) + dy(P, n))
                / S_ne
                + (dx(ne, nE) + dx(ne, Ne))
                * (dx(e, Ne) / 4 + dx(n, nE) / 4 + dx(P, e) + dx(P, n))
                / S_ne
                + dy(ne, e) * (6 * dy(E, nE) / 4 + dy(n, nE) / 4 + dy(E, P) / 2) / S_nne
                + dx(ne, e) * (6 * dx(E, nE) / 4 + dx(n, nE) / 4 + dx(E, P) / 2) / S_nne
            ) / (2 * S_nene)

            stencil[self.index(i, j)] = D0
            stencil[self.index(i - 1, j)] = D_1
            stencil[self.index(i, j + 1)] = D3
            stencil[self.index(i - 1, j + 1)] = D2

        return stencil, b

    def build_SE(self, i, j):
        stencil = np.zeros(self.nm)
        b = 0
        if self.boundary[1] == "D" and self.boundary[3] == "D":
            stencil[self.index(i, j)] = 1.0
            b = (self.TD[1] + self.TD[3]) / 2
        elif self.boundary[1] == "D":
            stencil[self.index(i, j)] = 1.0
            b = self.TD[1]
        elif self.boundary[3] == "D":
            stencil[self.index(i, j)] = 1.0
            b = self.TD[3]
        else:
            # principle node coordinate
            P = Coordicate2D(self.X[i, j], self.Y[i, j])
            N = Coordicate2D(self.X[i - 1, j], self.Y[i - 1, j])
            W = Coordicate2D(self.X[i, j - 1], self.Y[i, j - 1])
            NW = Coordicate2D(self.X[i - 1, j - 1], self.Y[i - 1, j - 1])

            # auxiliary node coordinate
            Nw = Coordicate2D((N.x + NW.x) / 2, (N.y + NW.y) / 2)
            nW = Coordicate2D((N.x + NW.x) / 2, (W.y + NW.y) / 2)

            n = Coordicate2D((N.x + P.x) / 2, (N.y + P.y) / 2)
            w = Coordicate2D((W.x + P.x) / 2, (W.y + P.y) / 2)

            nw = Coordicate2D((Nw.x + w.x) / 2, (Nw.y + w.y) / 2)

            # calculate the area of the cell
            S_nnw = calculate_area(n, P, W, nW)
            S_wwn = calculate_area(N, P, w, Nw)
            S_nw = calculate_area(N, P, W, NW)
            S_nwnw = calculate_area(n, P, w, nw)

            # West
            D_3 = (
                dy(n, nW) * (dy(NW, W) / 4) / S_nw
                + dx(n, nW) * (dx(NW, W) / 4) / S_nw
                + dy(nw, w) * (dy(n, nW) / 4 + 3 * dy(nW, W) / 4 + dy(W, P) / 2) / S_nnw
                + dx(nw, w) * (dx(n, nW) / 4 + 3 * dx(nW, W) / 4 + dx(W, P) / 2) / S_nnw
            ) / (2 * S_nwnw)

            # North
            D_1 = (
                dy(w, Nw) * (dy(NW, N) / 4) / S_nw
                + dx(w, Nw) * (dx(NW, N) / 4) / S_nw
                + dy(nw, n) * (dy(w, Nw) / 4 + 3 * dy(Nw, N) / 4 + dy(N, P) / 2) / S_wwn
                + dx(nw, n) * (dx(w, Nw) / 4 + 3 * dx(Nw, N) / 4 + dx(N, P) / 2) / S_wwn
            ) / (2 * S_nwnw)

            # NE
            D_4 = (
                (dy(nw, Nw) + dy(nw, Nw))
                * (dy(P, W) / 4 + dy(W, nW) + dy(Nw, N) + dy(P, N) / 4)
                / S_nw
                + (dy(nw, Nw) + dy(nw, Nw))
                * (dx(P, W) / 4 + dx(W, nW) + dx(Nw, N) + dx(P, N) / 4)
                / S_nw
                + dy(nw, w) * (dy(n, nW) / 4 + dy(n, P) / 4) / S_nnw
                + dx(nw, w) * (dx(n, nW) / 4 + dx(n, P) / 4) / S_nnw
                + dy(nw, n) * (dy(w, Nw) / 4 + dy(w, P) / 4) / S_wwn
                + dx(nw, n) * (dx(w, Nw) / 4 + dx(w, P) / 4) / S_wwn
            ) / (2 * S_nwnw)

            coefficient = 0.0
            if self.boundary[1] == "N" or self.boundary[3] == "N":
                coefficient = 0.0
                b = self.q * (dist(P, w) + dist(P, n)) / S_nwnw
            elif self.boundary[1] == "R" and self.boundary[3] == "R":
                coefficient = self.alpha
                b = self.alpha * self.Tinf * (dist(P, w) + dist(P, n)) / S_nwnw

            D0 = (
                coefficient * (dist(P, w) + dist(P, n))
                + dy(nw, n) * (dy(w, Nw) / 4 + 6 * dy(Nw, N) / 4 + dy(P, N) / 2) / S_wwn
                + dx(nw, n) * (dx(w, Nw) / 4 + 6 * dx(Nw, N) / 4 + dx(P, N) / 2) / S_wwn
                + (dy(nw, nW) + dy(nw, Nw))
                * (dy(w, Nw) / 4 + dy(n, nW) / 4 + dy(P, w) + dy(P, n))
                / S_nw
                + (dx(nw, nW) + dx(nw, Nw))
                * (dx(w, Nw) / 4 + dx(n, nW) / 4 + dx(P, w) + dx(P, n))
                / S_nw
                + dy(nw, w) * (6 * dy(W, nW) / 4 + dy(n, nW) / 4 + dy(W, P) / 2) / S_nnw
                + dx(nw, w) * (6 * dx(W, nW) / 4 + dx(n, nW) / 4 + dx(W, P) / 2) / S_nnw
            ) / (2 * S_nwnw)

            stencil[self.index(i, j)] = D0
            stencil[self.index(i - 1, j)] = D_1
            stencil[self.index(i, j - 1)] = D_3
            stencil[self.index(i - 1, j - 1)] = D_4

        return stencil, b

    def gen_init_guess(self, conjugate=False):
        init = np.zeros((self.n, self.m))
        init[0, :] = self.TD[0]  # Top boundary
        init[-1, :] = self.TD[1]  # Bottom boundary
        init[:, 0] = self.TD[2]  # Left boundary
        init[:, -1] = self.TD[3]  # Right boundary
        if conjugate:
            init[self.n // 2, :] = self.mid_TD

        # Average the boundaries to fill the initial guess
        ave_TD = np.mean(np.array([np.mean(x) for x in self.TD]))
        init[1:-1, 1:-1] = ave_TD
        T_guess = init.flatten()

        return T_guess

    def get_solver(self, solver_type, matrix_type):
        match (solver_type, matrix_type):
            case ("direct", "sparse"):
                return sparse.linalg.spsolve

            case ("direct", "dense"):
                return np.linalg.solve

            case ("jacobi", "sparse"):

                def jacobi(A, B, T_guess, tolerance, iter_limit, relaxation=None):
                    D = A.diagonal()
                    C_inv = np.ravel(1 / D)
                    T = T_guess.copy()
                    i = 0
                    residual = C_inv * np.ravel(B - A @ T)
                    while (
                        np.linalg.norm(residual) / np.linalg.norm(B) > tolerance
                    ) and (i <= iter_limit):
                        residual = C_inv * np.ravel(B - A @ T)
                        T = T + residual
                        i += 1
                    self.iters = i
                    return T

                return jacobi

            case ("gauss", "sparse"):

                def gauss(A, B, T_guess, tolerance, iter_limit, relaxation=None):
                    D = A.diagonal()
                    D_inv = 1 / D
                    E = sparse.tril(A, -1, format="csr")
                    F = sparse.triu(A, 1, format="csr")
                    C_inv = np.linalg.inv(np.diag(D) + E)
                    T = T_guess.copy()
                    i = 0
                    residual = D_inv * (B - A @ T)
                    while (
                        np.linalg.norm(residual) / np.linalg.norm(B) > tolerance
                    ) and (i <= iter_limit):
                        for i in range(self.nm):
                            T[i] = D_inv[i] * (B[i] - E[i, :] @ T - F[i, :] @ T)[0]
                        residual = C_inv @ (B - A @ T)
                        i += 1
                    self.iters = i
                    return T

                return gauss

            case ("sor", "sparse"):

                def sor(A, B, T_guess, tolerance, iter_limit, relaxation):
                    assert relaxation is not None, "Provide a relaxation factor"
                    assert 0 < relaxation < 2, "Relaxation out of bounds: (0, 2)"
                    D = A.diagonal()
                    D_inv = 1 / D
                    E = sparse.tril(A, -1, format="csr")
                    F = sparse.triu(A, 1, format="csr")
                    T = T_guess.copy()
                    C_inv = (1 / relaxation) * np.linalg.inv(
                        np.diag(D) + (relaxation * E)
                    )
                    i = 0
                    residual = D_inv * (B - A @ T)
                    while (
                        np.linalg.norm(residual) / np.linalg.norm(B) > tolerance
                    ) and (i <= iter_limit):
                        for i in range(self.nm):
                            T[i] = (1 - relaxation) * T[i] + relaxation * D_inv[i] * (
                                B[i] - E[i, :] @ T - F[i, :] @ T
                            )[0]
                        residual = C_inv @ (B - A @ T)
                        i += 1
                    self.iters = i
                    return T

                return sor

            case _:
                raise AssertionError("Improper Solver and Matrix type passed")

    def solve(
        self,
        problem_type,
        solver_type,
        matrix_type,
        theta=0,
        t_steps=100,
        tolerance=0.0001,
        iter_limit=1000,
        relaxation=None,
        update=True,
        conjugate=True,
        greens=None,
        green_limit=None,
    ):
        solver = self.get_solver(solver_type=solver_type, matrix_type=matrix_type)
        match problem_type.lower():
            case "steady":
                if update:
                    self.update_AB(matrix_type)
                self.timeline = None
                if solver_type == "direct":
                    T = solver(self.A, self.B)
                else:
                    T_guess = self.gen_init_guess()
                    T = solver(
                        self.A, self.B, T_guess, tolerance, iter_limit, relaxation
                    )
                T_map = T.reshape(self.n, self.m)
                T_map = T_map[1:-1, 1:-1]
                return T_map

            case "unsteady":
                assert (
                    0 <= theta <= 1
                ), "Specify a theta: 0: Explicit, 0.5: Crank-Nicolson, 1: Implicit"
                T_history = np.zeros((t_steps, self.n, self.m))
                if update:
                    self.update_AB("dense")  # we do this first to get the timesteps
                dt, self.timeline = self.get_timestep(t_steps, theta)
                if matrix_type == "sparse" and update:
                    self.update_AB("sparse")
                if theta == 0:
                    for t in range(1, t_steps - 1):
                        for i, b in enumerate(self.boundary):
                            _td = self.TD[i]
                            match i:
                                case 0:
                                    T_history[t, 0, :] = _td
                                case 1:
                                    T_history[t, -1, :] = _td
                                case 2:
                                    T_history[t, :, 0] = _td
                                case 3:
                                    T_history[t, :, -1] = _td
                        for i in range(1, self.n - 1):
                            for j in range(1, self.m - 1):
                                idx = self.index(i, j)
                                T_history[t + 1, i, j] = T_history[t, i, j] - dt * (
                                    self.A[idx] @ T_history[t].flatten() - self.B[idx]
                                )
                                T_copy = T_history[:, 1:-1, 1:-1].copy()

                else:
                    for t in range(1, t_steps - 1):
                        for i, b in enumerate(self.boundary):
                            _td = self.TD[i]
                            match i:
                                case 0:
                                    T_history[t, 0, :] = _td
                                case 1:
                                    T_history[t, -1, :] = _td
                                case 2:
                                    T_history[t, :, 0] = _td
                                case 3:
                                    T_history[t, :, -1] = _td
                        if conjugate:
                            T_history[t, self.n // 2, :] = self.mid_TD
                            self.B[self.n // 2] = 1.0

                        B_star = (
                            (np.eye(self.A.shape[0]) + (dt * (1 - theta)) * self.A)
                            @ T_history[t].flatten()
                        ) + dt * self.B
                        B_star = np.ravel(B_star)  # ensure 1D vector
                        A_star = np.eye(self.A.shape[0]) - dt * theta * self.A
                        T_guess = self.gen_init_guess(conjugate)
                        if solver_type == "direct":
                            T = solver(A_star, B_star)
                        else:
                            T = solver(
                                A_star,
                                B_star,
                                T_guess,
                                tolerance,
                                iter_limit,
                                relaxation,
                            )

                        T_history[t + 1] = T.reshape(self.n, self.m)
                        if (green_limit is None) or (t < green_limit):
                            T_history[t + 1] = combine_finite_diff_and_green(
                                T_history[t + 1], greens, fill=True
                            )

                        # get the current temperature from bottom north for top south dirichlet bc
                        if conjugate:
                            self.mid_TD = T_history[t + 1][self.n // 2 + 1]

                    T_copy = T_history[:, 1:-1, 1:-1].copy()
                return T_copy

    def get_timestep(self, t_steps, theta):
        min_timestep = 1
        if theta == 1:
            min_timestep = 0.001  # for theta = 1, default 0.1, square 0.01
        elif theta == 0.5:
            min_timestep = 0.1
        else:
            for i in range(1, self.n - 1):
                for j in range(1, self.m - 1):
                    idx = self.index(i, j)
                    # dX, dY Values
                    dX = abs(
                        self.A[idx][self.index(i, j - 1)]
                        - self.A[idx][self.index(i, j + 1)]
                    )
                    dY = abs(
                        self.A[idx][self.index(i + 1, j)]
                        - self.A[idx][self.index(i - 1, j)]
                    )

                    if (dX < 1e-10) and (dY < 1e-10):
                        continue
                    # change multiplier to get rough dt
                    multiplier = 0.95
                    dt = multiplier * ((dX**2 * dY**2) / (dX**2 + dY**2)) / (2 * self.D)
                    if theta < 0.5:
                        assert dt < 0.99 * (
                            (dX**2 * dY**2) / (dX**2 + dY**2)
                        ) / (2 * self.D * (1 - 2 * theta))

                    if dt < min_timestep:
                        min_timestep = dt

        # min_timestep = 0.002690783839048107 #for theta = 0
        timeline = np.arange(0, t_steps * min_timestep, min_timestep)
        timeline = np.asarray(timeline)
        timeline *= 1000
        return min_timestep, timeline

    def inspect_matrix(self):
        # matrix visualisation for debugging

        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(14, 7, forward=True)
        ax1 = axes[0]
        ax2 = axes[1]
        # ax1.matshow(self.A.toarray(), cmap=plt.cm.Blues)
        ax1.matshow(self.A.toarray(), cmap=plt.cm.Greys)
        ax1.set_title(f"Matrix: {self.A.shape}")
        # np.savetxt("A.csv", self.A, delimiter=",")
        ax2.set_title(f"Matrix: {self.B.shape}")
        ax2.matshow(np.diag(self.B), cmap=plt.cm.Greys)
        plt.show()


class Diffusive(Heat2D_FVM):
    def __init__(self, N, M, X, Y, D, dX, boundary=[], TD=[], q=0, alpha=0, Tinf=0):
        super().__init__(N, M, X, Y, D, dX, boundary, TD, q, alpha, Tinf)

    def update_AB(self, matrix_type, use_inner_at=False):
        self.A = np.zeros((self.nm, self.nm))
        self.B = np.zeros(self.nm)
        A = np.zeros((self.nm, self.nm))
        for i in range(self.n):
            for j in range(self.m):
                stencil, b = self.second_order_stencil(i, j, use_inner_at)
                idx = self.index(i, j)
                A[idx] = stencil
                self.B[idx] = b
        A = self.D * A

        match matrix_type:
            case "sparse":
                if isinstance(A, np.ndarray):
                    self.A = sparse.csr_matrix(A)

            case "dense":
                self.A = A


class ConvectiveDiffusive(Heat2D_FVM):
    def __init__(self, N, M, X, Y, D, dX, Vx, boundary=[], TD=[], q=0, alpha=0, Tinf=0):
        super().__init__(N, M, X, Y, D, dX, boundary, TD, q, alpha, Tinf)
        self.Vx = Vx
        self.conjugate = False
        self.mid_TD = 0

    def first_order_stencil(self, i, j):
        # we only are about the N, N, D, N case for now...
        if i == 0 and j != 0 and j != self.m - 1:
            # north neumann
            if self.boundary[0] == "D":
                return self.get_dirichlet(i, j, self.TD[0])

            else:
                return self.get_north(i, j)

        elif i == self.n - 1 and j != 0 and j != self.m - 1:
            # south neumann
            if self.boundary[1] == "D":
                return self.get_dirichlet(i, j, self.TD[1])

            else:
                return self.get_south(i, j)

        elif j == 0 and i != 0 and i != self.n - 1:
            # west dirichlet
            return self.get_dirichlet(i, j, self.TD[2])
        elif j == self.m - 1 and i != 0 and i != self.n - 1:
            # east Neumann
            if self.boundary[3] == "N":
                return self.get_east(i, j)
            elif self.boundary[3] == "D":
                return self.get_dirichlet(i, j, self.TD[3])
        elif i == 0 and j == 0:
            # NW neumann
            return self.get_NW(i, j)
        elif i == 0 and j == self.m - 1:
            # NE Neumann
            return self.get_NE(i, j)
        elif i == self.n - 1 and j == 0:
            # SW neumann
            return self.get_SW(i, j)
        elif i == self.n - 1 and j == self.m - 1:
            return self.get_SE(i, j)
        else:
            return self.get_inner(i, j)

    def get_inner(self, i, j):
        stencil = np.zeros(self.nm)

        # principle node coordinate
        P = Coordicate2D(self.X[i, j], self.Y[i, j])
        N = Coordicate2D(self.X[i - 1, j], self.Y[i - 1, j])
        S = Coordicate2D(self.X[i + 1, j], self.Y[i + 1, j])
        W = Coordicate2D(self.X[i, j - 1], self.Y[i, j - 1])
        E = Coordicate2D(self.X[i, j + 1], self.Y[i, j + 1])
        NW = Coordicate2D(self.X[i - 1, j - 1], self.Y[i - 1, j - 1])
        NE = Coordicate2D(self.X[i - 1, j + 1], self.Y[i - 1, j + 1])
        SW = Coordicate2D(self.X[i + 1, j - 1], self.Y[i + 1, j - 1])
        SE = Coordicate2D(self.X[i + 1, j + 1], self.Y[i + 1, j + 1])

        # auxiliary node coordinate
        Nw = Coordicate2D((N.x + NW.x) / 2, (N.y + NW.y) / 2)
        Ne = Coordicate2D((N.x + NE.x) / 2, (N.y + NE.y) / 2)
        Sw = Coordicate2D((S.x + SW.x) / 2, (S.y + SW.y) / 2)
        Se = Coordicate2D((S.x + SE.x) / 2, (S.y + SE.y) / 2)
        nW = Coordicate2D((W.x + NW.x) / 2, (W.y + NW.y) / 2)
        nE = Coordicate2D((E.x + NE.x) / 2, (E.y + NE.y) / 2)
        sW = Coordicate2D((W.x + SW.x) / 2, (W.y + SW.y) / 2)
        sE = Coordicate2D((E.x + SE.x) / 2, (E.y + SE.y) / 2)

        n = Coordicate2D((N.x + P.x) / 2, (N.y + P.y) / 2)
        s = Coordicate2D((S.x + P.x) / 2, (S.y + P.y) / 2)
        w = Coordicate2D((W.x + P.x) / 2, (W.y + P.y) / 2)
        e = Coordicate2D((E.x + P.x) / 2, (E.y + P.y) / 2)

        se = Coordicate2D((Se.x + e.x) / 2, (Se.y + e.y) / 2)
        sw = Coordicate2D((Sw.x + w.x) / 2, (Sw.y + w.y) / 2)
        ne = Coordicate2D((Ne.x + e.x) / 2, (Ne.y + e.y) / 2)
        nw = Coordicate2D((Nw.x + w.x) / 2, (Nw.y + w.y) / 2)

        # calculate the area of the cell
        S_n = calculate_area(Ne, e, w, Nw)
        S_s = calculate_area(e, Se, Sw, w)
        S_w = calculate_area(n, s, sW, nW)
        S_e = calculate_area(nE, sE, s, n)
        # print(S_P, S_n, S_s, S_w, S_e)

        D3 = (dy(nE, n) / 4 + dy(s, sE) / 4 + dy(sE, nE)) / S_e
        D_3 = (dy(n, nW) / 4 + dy(sW, s) / 4 + dy(nW, sW)) / S_w
        D1 = (dy(Se, e) / 4 + dy(w, Sw) / 4 + dy(Sw, Se)) / S_s
        D_1 = (dy(e, Ne) / 4 + dy(Nw, w) / 4 + dy(Ne, Nw)) / S_n

        # NW
        D_4 = dy(ne, nw) / (4 * S_n) + dy(nw, sw) / (4 * S_w)

        # NE
        D2 = dy(se, ne) / (4 * S_e) + dy(ne, nw) / (4 * S_n)

        # SW
        D_2 = dy(sw, se) / (4 * S_s) + dy(nw, sw) / (4 * S_w)

        # SE
        D4 = dy(se, ne) / (4 * S_e) + dy(sw, se) / (4 * S_s)

        # Center (P)
        D0 = (
            (dy(n, s) + dy(nE, n) / 4 + dy(s, sE) / 4) / S_e
            + (dy(w, e) + dy(e, Ne) / 4 + dy(Nw, w) / 4) / S_n
            + (dy(e, w) + dy(Se, e) / 4 + dy(w, Sw) / 4) / S_s
            + (dy(s, n) + dy(n, nW) / 4 + dy(sW, s) / 4) / S_w
        )
        stencil[self.index(i, j)] = -D0
        stencil[self.index(i - 1, j)] = D_1
        stencil[self.index(i + 1, j)] = D1
        stencil[self.index(i, j - 1)] = D_3
        stencil[self.index(i, j + 1)] = D3
        stencil[self.index(i - 1, j - 1)] = D_4
        stencil[self.index(i - 1, j + 1)] = D2
        stencil[self.index(i + 1, j - 1)] = D_2
        stencil[self.index(i + 1, j + 1)] = D4

        return stencil

    def get_north(self, i, j):
        stencil = np.zeros(self.nm)
        # principle node coordinate
        P = Coordicate2D(self.X[i, j], self.Y[i, j])
        S = Coordicate2D(self.X[i + 1, j], self.Y[i + 1, j])
        W = Coordicate2D(self.X[i, j - 1], self.Y[i, j - 1])
        E = Coordicate2D(self.X[i, j + 1], self.Y[i, j + 1])
        SW = Coordicate2D(self.X[i + 1, j - 1], self.Y[i + 1, j - 1])
        SE = Coordicate2D(self.X[i + 1, j + 1], self.Y[i + 1, j + 1])

        # auxiliary node coordinate
        Sw = Coordicate2D((S.x + SW.x) / 2, (S.y + SW.y) / 2)
        Se = Coordicate2D((S.x + SE.x) / 2, (S.y + SE.y) / 2)
        sW = Coordicate2D((W.x + SW.x) / 2, (W.y + SW.y) / 2)
        sE = Coordicate2D((E.x + SE.x) / 2, (E.y + SE.y) / 2)

        s = Coordicate2D((S.x + P.x) / 2, (S.y + P.y) / 2)
        w = Coordicate2D((W.x + P.x) / 2, (W.y + P.y) / 2)
        e = Coordicate2D((E.x + P.x) / 2, (E.y + P.y) / 2)

        # calculate the area of the cell
        S_s = calculate_area(e, Se, Sw, w)
        S_ssw = calculate_area(P, s, sW, W)
        S_sse = calculate_area(E, sE, s, P)

        # East
        D3 = (dy(Se, e) / 4) / S_s + (
            dy(s, sE) / 4 + 3 * dy(sE, E) / 4 + dy(E, P) / 2
        ) / S_sse

        # West
        D_3 = (3 * dy(W, sW) / 4 + dy(sW, s) / 4 + dy(P, W) / 2) / S_ssw + (
            dy(w, Sw) / 4
        ) / S_s

        # South
        D1 = (
            (dy(sW, s) / 4 + dy(s, P) / 4) / S_ssw
            + (dy(w, Sw) / 4 + dy(Sw, Se) + dy(Se, e) / 4) / S_s
            + (dy(P, s) / 4 + dy(s, sE) / 4) / S_sse
        )

        # SW
        D_2 = (dy(W, sW) / 4 + dy(sW, s) / 4) / S_ssw + (dy(w, Sw) / 4) / S_s

        # SE
        D4 = (dy(Se, e) / 4) / S_s + (dy(s, sE) / 4 + dy(sE, E) / 4) / S_sse

        if self.boundary[0] == "N":
            coefficient = 0.0
        elif self.boundary[0] == "R":
            coefficient = self.alpha

        D0 = (
            coefficient * dist(e, w)
            + (dy(sW, s) / 4 + 3 * dy(s, P) / 4 + dy(P, W) / 2) / S_ssw
            + (dy(w, Sw) / 4 + dy(Se, e) / 4 + dy(e, w)) / S_s
            + (3 * dy(P, s) / 4 + dy(s, sE) / 4 + dy(E, P) / 2) / S_sse
        )

        stencil[self.index(i, j)] = D0
        stencil[self.index(i + 1, j)] = D1
        stencil[self.index(i, j - 1)] = D_3
        stencil[self.index(i, j + 1)] = D3
        stencil[self.index(i + 1, j - 1)] = D_2
        stencil[self.index(i + 1, j + 1)] = D4

        return stencil

    def get_south(self, i, j):
        stencil = np.zeros(self.nm)
        # principle node coordinate
        P = Coordicate2D(self.X[i, j], self.Y[i, j])
        N = Coordicate2D(self.X[i - 1, j], self.Y[i - 1, j])
        W = Coordicate2D(self.X[i, j - 1], self.Y[i, j - 1])
        E = Coordicate2D(self.X[i, j + 1], self.Y[i, j + 1])
        NW = Coordicate2D(self.X[i - 1, j - 1], self.Y[i - 1, j - 1])
        NE = Coordicate2D(self.X[i - 1, j + 1], self.Y[i - 1, j + 1])

        # auxiliary node coordinate
        Nw = Coordicate2D((N.x + NW.x) / 2, (N.y + NW.y) / 2)
        Ne = Coordicate2D((N.x + NE.x) / 2, (N.y + NE.y) / 2)
        nW = Coordicate2D((W.x + NW.x) / 2, (W.y + NW.y) / 2)
        nE = Coordicate2D((E.x + NE.x) / 2, (E.y + NE.y) / 2)

        n = Coordicate2D((N.x + P.x) / 2, (N.y + P.y) / 2)
        w = Coordicate2D((W.x + P.x) / 2, (W.y + P.y) / 2)
        e = Coordicate2D((E.x + P.x) / 2, (E.y + P.y) / 2)

        # calculate the area of the cell
        S_n = calculate_area(Ne, e, w, Nw)
        S_nnw = calculate_area(n, P, W, nW)
        S_nne = calculate_area(nE, E, P, n)

        # East
        D3 = (dy(Ne, e) / 4) / S_n + (
            dy(n, nE) / 4 + 3 * dy(nE, E) / 4 + dy(E, P) / 2
        ) / S_nne

        # West
        D_3 = (3 * dy(W, nW) / 4 + dy(nW, n) / 4 + dy(P, W) / 2) / S_nnw + (
            dy(w, Nw) / 4
        ) / S_n

        # North
        D_1 = (
            (dy(nW, n) / 4 + dy(n, P) / 4) / S_nnw
            + (dy(w, Nw) / 4 + dy(Nw, Ne) + dy(Ne, e) / 4) / S_n
            + (dy(P, n) / 4 + dy(n, nE) / 4) / S_nne
        )

        # NW
        D_4 = (dy(W, nW) / 4 + dy(nW, n) / 4) / S_nnw + (dy(w, Nw) / 4) / S_n

        # NE
        D2 = (dy(Ne, e) / 4) / S_n + (dy(n, nE) / 4 + dy(nE, E) / 4) / S_nne

        coefficient = 0.0
        if self.boundary[1] == "N":
            coefficient = 0.0
        elif self.boundary[1] == "R":
            coefficient = self.alpha

        D0 = (
            coefficient * dist(e, w)
            + (dy(nW, n) / 4 + 3 * dy(n, P) / 4 + dy(P, W) / 2) / S_nnw
            + (dy(w, Nw) / 4 + dy(Ne, e) / 4 + dy(e, w)) / S_n
            + (3 * dy(P, n) / 4 + dy(n, nE) / 4 + dy(E, P) / 2) / S_nne
        )

        stencil[self.index(i, j)] = D0
        stencil[self.index(i - 1, j)] = D_1
        stencil[self.index(i, j - 1)] = D_3
        stencil[self.index(i, j + 1)] = D3
        stencil[self.index(i - 1, j - 1)] = D_4
        stencil[self.index(i - 1, j + 1)] = D2

        return stencil

    def get_east(self, i, j):
        stencil = np.zeros(self.nm)
        # principle node coordinate
        P = Coordicate2D(self.X[i, j], self.Y[i, j])
        N = Coordicate2D(self.X[i - 1, j], self.Y[i - 1, j])
        S = Coordicate2D(self.X[i + 1, j], self.Y[i + 1, j])
        W = Coordicate2D(self.X[i, j - 1], self.Y[i, j - 1])
        NW = Coordicate2D(self.X[i - 1, j - 1], self.Y[i - 1, j - 1])
        SW = Coordicate2D(self.X[i + 1, j - 1], self.Y[i + 1, j - 1])

        # auxiliary node coordinate
        Nw = Coordicate2D((N.x + NW.x) / 2, (N.y + NW.y) / 2)
        Sw = Coordicate2D((S.x + SW.x) / 2, (S.y + SW.y) / 2)
        nW = Coordicate2D((W.x + NW.x) / 2, (W.y + NW.y) / 2)
        sW = Coordicate2D((W.x + SW.x) / 2, (W.y + SW.y) / 2)

        n = Coordicate2D((N.x + P.x) / 2, (N.y + P.y) / 2)
        s = Coordicate2D((S.x + P.x) / 2, (S.y + P.y) / 2)
        w = Coordicate2D((W.x + P.x) / 2, (W.y + P.y) / 2)

        # calculate the area of the cell
        S_w = calculate_area(n, s, sW, nW)
        S_wws = calculate_area(P, S, Sw, w)
        S_wwn = calculate_area(N, P, w, Nw)

        # West
        D_3 = (
            (dy(Sw, w) / 4 + dy(w, P) / 4) / S_wws
            + (dy(s, sW) / 4 + dy(sW, nW) + dy(nW, n) / 4) / S_w
            + (dy(P, w) / 4 + dy(w, Nw) / 4) / S_wwn
        )

        # North
        D_1 = (3 * dy(N, Nw) / 4 + dy(Nw, w) / 4 + dy(P, N) / 2) / S_wwn + (
            dy(n, nW) / 4
        ) / S_w

        # South
        D1 = (3 * dy(S, Sw) / 4 + dy(Sw, w) / 4 + dy(P, S) / 2) / S_wws + (
            dy(s, sW) / 4
        ) / S_w

        # SW
        D_2 = (dy(w, Sw) / 4 + dy(Sw, S) / 4) / S_wws + (dy(s, sW) / 4) / S_w

        # NW
        D_4 = (dy(nW, n) / 4) / S_w + (dy(w, Nw) / 4 + dy(Nw, N) / 4) / S_wwn

        coefficient = 0.0

        D0 = (
            coefficient * dist(n, s)
            + (dy(Nw, w) / 4 + 3 * dy(w, P) / 4 + dy(P, N) / 2) / S_wwn
            + (dy(n, nW) / 4 + dy(s, sW) / 4 + dy(n, s)) / S_w
            + (3 * dy(P, w) / 4 + dy(w, Sw) / 4 + dy(S, P) / 2) / S_wws
        )

        stencil[self.index(i, j)] = D0
        stencil[self.index(i - 1, j)] = D_1
        stencil[self.index(i + 1, j)] = D1
        stencil[self.index(i, j - 1)] = D_3
        stencil[self.index(i - 1, j - 1)] = D_4
        stencil[self.index(i + 1, j - 1)] = D_2

        return stencil

    def get_NW(self, i, j):
        stencil = np.zeros(self.nm)
        # principle node coordinate
        P = Coordicate2D(self.X[i, j], self.Y[i, j])
        S = Coordicate2D(self.X[i + 1, j], self.Y[i + 1, j])
        E = Coordicate2D(self.X[i, j + 1], self.Y[i, j + 1])
        SE = Coordicate2D(self.X[i + 1, j + 1], self.Y[i + 1, j + 1])

        # auxiliary node coordinate
        Se = Coordicate2D((S.x + SE.x) / 2, (S.y + SE.y) / 2)
        sE = Coordicate2D((E.x + SE.x) / 2, (E.y + SE.y) / 2)

        s = Coordicate2D((S.x + P.x) / 2, (S.y + P.y) / 2)
        e = Coordicate2D((E.x + P.x) / 2, (E.y + P.y) / 2)

        # calculate the area of the cell
        S_sse = calculate_area(E, sE, s, P)
        S_ees = calculate_area(e, Se, S, P)
        S_se = calculate_area(E, SE, S, P)

        # East
        D3 = (dy(SE, E) / 4) / S_se + (
            dy(s, sE) / 4 + 3 * dy(sE, E) / 4 + dy(E, P) / 2
        ) / S_sse

        # South
        D1 = (dy(SE, S) / 4) / S_se + (
            dy(e, Se) / 4 + 3 * dy(Se, S) / 4 + dy(S, P) / 2
        ) / S_ees

        # SE
        D4 = (
            (dy(P, E) / 4 + dy(E, sE) + dy(Se, S) + dy(P, S) / 4) / S_se
            + (dy(s, sE) / 4 + dy(s, P) / 4) / S_sse
            + (dy(e, Se) / 4 + dy(e, P) / 4) / S_ees
        )

        if self.boundary[0] == "N":
            coefficient = 0.0
        elif self.boundary[0] == "R":
            coefficient = self.alpha

        D0 = (
            coefficient * (dist(P, e) + dist(P, s))
            + (dy(e, Se) / 4 + 6 * dy(Se, S) / 4 + dy(P, S) / 2) / S_ees
            + (dy(e, Se) / 4 + dy(s, sE) / 4 + dy(P, e) + dy(P, s)) / S_se
            + (6 * dy(E, sE) / 4 + dy(s, sE) / 4 + dy(E, P) / 2) / S_sse
        )

        stencil[self.index(i, j)] = D0
        stencil[self.index(i + 1, j)] = D1
        stencil[self.index(i, j + 1)] = D3
        stencil[self.index(i + 1, j + 1)] = D4

        return stencil

    def get_NE(self, i, j):
        stencil = np.zeros(self.nm)
        # principle node coordinate
        P = Coordicate2D(self.X[i, j], self.Y[i, j])
        S = Coordicate2D(self.X[i + 1, j], self.Y[i + 1, j])
        W = Coordicate2D(self.X[i, j - 1], self.Y[i, j - 1])
        SW = Coordicate2D(self.X[i + 1, j - 1], self.Y[i + 1, j - 1])

        # auxiliary node coordinate
        Sw = Coordicate2D((S.x + SW.x) / 2, (S.y + SW.y) / 2)
        sW = Coordicate2D((W.x + SW.x) / 2, (W.y + SW.y) / 2)

        s = Coordicate2D((S.x + P.x) / 2, (S.y + P.y) / 2)
        w = Coordicate2D((W.x + P.x) / 2, (W.y + P.y) / 2)

        # calculate the area of the cell
        S_ssw = calculate_area(P, s, sW, W)
        S_wws = calculate_area(P, S, Sw, w)
        S_sw = calculate_area(P, S, SW, W)

        # West
        D_3 = (dy(SW, W) / 4) / S_sw + (
            dy(s, sW) / 4 + 3 * dy(sW, W) / 4 + dy(W, P) / 2
        ) / S_ssw

        # South
        D1 = (dy(SW, S) / 4) / S_sw + (
            dy(w, Sw) / 4 + 3 * dy(Sw, S) / 4 + dy(S, P) / 2
        ) / S_wws

        # SW
        D_2 = (
            (dy(P, W) / 4 + dy(W, sW) + dy(Sw, S) + dy(P, S) / 4) / S_sw
            + (dy(s, sW) / 4 + dy(s, P) / 4) / S_ssw
            + (dy(w, Sw) / 4 + dy(w, P) / 4) / S_wws
        )

        if self.boundary[0] == "N":
            coefficient = 0.0
        elif self.boundary[0] == "R":
            coefficient = self.alpha

        D0 = (
            coefficient * (dist(P, w) + dist(P, s))
            + (dy(w, Sw) / 4 + 6 * dy(Sw, S) / 4 + dy(P, S) / 2) / S_wws
            + (dy(w, Sw) / 4 + dy(s, sW) / 4 + dy(P, w) + dy(P, s)) / S_sw
            + (6 * dy(W, sW) / 4 + dy(s, sW) / 4 + dy(W, P) / 2) / S_ssw
        )

        stencil[self.index(i, j)] = D0
        stencil[self.index(i + 1, j)] = D1
        stencil[self.index(i, j - 1)] = D_3
        stencil[self.index(i + 1, j - 1)] = D_2

        return stencil

    def get_SW(self, i, j):
        stencil = np.zeros(self.nm)
        # principle node coordinate
        P = Coordicate2D(self.X[i, j], self.Y[i, j])
        N = Coordicate2D(self.X[i - 1, j], self.Y[i - 1, j])
        E = Coordicate2D(self.X[i, j + 1], self.Y[i, j + 1])
        NE = Coordicate2D(self.X[i - 1, j + 1], self.Y[i - 1, j + 1])

        # auxiliary node coordinate
        Ne = Coordicate2D((N.x + NE.x) / 2, (N.y + NE.y) / 2)
        nE = Coordicate2D((N.x + NE.x) / 2, (E.y + NE.y) / 2)

        n = Coordicate2D((N.x + P.x) / 2, (N.y + P.y) / 2)
        e = Coordicate2D((E.x + P.x) / 2, (E.y + P.y) / 2)

        # calculate the area of the cell
        S_nne = calculate_area(nE, E, P, n)
        S_een = calculate_area(Ne, e, P, N)
        S_ne = calculate_area(NE, E, P, N)

        # East
        D3 = (dy(NE, E) / 4) / S_ne + (
            dy(n, nE) / 4 + 3 * dy(nE, E) / 4 + dy(E, P) / 2
        ) / S_nne

        # North
        D_1 = (dy(NE, N) / 4) / S_ne + (
            dy(e, Ne) / 4 + 3 * dy(Ne, N) / 4 + dy(N, P) / 2
        ) / S_een

        # NE
        D2 = (
            (dy(P, E) / 4 + dy(E, nE) + dy(Ne, N) + dy(P, N) / 4) / S_ne
            + (dy(n, nE) / 4 + dy(n, P) / 4) / S_nne
            + (dy(e, Ne) / 4 + dy(e, P) / 4) / S_een
        )

        coefficient = 0.0
        if self.boundary[1] == "N":
            coefficient = 0.0
        elif self.boundary[1] == "R":
            coefficient = self.alpha

        D0 = (
            coefficient * (dist(P, e) + dist(P, n))
            + (dy(e, Ne) / 4 + 6 * dy(Ne, N) / 4 + dy(P, N) / 2) / S_een
            + (dy(e, Ne) / 4 + dy(n, nE) / 4 + dy(P, e) + dy(P, n)) / S_ne
            + (6 * dy(E, nE) / 4 + dy(n, nE) / 4 + dy(E, P) / 2) / S_nne
        )

        stencil[self.index(i, j)] = D0
        stencil[self.index(i - 1, j)] = D_1
        stencil[self.index(i, j + 1)] = D3
        stencil[self.index(i - 1, j + 1)] = D2

        return stencil

    def get_SE(self, i, j):
        stencil = np.zeros(self.nm)
        # principle node coordinate
        P = Coordicate2D(self.X[i, j], self.Y[i, j])
        N = Coordicate2D(self.X[i - 1, j], self.Y[i - 1, j])
        W = Coordicate2D(self.X[i, j - 1], self.Y[i, j - 1])
        NW = Coordicate2D(self.X[i - 1, j - 1], self.Y[i - 1, j - 1])

        # auxiliary node coordinate
        Nw = Coordicate2D((N.x + NW.x) / 2, (N.y + NW.y) / 2)
        nW = Coordicate2D((N.x + NW.x) / 2, (W.y + NW.y) / 2)

        n = Coordicate2D((N.x + P.x) / 2, (N.y + P.y) / 2)
        w = Coordicate2D((W.x + P.x) / 2, (W.y + P.y) / 2)

        # calculate the area of the cell
        S_nnw = calculate_area(n, P, W, nW)
        S_wwn = calculate_area(N, P, w, Nw)
        S_nw = calculate_area(N, P, W, NW)

        # West
        D_3 = (dy(NW, W) / 4) / S_nw + (
            dy(n, nW) / 4 + 3 * dy(nW, W) / 4 + dy(W, P) / 2
        ) / S_nnw

        # North
        D_1 = (dy(NW, N) / 4) / S_nw + (
            dy(w, Nw) / 4 + 3 * dy(Nw, N) / 4 + dy(N, P) / 2
        ) / S_wwn

        # NE
        D_4 = (
            (dy(P, W) / 4 + dy(W, nW) + dy(Nw, N) + dy(P, N) / 4) / S_nw
            + (dy(n, nW) / 4 + dy(n, P) / 4) / S_nnw
            + (dy(w, Nw) / 4 + dy(w, P) / 4) / S_wwn
        )

        coefficient = 0.0
        if self.boundary[1] == "N":
            coefficient = 0.0
        elif self.boundary[1] == "R":
            coefficient = self.alpha

        D0 = (
            coefficient * (dist(P, w) + dist(P, n))
            + (dy(w, Nw) / 4 + 6 * dy(Nw, N) / 4 + dy(P, N) / 2) / S_wwn
            + (dy(w, Nw) / 4 + dy(n, nW) / 4 + dy(P, w) + dy(P, n)) / S_nw
            + (6 * dy(W, nW) / 4 + dy(n, nW) / 4 + dy(W, P) / 2) / S_nnw
        )

        stencil[self.index(i, j)] = D0
        stencil[self.index(i - 1, j)] = D_1
        stencil[self.index(i, j - 1)] = D_3
        stencil[self.index(i - 1, j - 1)] = D_4

        return stencil

    def get_dirichlet(self, i, j, t):
        # always dirichlet
        stencil = np.zeros(self.nm)
        stencil[self.index(i, j)] = 1.0
        return stencil

    def update_AB(self, matrix_type, use_inner_at=None):
        self.A = np.zeros((self.nm, self.nm))
        self.B = np.zeros(self.nm)
        R = np.zeros((self.nm, self.nm))
        dTdX = np.zeros((self.nm, self.nm))

        for i in range(self.n):
            for j in range(self.m):
                stencil, b = self.second_order_stencil(i, j, use_inner_at)
                idx = self.index(i, j)
                dTdX[idx] = self.first_order_stencil(i, j)
                R[idx] = stencil
                self.B[idx] = b

        R = self.D * R
        L = self.Vx * dTdX
        A = R - L
        self.sparse_or_dense(A, matrix_type=matrix_type)

    def sparse_or_dense(self, A, matrix_type):
        match matrix_type:
            case "sparse":
                if isinstance(A, np.ndarray):
                    self.A = sparse.csr_matrix(A)

            case "dense":
                self.A = A

    def combine_boundaries(self, top_TD, bot_TD):
        self.TD[0] = top_TD[0]
        self.TD[1] = bot_TD[1]
        self.TD[2] = np.concatenate(
            (np.full(int(self.n / 2), top_TD[2]), np.full(int(self.n / 2), bot_TD[2]))
        )
        self.TD[3] = np.concatenate(
            (np.full(int(self.n / 2), top_TD[3]), np.full(int(self.n / 2), bot_TD[3]))
        )
