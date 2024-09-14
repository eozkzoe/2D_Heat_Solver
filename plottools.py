import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import numpy as np


def plot_tmap(T_map, shape, N, M, X, Y, problem_type, timeline=None, t_steps=1):
    match problem_type:
        case "steady":
            plt.figure(figsize=(10, 10))
            plt.contourf(X, Y, T_map, levels=20, cmap="jet")
            plt.colorbar(label="Temperature")
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.title(f"2D Temperature Distribution for {shape}")

            for i in range(N - 1):
                plt.plot(X[i, :], Y[i, :], color="black", linestyle="-", linewidth=0.5)
            for j in range(M - 1):
                plt.plot(X[:, j], Y[:, j], color="black", linestyle="-", linewidth=0.5)

            plt.show()

            # 3D
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Temperature")
            ax.set_title(f"3D Temperature Distribution for {shape}")
            surf = ax.plot_surface(X, Y, T_map, cmap="jet")
            # Adding color bar
            fig.colorbar(surf, shrink=0.5, aspect=5)

            plt.show()

        case "unsteady":
            assert t_steps > 1, "for unsteady problems, specify no. of time steps > 1"
            assert timeline is not None, "for unsteady problems, provide timeline"
            assert (
                T_map.ndim == 3
            ), "for unsteady problems, T_map must include time dimension"
            assert (
                T_map.shape[0] == t_steps
            ), "Ensure T_map has the required no. of timesteps"
            # T_map[-1] = np.roll(T_map[-1], -2, axis=0)
            # T_map[-1] = np.concatenate(
            #     (T_map[-1][:20], np.roll(T_map[-1][20:], -5, axis=0))
            # )
            plt.figure(figsize=(10, 10))
            plt.contourf(X, Y, T_map[-1], levels=20, cmap="jet")
            plt.colorbar(label="Temperature")
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.title(f"2D Temperature Distribution for linear")

            for i in range(N - 1):
                plt.plot(X[i, :], Y[i, :], color="black", linestyle="-", linewidth=0.5)
            for j in range(M - 1):
                plt.plot(X[:, j], Y[:, j], color="black", linestyle="-", linewidth=0.5)

            plt.show()
            plt.cla()

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Temperature")
            max_temp, min_temp = T_map.max(), T_map.min()
            ticks = np.linspace(min_temp, max_temp, 10)
            ax.set_zlim3d(0, 100)
            # initial T
            surf = ax.plot_surface(
                X, Y, T_map[0], cmap="jet", vmin=min_temp, vmax=max_temp
            )
            cbar = fig.colorbar(surf, shrink=0.5, aspect=5, ticks=ticks)

            def anim_callback(frame):
                ax.clear()
                ax.set_title(
                    f"3D Temperature Distribution for linear at t = {timeline[frame]:.2f}ms"
                )
                surf = ax.plot_surface(
                    X, Y, T_map[frame], cmap="jet", vmin=min_temp, vmax=max_temp
                )
                cbar.update_normal(surf)
                ax.set_zlim3d(min_temp, max_temp)
                # change color bar
                return (surf,)

            global unsteady_animation
            unsteady_animation = FuncAnimation(
                fig=fig,
                func=anim_callback,
                frames=t_steps,
                interval=10,
            )

            unsteady_animation.save("/Users/eozk/Desktop/ctfd_simulation.mp4")
            plt.show()


def inspect_matrix(self, matrix):
    # matrix visualisation for debugging
    rows, cols = np.nonzero(matrix)
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(14, 7, forward=True)
    ax1 = axes[0]
    ax2 = axes[1]
    ax1.scatter(cols, rows, c="blue")
    ax1.set_title(f"Shape: {matrix.shape}")
    ax1.invert_yaxis()
    ax1.grid(True)

    ax2.spy(matrix)
    plt.show()
