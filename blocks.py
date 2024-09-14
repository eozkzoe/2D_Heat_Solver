import numpy as np
import matplotlib.pyplot as plt


def get_combined_boundary(solid_boundary, fluid_boundary, order="fs"):
    if order == "fs":
        return np.hstack([fluid_boundary, solid_boundary])
    elif order == "sf":
        return np.hstack([solid_boundary, fluid_boundary])


def get_row(mesh, shape, row):
    """
    Parameters:
        Shape: should be (rows, columns) format
        Row: is row index
    """
    rows = shape[0]
    # row_mesh = np.zeros((rows, rows**2))
    row_mesh = []
    match row:
        case -1:
            for i in range(rows * (rows - 1), rows**2):
                row_mesh.append(mesh[i])

        case _:
            for i in range(rows * row, rows * (row + 1)):
                row_mesh.append(mesh[i])
                # np.append(row_mesh, mesh[i])
    return np.vstack(np.asarray(row_mesh))


def get_equal_diag(bot_row, top_row):
    bot_sums = np.sum(bot_row, axis=1)
    top_sums = np.sum(top_row, axis=1)

    average_row_sums = (bot_sums + top_sums) / 2
    bot_diff = average_row_sums - bot_sums
    top_diff = average_row_sums - top_sums

    bot_diag = np.zeros_like(bot_row)
    top_diag = np.zeros_like(top_row)

    np.fill_diagonal(bot_diag, bot_diff)
    np.fill_diagonal(top_diag, top_diff)
    return (bot_diag, top_diag)


def get_block_matrix(bottom_mesh, top_mesh, shape):
    """
    Node shapes should be similar
    Links the solid mesh to the fluid mesh

    North row has D0, D1, D_3, D3, D_2, D4
    South row has D0, D_1, D_3, D3, D_4, D2

    Result should be an 'inner node' of D_x and Dx, [0, 4]

    For the bottom mesh, you want the top row, where only
    D_2, D1, D4 are important

    For the top mesh, you want the bottom row, where only
    D_4, D_1, D2 are important
    """
    solid_top_row = get_row(bottom_mesh, shape=shape, row=0)
    fluid_bot_row = get_row(top_mesh, shape=shape, row=-1)

    # Average of two values method
    # bot_diag, top_diag = get_equal_diag(fluid_bot_row, solid_top_row)
    # fig, axes = plt.subplots(1, 2)
    # fig.set_size_inches(14, 7, forward=True)
    # ax1 = axes[0]
    # ax2 = axes[1]
    # ax1.matshow(solid_top_row, cmap=plt.cm.Blues)
    # ax2.matshow(fluid_bot_row, cmap=plt.cm.Blues)
    # plt.show()

    #  Copying missing nodes method
    top_mask = np.zeros((shape[0], shape[0] * shape[1]), dtype=bool)
    bot_mask = np.zeros((shape[0], shape[0] * shape[1]), dtype=bool)

    # # Taking the next row
    for offset in range(-1, 2):
        row_indices, col_indices = np.diag_indices(
            min(shape[0], shape[0] * shape[1] - offset)
        )
        top_mask[row_indices, col_indices + offset] = True

    solid_top_row[~top_mask] = 0

    _bot_main_diag = shape[0] * shape[1] - (shape[0])
    for offset in range(_bot_main_diag - 1, _bot_main_diag + 2):
        row_indices, col_indices = np.diag_indices(
            min(shape[0], shape[0] * shape[1] - offset)
        )
        bot_mask[row_indices, col_indices + offset] = True

    fluid_bot_row[~bot_mask] = 0

    # # Taking the corresponding row
    # for offset in range(shape[0] - 1, shape[0] + 2):
    #     row_indices, col_indices = np.diag_indices(
    #         min(shape[0], shape[0] * shape[1] - offset)
    #     )
    #     top_mask[row_indices, col_indices + offset] = True

    # solid_top_row[~top_mask] = 0

    # _bot_main_diag = shape[0] * shape[1] - (shape[0] + shape[1])
    # for offset in range(_bot_main_diag - 1, _bot_main_diag + 2):
    #     row_indices, col_indices = np.diag_indices(
    #         min(shape[0], shape[0] * shape[1] - offset)
    #     )
    #     bot_mask[row_indices, col_indices + offset] = True

    # fluid_bot_row[~bot_mask] = 0

    # Create fs, sf blocks
    mesh_dim = shape[0] * shape[1]
    fs_block = np.zeros((mesh_dim, mesh_dim))  # fluid-solid
    sf_block = np.zeros((mesh_dim, mesh_dim))  # solid-fluid

    # if you wish to align the rows
    # fs_block[shape[0] * (shape[0] - 1) : shape[0] ** 2] = np.roll(
    #     solid_top_row, -shape[0]
    # )
    # sf_block[0 : shape[0]] = np.roll(fluid_bot_row, shape[0])
    fs_block[shape[0] * (shape[0] - 1) : shape[0] ** 2] = -solid_top_row
    sf_block[0 : shape[0]] = -fluid_bot_row  

    # fig, axes = plt.subplots(1, 2)
    # fig.set_size_inches(14, 7, forward=True)
    # ax1 = axes[0]
    # ax2 = axes[1]
    # ax1.matshow(fs_block, cmap=plt.cm.Greys)
    # ax2.matshow(sf_block, cmap=plt.cm.Greys)
    # plt.show()

    big_block = np.block(
        [
            [top_mesh, fs_block],
            [sf_block, bottom_mesh],
        ]
    )
    return big_block
