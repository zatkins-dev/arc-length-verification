import pyvista as pv
import pyvistaqt as pvqt
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams[
    "text.latex.preamble"
] = """
\\usepackage{amsmath}
\\usepackage{bm}
"""


def merge_vector_components(mesh: pv.DataSet, field_name: str = 'Solution', components: list = ['x', 'y', 'z']):
    # merge vector components into a single field
    # print(mesh.point_data)
    solution = [mesh.point_data[f'{field_name}.{comp}'] for comp in components]
    mesh.point_data[f'Merged{field_name}'] = np.vstack(solution).T
    return mesh


def read_series(mesh_name: str, dir: Path = Path.cwd(), field_name: str = 'Solution'):
    # mesh_name looks like 'name_..vts'
    file_pattern = mesh_name.replace('..', '*.')
    files = sorted(dir.glob(file_pattern), key=lambda x: int(x.stem.split('_')[-1]))
    meshes = [pv.read(str(file)) for file in files]
    meshes = [merge_vector_components(mesh, field_name) for mesh in meshes]
    meshes = [mesh.warp_by_vector(f'Merged{field_name}') for mesh in meshes]

    return pv.MultiBlock(meshes)


def plot_force_displacement(df: pd.DataFrame, x_axis: str = 'u_x', ax=None):
    if ax is None:
        f, ax = plt.subplots(tight_layout=True, dpi=150)
    else:
        f = None
    line = ax.plot(df[x_axis], df["force"])[0]
    point = ax.plot(df[x_axis][0], df["force"][0], 'o')[0]
    # ax.set_ylim([-1, 1])
    ax.set_xlabel(f'${x_axis}$ (mm)')
    ax.grid(True)
    return f, x_axis, point


def plot_series(meshes: pv.MultiBlock, base_mesh: pv.DataSet, force_displacement_df: pd.DataFrame = None, field_name: str = 'MergedSolution',
                filename: str = 'solution.gif', skip: int = 1, zoom: float = 1.0):
    dargs = dict(
        scalars=field_name,
        cmap='jet',
        lighting=False,
    )
    sargs = dict(
        height=0.1,
        width=0.8,
        position_x=0.1,
        position_y=0.85,
        title_font_size=22,
        label_font_size=16,
        title='Y Displacement (mm)',
    )
    write_gif = filename is not None

    plotter: pv.Plotter = pv.Plotter(notebook=False, off_screen=write_gif)

    grid = meshes[-1].copy(True)
    grid.points = base_mesh.points.copy()
    plotter.add_mesh(grid, show_edges=True, scalar_bar_args=sargs, **dargs)
    plotter.view_xy()
    plotter.camera.zoom(zoom)
    t = np.array(range(len(meshes)))
    grid.translate([0, 10, 0], inplace=True)
    figs = []
    fig, ax = plt.subplots(1, 2, tight_layout=True, dpi=150, sharey=True, figsize=(8, 4))
    if force_displacement_df is not None:
        figs.append(plot_force_displacement(force_displacement_df, ax=ax[0]))
        figs.append(plot_force_displacement(force_displacement_df, "u_y", ax=ax[1]))
        ax[0].set_ylabel('Force (N)')

    locs = [(0.02, 0.02), (0.52, 0.06)]
    if len(figs) > 0:
        chart = pv.ChartMPL(fig, size=(0.96, 0.35), loc=locs[0])
        chart.background_color = (1.0, 1.0, 1.0, 0.4)
        plotter.add_chart(chart)
        # for i, (fig, _, _) in enumerate(figs):
        #     chart = pv.ChartMPL(fig, size=(0.46, 0.25), loc=locs[i])
        #     chart.background_color = (1.0, 1.0, 1.0, 0.4)
        #     plotter.add_chart(chart)

    def update_time(time):
        time = np.clip([int(np.floor(time))], 0, len(meshes) - 1)[0]
        for _, x_axis, point in figs:
            if time < force_displacement_df.shape[0] - 1:
                point.set_data(force_displacement_df[x_axis][time + 1], force_displacement_df["force"][time + 1])
        grid.points[:] = meshes[time].points[:]
        grid.translate([0, 10, 0], inplace=True)
        grid.point_data[field_name][:] = meshes[time].point_data[field_name][:]
        if not write_gif:
            plotter.update()
        return time

    if not write_gif:
        plotter.show(auto_close=False, interactive=True, interactive_update=True)
        time_slider = plotter.add_slider_widget(
            update_time,
            [np.min(t), np.max(t)],
            value=0,
            title='Step',
            fmt="%12f",
            interaction_event='always',
            pointa=(0.1, 0.92),
            pointb=(0.9, 0.92),
            title_height=0.024,
            style='modern',
        )
    else:
        plotter.open_gif(filename)

    for time in t[::skip]:
        if write_gif:
            plotter.write_frame()
            update_time(time)
        else:
            time_slider.GetSliderRepresentation().SetValue(time)
            update_time(time)

    if write_gif:
        plotter.write_frame()
        plotter.close()
    elif not plotter._closed:
        plotter.show()


if __name__ == '__main__':
    # dir = Path("output/arclength_exact_-12e3")
    dir = Path("output/newton_lu_-12e3")
    meshes = read_series('solution_..vts', dir=dir)
    forces = read_series('force_..vts', field_name='Force', dir=dir)
    for mesh, force in zip(meshes, forces):
        mesh['Force'] = force['MergedForce']
    df = pd.read_csv(dir / "force_displacement.csv")
    base_mesh = pv.read(dir / "solution_1.vts")
    plot_series(meshes, base_mesh, df, field_name="Solution.y", filename="fd_fem.gif", zoom=1.25)
