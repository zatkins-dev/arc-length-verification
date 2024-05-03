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


def plot_force_displacement(df: pd.DataFrame, x_axis: str = 'u_x'):
    f, ax = plt.subplots(tight_layout=True)
    line = ax.plot(df[x_axis], df["force"])[0]
    point = ax.plot(df[x_axis][0], df["force"][0], 'o')[0]
    # ax.set_ylim([-1, 1])
    ax.set_xlabel(f'${x_axis}$ (mm)')
    ax.set_ylabel('Force (N)')
    ax.grid(True)
    return f, x_axis, point


def plot_series(meshes: pv.MultiBlock, force_displacement_df: pd.DataFrame = None, field_name: str = 'MergedSolution',
                filename: str = 'solution.gif', skip: int = 1):
    dargs = dict(
        scalars=field_name,
        cmap='jet',
        show_scalar_bar=False,
    )
    write_gif = filename is not None

    plotter: pv.Plotter = pv.Plotter(notebook=False, off_screen=write_gif)
    if write_gif:
        plotter.open_gif(filename)

    grid = meshes[0].copy(True)
    plotter.add_mesh(grid, component=1, **dargs)
    plotter.view_xy()
    t = np.array(range(len(meshes)))
    figs = []
    if force_displacement_df is not None:
        figs.append(plot_force_displacement(force_displacement_df))
        figs.append(plot_force_displacement(force_displacement_df, "u_y"))

    locs = [(0.02, 0.06), (0.52, 0.06)]
    for i, (fig, _, _) in enumerate(figs):
        chart = pv.ChartMPL(fig, size=(0.46, 0.25), loc=locs[i])
        chart.background_color = (1.0, 1.0, 1.0, 0.4)
        plotter.add_chart(chart)

    def update_time(time):
        time = np.clip([int(np.floor(time))], 0, len(meshes) - 1)[0]
        for _, x_axis, point in figs:
            point.set_data(force_displacement_df[x_axis][time + 1], force_displacement_df["force"][time + 1])
        grid.points = meshes[time].points
        grid[field_name] = meshes[time].point_data[field_name]
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

    for time in t[::skip]:
        if write_gif:
            update_time(time)
            plotter.write_frame()
        else:
            time_slider.GetSliderRepresentation().SetValue(time)
            update_time(time)

    if write_gif:
        plotter.close()
    elif not plotter._closed:
        plotter.show()


if __name__ == '__main__':
    meshes = read_series('solution_..vts')
    forces = read_series('force_..vts', field_name='Force')
    for mesh, force in zip(meshes, forces):
        mesh['Force'] = force['MergedForce']
    df = pd.read_csv("force_displacement.csv")
    plot_series(meshes, df, field_name="Force", filename=None, skip=20)
