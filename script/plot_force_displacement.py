import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import argparse


matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams[
    "text.latex.preamble"
] = """
\\usepackage{amsmath}
\\usepackage{bm}
"""
plt.rcParams.update({'font.size': 14})
plt.rc('legend', fontsize=12)    # legend fontsize

series_names = {
    'arclength_exact': "Arc-Length, Crisfield",
    'arclength_normal': "Arc-Length, Updated Normal",
    'newton': "Newton with CG and Multigrid",
    'newton_lu': "Newton with LU Factorization",
}

filename = "force_displacement.csv"


def series_from_file(file: Path) -> int:
    return file.parent.name.rsplit("_", 1)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-dir",
        type=Path,
        default=Path.cwd() / "output",
        help="Path to output data directory",
    )
    parser.add_argument(
        "-o", "--output-file-name",
        type=str,
        default="force_displacement.png",
        help="Base name of the file to save the plot to",
    )
    parser.add_argument(
        "-t", "--title",
        type=str,
        default="Full Sinker"
    )

    args = parser.parse_args()
    data_dir: Path = args.data_dir.resolve()
    output_file_name: str = args.output_file_name
    file_extension = Path(output_file_name).suffix or ".png"
    output_file_name = output_file_name.removesuffix(file_extension)

    fig, ax = plt.subplots()
    experiments = {
        force: list(data_dir.glob(f"./*_{force}"))
        for force in set(d.name.split("_")[-1] for d in data_dir.glob("./*"))
    }

    for pforce, cases in experiments.items():
        fig, ax = plt.subplots()
        for case in sorted(cases):
            print(f"Experiment {case.name}")
            file = case / filename
            data = pd.read_csv(file)
            print(f"Read {len(data)} data points")
            force = data["force"]
            displacement = data["displacement"]

            label = f"{series_names[series_from_file(file)]}"
            ax.plot(displacement, force, label=label, marker="x")
        ax.set_title(args.title)
        ax.set_xlabel("Displacement [$m$]")
        ax.set_ylabel("Force [$N$]")
        ax.legend(loc="best")
        output_name = Path.cwd() / f"{output_file_name}_{pforce}{file_extension}"
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
