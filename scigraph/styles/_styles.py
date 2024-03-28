import contextlib
import os
import pathlib

import matplotlib.pyplot as plt


ss_dir = pathlib.Path(os.path.dirname(__file__)).joinpath("stylesheets")
available_ss = [f.replace(".mplstyle", "") for f in os.listdir(ss_dir)
                if f.endswith(".mplstyle")]


def _get_stylesheet_path(ss_name) -> pathlib.Path:
    filename = ss_name + ".mplstyle"
    return ss_dir.joinpath(filename)


def _build_ss_stack(*ss) -> list[str]:
    ss_stack = []
    for s in ss:
        if s in available_ss:
            s = _get_stylesheet_path(s)
        ss_stack.append(s)
    return ss_stack


def use(*ss) -> None:
    ss_stack = _build_ss_stack(*ss)
    plt.style.use(ss_stack)


@contextlib.contextmanager
def context(*ss, after_reset: bool):
    ss_stack = _build_ss_stack(*ss)
    with plt.style.context(ss_stack, after_reset):
        yield
