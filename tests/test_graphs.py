import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from scigraph.datatables import ColumnTable, XYTable


def test_xy_graph_draws_registered_components():
    table = XYTable([[0.0, 1.0], [1.0, 3.0]], 1, 1, 1, ["response"])
    graph = table.create_xy_graph().add_points("mean").add_connecting_line("mean")
    figure, axes = plt.subplots()

    graph.draw(axes)

    assert len(axes.lines) == 2
    plt.close(figure)


def test_column_graph_factory_preserves_dataset_labels():
    table = ColumnTable([[1.0, 2.0], [3.0, 4.0]], ["control", "treated"])
    graph = table.create_xy_graph()

    assert graph.table.dataset_ids == ["control", "treated"]


def test_column_graph_draws_one_violin_per_dataset():
    table = ColumnTable(
        [[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 6.0]],
        ["control", "treated"],
    )
    graph = table.create_column_graph("vertical").add_violins(
        showmeans=True,
        body_kws={"alpha": 0.5},
    )
    figure, axes = plt.subplots()

    graph.draw(axes)

    bodies = [body for body in axes.collections if isinstance(body, PolyCollection)]
    assert len(bodies) == 2
    assert all(body.get_alpha() == 0.5 for body in bodies)
    plt.close(figure)
