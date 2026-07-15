import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

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
