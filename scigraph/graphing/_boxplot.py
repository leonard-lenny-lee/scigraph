import matplotlib.pyplot as plt
from numpy import isnan

from tables import ColumnTable
from ._graph import Graph


class BoxPlot(Graph):

    def _check_dt_type(self) -> None:
        if not isinstance(self.dt, ColumnTable):
            raise ValueError

    def plot(self, *args, **kwargs):
        fig, ax = plt.subplots()
        data = self.dt.values
        # Remove NaN values
        mask = ~isnan(data)
        filtered_data = [d[m] for d, m in zip(data.T, mask.T)]
        labels = self.dt.group_names
        ax.boxplot(filtered_data, labels=labels, *args, **kwargs)
        return fig
