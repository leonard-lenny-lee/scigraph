from typing import override, Literal

from numpy.typing import NDArray
from pandas import DataFrame, MultiIndex
from scipy.stats import t, gstd

from .abc import DataTable, DataSet
from scigraph.config import SG_DEFAULTS


class XYTable(DataTable):

    def __init__(
        self,
        values: NDArray,
        n_x_replicates: int,
        n_y_replicates: int,
        n_datasets: int,
    ) -> None:
        values = self._sanitize_values(values)
        expected_ncols = n_x_replicates + n_y_replicates * n_datasets
        if expected_ncols != (n_cols := values.shape[1]):
            raise ValueError(f"Expected {expected_ncols}, found {n_cols}")

        # Protected attributes
        self._values = values
        self._n_x_replicates = n_x_replicates
        self._n_y_replicates = n_y_replicates
        self._n_datasets = n_datasets
        self._dataset_names = self._default_names(n_datasets)

        self.x_title: str = SG_DEFAULTS["datatables.xy.x_title"]
        self.y_title: str = SG_DEFAULTS["datatables.xy.y_title"]

    @override
    def as_df(self) -> DataFrame:
        return DataFrame(self._values, columns=self._columns())

    @property
    @override
    def dataset_ids(self) -> list[str]:
        return self._dataset_names

    @override
    def get_dataset(self, name: str) -> DataSet:
        if name not in self._dataset_index_map:
            raise KeyError(f"{name} is not a dataset name.")

        dataset_idx = self._dataset_index_map[name]
        start_col = dataset_idx * self._n_y_replicates
        end_col = start_col + self._n_y_replicates

        return DataSet(
            x=self.x_values,
            y=self.y_values[:, start_col:end_col]
       )

    def summarize(
        self,
        method: Literal["mean", "median", "geometric mean", "sd", "sem",
                        "ci95", "min", "max", "range", "geometric sd", "n"],
    ) -> DataFrame:
        cols = self._columns().unique(level=0)
        return self._summarize(method).T.reindex(columns=cols)

    def _summarize(
        self,
        method: Literal["mean", "median", "geometric mean", "sd", "sem",
                        "ci95", "min", "max", "range", "geometric sd", "n"],
    ) -> DataFrame:
        groupby = self.as_df().T.groupby(level=0)
        match method:
            case "mean":
                out = groupby.mean()
            case "median":
                out = groupby.median()
            case "geometric mean":
                out = groupby.prod() ** (1 / self._summarize("n"))
            case "sd":
                out = groupby.std()
            case "sem":
                out = groupby.std() / self._summarize("n") ** 0.5
            case "ci95":
                n = self._summarize("n")
                critical_vals = t.ppf(0.975, n - 1)
                out = critical_vals * groupby.std() / n ** 0.5
            case "min":
                out = groupby.min()
            case "max":
                out = groupby.max()
            case "range":
                out = self._summarize("max") - self._summarize("min")
            case "geometric sd":
                out = groupby.agg(gstd)
            case "n":
                out = groupby.agg(lambda x: (~x.isna()).sum())
            case _:
                raise NotImplementedError
        return out

    @property
    @override
    def values(self) -> NDArray:
        return self._values

    @property
    def y_values(self) -> NDArray:
        return self._values[:, self._n_x_replicates:]

    @property
    def x_values(self) -> NDArray:
        return self._values[:, :self._n_x_replicates]

    @property
    def n_x_replicates(self) -> int:
        return self._n_x_replicates

    @property
    def n_y_replicates(self) -> int:
        return self._n_y_replicates

    @property
    def n_datasets(self) -> int:
        return self._n_datasets

    @property
    def dataset_names(self) -> set[str]:
        return set(self._dataset_names)

    @dataset_names.setter
    def dataset_names(self, names: list[str]) -> None:
        contains_duplicates = len(names) != len(set(names))
        incorrect_len = len(names) != self._n_datasets

        if contains_duplicates:
            raise ValueError("Duplicate dataset names not allowed.")
        if incorrect_len:
            raise ValueError("Inappropriate number of names provided")

        self._dataset_names = names
        self._generate_dataset_index_map()

    def _columns(self) -> MultiIndex:
        tuples = []
        for n in range(self._n_x_replicates):
            tuples.append((self.x_title, n + 1))
        for dataset in self._dataset_names:
            for n in range(self._n_y_replicates):
                tuples.append((dataset, n + 1))
        return MultiIndex.from_tuples(tuples)

    def _generate_dataset_index_map(self) -> None:
        assert len(self._dataset_names) == self._n_datasets

        self._dataset_index_map: dict[str, int] = dict(
            zip(self._dataset_names, range(self._n_datasets))
        )

    def _reduce_by_row_dataset_column(self, f) -> DataFrame:
        out = self.as_df().T.groupby(level=0).agg(f)
        # Transform shape back into original
        cols = self._columns().unique(level=0)
        return out.T.reindex(columns=cols)
