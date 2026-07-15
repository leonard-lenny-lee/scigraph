import numpy as np
import pandas as pd
import pytest

from scigraph.analyses import DescriptiveStatistics, RowStatistics
from scigraph.analyses._stats import Basic
from scigraph.analyses.ttest import StudentsTTest
from scigraph.datatables import ColumnTable, GroupedTable, XYTable


def test_grouped_table_dataframe_round_trip_preserves_values_and_labels():
    columns = pd.MultiIndex.from_product([["control", "treated"], [1, 2]])
    frame = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        index=["baseline", "follow-up"],
        columns=columns,
    )

    table = GroupedTable.from_dataframe(frame)

    np.testing.assert_allclose(table.values, frame.to_numpy())
    assert table.dataset_ids == ["control", "treated"]
    assert table.row_names == ("baseline", "follow-up")
    assert table.as_df().equals(frame)


def test_grouped_table_dataframe_requires_a_uniform_replicate_count():
    frame = pd.DataFrame(
        [[1.0, 2.0, 3.0]],
        columns=pd.MultiIndex.from_tuples(
            [("control", 1), ("treated", 1), ("treated", 2)]
        ),
    )

    with pytest.raises(ValueError, match="same number of replicates"):
        GroupedTable.from_dataframe(frame)


def test_descriptive_statistics_separate_returns_one_value_per_column_dataset():
    table = ColumnTable([[1.0, 3.0], [5.0, 7.0]], ["a", "b"])

    result = DescriptiveStatistics(table, "mean", subcolumn_policy="separate").result

    assert list(result.columns) == ["a", "b"]
    np.testing.assert_allclose(result.to_numpy(), [[3.0, 5.0]])


def test_analysis_result_cache_is_invalidated_when_statistics_are_added():
    table = XYTable(
        [[0.0, 1.0], [1.0, 5.0]],
        n_x_replicates=1,
        n_y_replicates=1,
        n_datasets=1,
        dataset_names=["response"],
    )
    analysis = RowStatistics(table, "row", "mean")

    _ = analysis.result
    analysis.add_statistics("max")

    assert list(analysis.result.columns) == ["mean", "max"]


def test_zero_is_a_valid_observation_for_n_and_sem():
    values = np.array([0.0, 2.0, np.nan])

    assert Basic.n(values) == 2
    assert Basic.sem(values) == pytest.approx(1 / np.sqrt(2))


def test_two_sample_significance_uses_alpha_not_confidence_level():
    table = ColumnTable(
        [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1], [4.0, 4.1], [5.0, 5.1]],
        ["a", "b"],
    )

    result = StudentsTTest(table, ("a", "b"), confidence_level=0.95).result

    assert result.p > 0.05
    assert not result.significant
