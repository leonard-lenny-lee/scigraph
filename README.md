# scigraph

`scigraph` is a small, reproducibility-focused analysis and graphing library.
It provides explicit data-table types, Prism-inspired summary analyses and
curve-fitting models, composable Matplotlib graphs, and simple multi-panel
layouts.

## Core workflow

Build a table first, then create analyses and graphs from that table. Analyses
never modify their input table unless their API explicitly returns a new table.

```python
from scigraph.datatables import XYTable
from scigraph.analyses.curvefit import Logistic4Parameter

table = XYTable(
    values,
    n_x_replicates=1,
    n_y_replicates=3,
    n_datasets=2,
    dataset_names=["control", "treated"],
)

fit = Logistic4Parameter(table)
fit.add_initial_value("top", 100)
fit.fit()

parameters = fit.result
confidence_intervals = fit.profile_likelihood_CI()

graph = table.create_xy_graph()
graph.add_points("mean").add_errorbars("sem")
graph.link_analysis(fit)
```

Use the data-table factories for standard analyses:

```python
from scigraph.analyses import Normalize

summary = table.row_statistics("dataset", "mean", "sem").result
normalised_table = Normalize(table).result
```

## Public packages

- `scigraph.datatables`: `ColumnTable`, `GroupedTable`, and `XYTable`.
- `scigraph.analyses`: descriptive statistics, row statistics, and
  normalisation. Curve fitting and hypothesis tests live in the explicit
  `scigraph.analyses.curvefit` and `scigraph.analyses.ttest` subpackages.
- `scigraph.graphs`: `ColumnGraph`, `GroupedGraph`, and `XYGraph`.
- `scigraph.layouts`: `GridLayout` for multi-panel figures.
- `scigraph.styles`: stylesheet selection and temporary style contexts.

The package exports are explicit; import public classes from these packages
rather than from private modules whose names begin with an underscore.

## Development

Run the regression suite with the project environment:

```bash
venv/bin/pytest -q
```

The tests cover data-table round trips, analysis cache invalidation, statistical
edge cases, graph rendering, global/profile-likelihood curve fitting, and the
documented public API surface.
