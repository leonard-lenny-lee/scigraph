from __future__ import annotations

from scigraph.analyses.abc import Analysis
from scigraph.analyses.curvefit import CurveFit, GlobalCurveFit

type Model = CurveFit | GlobalCurveFit


class ModelComparison(Analysis):

    def __init__(self, *models: Model) -> None:
        if not models:
            raise ValueError()

        for model in models:
            model.fit()
            model.table
        self._models = [_ModelAdapter(model) for model in models]


class _ModelAdapter:

    def __init__(self, model: Model) -> None:
        model.analyze()
        self._model = model
