from __future__ import annotations

from abc import abstractmethod
from typing import override

from scigraph.analyses.abc import Analysis
from scigraph.analyses.curvefit import CurveFit


class ModelComparison(Analysis):

    def __init__(self, model_one: CurveFit, model_two: CurveFit) -> None:
        self._model_one = model_one
        self._model_two = model_two
