import inspect

import scigraph.analyses as analyses
import scigraph.analyses.curvefit as curvefit
import scigraph.analyses.ttest as ttest
import scigraph.datatables as datatables
import scigraph.graphs as graphs
import scigraph.layouts as layouts
import scigraph.styles as styles


PUBLIC_MODULES = [
    analyses,
    curvefit,
    ttest,
    datatables,
    graphs,
    layouts,
    styles,
]


def test_public_classes_and_functions_have_docstrings():
    """Keep the supported import surface discoverable in notebooks and IDEs."""
    missing = []
    for module in PUBLIC_MODULES:
        for name in module.__all__:
            value = getattr(module, name)
            if inspect.isclass(value) or inspect.isfunction(value):
                if not inspect.getdoc(value):
                    missing.append(f"{module.__name__}.{name}")

    assert not missing, "Missing public docstrings: " + ", ".join(missing)
