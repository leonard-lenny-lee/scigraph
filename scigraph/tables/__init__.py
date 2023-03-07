"""Contains the DataTable classes upon which are used by the graphing and
analysis modules. The type of DataTable object will determine which types of
analyses and graphs can be constructed from the dataset. These are clones of
the DataTable types in GraphPad Prism 9. See the user guide for in-depth usage
https://www.graphpad.com/guides/prism/latest/user-guide/index.htm

All DataTables are thin wrappers for pandas DataFrames. While the functions and
methods in the graphing and analysis modules will operate on pandas DataFrames,
DataTables will validate the data structure and shape and guarantee the
correct operation of graphing and analysis tools.
"""

from ._datatable import *
from ._xy import *
from ._column import *
from ._grouped import *
from ._contingency import *
from ._survival import *
from ._partsofwhole import *
from ._multiplevariables import *
from ._nested import *
