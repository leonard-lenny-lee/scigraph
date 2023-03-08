import scigraph as s

data = [
    [0.5, 0.537],
    [0.25, 0.256],
    [0.125, 0.109],
    [0.0625, 0.046],
    [0.03125, 0.016],
]

dt = s.tables.XYTable(
    data, 1, 1, 1, "[BGTX]",
    "$\mathregular{mgml^{-1}}$",
    "$\mathregular{A_{280}}$"
)
analysis = s.analyses.SimpleLinearRegression(dt)
analysis.solve()
print(analysis._params)
print(analysis.inv_interpolate_extrapolate([0.1, 0.2]))
graph = analysis.graph()
graph.axes.set_xbound(lower=0)
graph.axes.set_ybound(lower=0)
