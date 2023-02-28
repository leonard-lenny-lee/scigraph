import scigraph as s

data_array = [
    [1, 1.2, 1.4, 1.32, 7.4, 6.7, 7.0],
    [2, 2.3, 1.76, 1.8, 6.51, 6.2, 6.1],
    [3, 4, 4.2, 4.32, 4.67, 4.8, 5.01],
    [4, 5.6, 5.8, 5.98, 2.4, 2.56, 2.7],
    [5, 6.3, 6.5, 6.7, 1.5, 1.23, 1.3]
]

dt = s.tables.XYTable(data_array, 1, 2, 3)
graph = s.graphing.LineGraph(dt, "mean", "all")
graph.plot()
