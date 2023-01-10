import pygrank as pg
import networkx as nx
from mylib_pymeleon import list2dict, signal2graph, str2list

def ppr(*args):
    if len(args) == 2:
        if isinstance(args[0], str):
            args[0] = str2list(args[0])
        elif isinstance(args[1], str):
            args[1] = str2list(args[1])
        if isinstance(args[0], pg.GraphSignal):
            args[0] = signal2graph(args[0])
        elif isinstance(args[1], pg.GraphSignal):
            args[1] = signal2graph(args[1])
        if isinstance(args[0], nx.Graph):
            graph = args[0]
            data = args[1]
        elif isinstance(args[1], nx.Graph):
            graph = args[1]
            data = args[0]
        if isinstance(data, list):
            data = list2dict(data)
        elif not isinstance(data, dict):
            raise ValueError
    return pg.to_signal(graph, data) >> pg.PageRank()

