import pymeleon as pym
import pygrank as pg
import networkx as nx


def list2dict(x: list) -> dict:
    return {v: 1 for v in x}


def signal2graph(x: pg.GraphSignal) -> nx.Graph:
    return x.graph


def str2list(x: str) -> list:
    return [x]


def concat(x: list, y: list) -> list:
    return x + y

# Don't measure these
G = nx.Graph()
G.add_edge("node_a", "node_b")

# Each function measured once
def ppr(*args):\
    # graph, list, string, graphsignal
    if len(args) == 2:
        if isinstance(args[0], str):
            args[0] = [args[0]]
        elif isinstance(args[1], str):
            args[1] = [args[1]]
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

# DSL measured once
viewer = pym.DSL(
    pym.autorule(signal2graph),
    pym.autorule(list2dict),
    pym.autorule(str2list),
    pym.autorule(concat),
    pym.Rule(pym.parse({"graph": nx.Graph, "data": dict}),
             pym.parse("pg.to_signal(graph, data)", {"pg.to_signal": ("noinput", pg.GraphSignal)})),
) >> pym.GeneticViewer({"pg": pg}, use_pretrained=True, hyperparams={"num_epochs": 10000}, device_str="cuda")

# Each function measured corresponding to each non dsl function
def ppr_pym(*args):
    view = viewer(*args)
    expected = pym.parse(pg.GraphSignal)
    signal = view >> expected
    return signal >> pg.PageRank()