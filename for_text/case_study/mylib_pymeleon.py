import pymeleon as pym
import pygrank as pg
import networkx as nx

def list2dict(x: list) -> dict:
    return {v: 1 for v in x}

def signal2graph(x: pg.GraphSignal) -> nx.Graph:
    return x.graph

def str2list(x: str) -> list:
    return [x]

viewer = pym.DSL(
    pym.autorule(signal2graph),
    pym.autorule(list2dict),
    pym.autorule(str2list),
    pym.Rule(pym.parse({"graph": nx.Graph, "data": dict}),
             pym.parse("pg.to_signal(graph, data)", {"pg.to_signal": ("noinput", pg.GraphSignal)})),
) >> pym.GeneticViewer({"pg": pg})
