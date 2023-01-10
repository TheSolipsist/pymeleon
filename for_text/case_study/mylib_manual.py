import pygrank as pg
import networkx as nx

def list2dict(x: list) -> dict:
    return {v: 1 for v in x}

def signal2graph(x: pg.GraphSignal) -> nx.Graph:
    return x.graph

def str2list(x: str) -> list:
    return [x]
