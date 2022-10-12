import pymeleon as pym
import pygrank as pg
import networkx as nx

def tuple2dict(x: tuple) -> dict:
    return {v: 1 for v in x}

def signal2graph(x: pg.GraphSignal) -> nx.Graph:
    return x.graph

def str2list(x: str) -> list:
    return [x]

def concat(x: list, y: list) -> tuple:
    return tuple(x + y)

def list2tuple(x: list) -> tuple:
    return tuple(x)

viewer = pym.DSL(
    pym.autorule(list2tuple),
    pym.autorule(signal2graph),
    pym.autorule(tuple2dict),
    pym.autorule(str2list),
    pym.autorule(concat),
    pym.Rule(pym.parse({"graph": nx.Graph, "data": dict}),
             pym.parse("pg.to_signal(graph, data)", {"pg.to_signal": ("output_signal", pg.GraphSignal)})),
    name="pygrank_test"
) >> pym.GeneticViewer({"pg": pg}, use_pretrained=True, hyperparams={"num_epochs": 10000}, device_str="cuda")

def ex_1(signal: pg.GraphSignal):
    """
    Apply signal2graph
    """
    return viewer(signal) >> pym.parse(nx.Graph)

def ex_2(signal: pg.GraphSignal, str_obj: str):
    """
    Apply signal2graph and str2list
    """
    return viewer(signal, str_obj) >> pym.parse({"a": nx.Graph, "b": list})

def ex_3(signal: pg.GraphSignal, str_obj: str, list_obj: list):
    """
    Apply signal2graph, str2list and list2dict
    """
    return viewer(signal, str_obj, list_obj) >> pym.parse({"a": nx.Graph, "b": list, "c": dict})

def ex_4(signal: pg.GraphSignal, str_obj: str):
    """
    Apply signal2graph, str2list, list2dict and use them to get pg.to_signal
    """
    return viewer(signal, str_obj) >> pym.parse({"a": "output_signal"})

def ex_5(signal: pg.GraphSignal, str_obj1: str, str_obj2: str):
    """
    Apply signal2graph, str2list each str, concat the 2 lists, list2dict and use these to get pg.to_signal
    """
    return viewer(signal, str_obj1, str_obj2) >> pym.parse({"a": "output_signal"})

G = nx.Graph()
G.add_edge("node_a", "node_b")
G_sig = pg.to_signal(G, {"node_a": 1})
x = "node_a"
y = "node_b"
l = [x, y]
print(ex_1(G_sig),
      ex_2(G_sig, x),
      ex_3(G_sig, x, l),
      ex_4(G_sig, x),
      ex_5(G_sig, x, y),
      sep="\n===============================\n")
