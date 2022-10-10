import pymeleon as pym
import pygrank as pg
import networkx as nx
import numpy as np
from time import perf_counter as pc

def list2dict(x: list) -> dict:
    return {v: 1 for v in x}


def signal2graph(x: pg.GraphSignal) -> nx.Graph:
    return x.graph


def str2list(x: str) -> list:
    return [x]


def concat(x: list, y: list) -> list:
    return x + y

viewer = pym.DSL(
    pym.autorule(signal2graph),
    pym.autorule(list2dict),
    pym.autorule(str2list),
    pym.autorule(concat),
    pym.Rule(pym.parse({"graph": nx.Graph, "data": dict}),
             pym.parse("pg.to_signal(graph, data)", {"pg.to_signal": ("output_signal", pg.GraphSignal)})),
    name="pygrank_dsl"
) >> pym.GeneticViewer({"pg": pg}, use_pretrained=True, hyperparams={"num_epochs": 10000}, device_str="cuda")

G = nx.Graph()
G.add_edge("node_a", "node_b")

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
    Apply signal2graph, str2list for each str, concat for the 2 lists, list2dict and use these to get pg.to_signal
    """
    return viewer(signal, str_obj1, str_obj2) >> pym.parse({"a": "output_signal"})