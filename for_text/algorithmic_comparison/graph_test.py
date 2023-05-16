import pymeleon as pym
import networkx as nx
from pymeleon.utilities.util_funcs import timer
from pymeleon.viewer.genetic_viewer import ViewerError

def dict2graph(x: dict) -> nx.Graph:
    G = nx.Graph()
    for k, v in x.items():
        G.add_edge(k, v)
    return G

def lists2dict(x: list, y: list) -> dict:
    return {k: v for k, v in zip(x, y)}

def ints2list(x: int, y: int) -> list:
    return [x, y]

def strs2list(x: str, y: str) -> list:
    return [x, y]

@timer
def ex_1(viewer, x: int, y: int):
    """
    Apply ints2list transformation
    """
    return viewer(x, y) >> pym.parse(list)

@timer
def ex_2(viewer, x: str, y: str):
    """
    Apply strs2list transformation
    """
    return viewer(x, y) >> pym.parse(list)

@timer
def ex_3(viewer, a: int, b: int, x: str, y: str):
    """
    Apply ints2list and strs2list transformations
    """
    return viewer(a, b, x, y) >> pym.parse({"a": list, "b": list})

@timer
def ex_4(viewer, a: int, b: int, x: str, y: str):
    """
    Apply ints2list, strs2list and lists2dict transformations
    """
    return viewer(a, b, x, y) >> pym.parse(dict)

@timer
def ex_5(viewer, a: int, b: int, x: str, y: str):
    """
    Apply ints2list, strs2list, lists2dict transformations and make the dict into a graph
    """
    return viewer(a, b, x, y) >> pym.parse(nx.Graph)

def test_example(viewer, foo, *args):
    try:
        return foo(viewer, *args)
    except ViewerError:
        return False
    
def test(fitness_str: str):
    viewer = pym.DSL(
        pym.autorule(dict2graph),
        pym.autorule(lists2dict),
        pym.autorule(ints2list),
        pym.autorule(strs2list),
        name="graph_test"
    ) >> pym.GeneticViewer(use_pretrained=True, hyperparams={"num_epochs": 2000}, device_str="cuda", fitness=fitness_str)
    a = 1
    b = 2
    c = "banana"
    d = "apple"
    return (test_example(viewer, ex_1, a, b),
            test_example(viewer, ex_2, c, d),
            test_example(viewer, ex_3, a, b, c, d),
            test_example(viewer, ex_4, a, b, c, d),
            test_example(viewer, ex_5, a, b, c, d))
