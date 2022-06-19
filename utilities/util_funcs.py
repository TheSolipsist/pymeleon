from time import perf_counter
import functools
import networkx as nx
from networkx import DiGraph
from matplotlib import pyplot as plt
from language.language import Language


def timer(func):
    """
    Timer decorator
    
    Causes a function to return the tuple (return_value, total_time):
        return_value:   the function's return value
        total_time:     the time it took for the function to execute
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        func_val = func(*args, **kwargs)
        end = perf_counter()
        total_time = end - start
        return func_val, total_time
    return wrapper


def save_graph(graph: DiGraph, filename: str = "temp_graph.png", print: bool = False):
    """
    Saves the given graph in a png file, or prints it if print==True
    """
    pos = nx.planar_layout(graph)
    nx.draw(graph, pos, node_size=400)
    for node in graph.nodes:
        if node == "root_node":
            nx.set_node_attributes(graph, {node: "root_node"}, "name")
        else:
            nx.set_node_attributes(graph, {node: node.value}, "name")
    nx.draw_networkx_labels(graph, pos, labels=nx.get_node_attributes(graph, "name"), font_size=7)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, "order"), font_size=7)
    if print:
        plt.show()
    else:
        plt.savefig(filename, dpi=600)
    plt.close()


def dfs_representation(graph: DiGraph, language: Language) -> list:
    """
    Returns a Depth First Search representation of the graph, in which each node is
    represented by a {len(language.types)}-bit vector of its constraint types
    """
