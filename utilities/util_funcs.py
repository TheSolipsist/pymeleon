from time import perf_counter
import functools
import networkx as nx
from networkx import DiGraph
from matplotlib import pyplot as plt
from language.language import Language
from neural_net.neural_net import NeuralNet

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


def save_graph(graph: DiGraph, filename: str = "temp_graph.png", print: bool = False, show_constraints = False):
    """
    Saves the given graph in a png file, or prints it if print==True
    """
    pos = nx.planar_layout(graph)
    nx.draw(graph, pos, node_size=400)
    for node in graph.nodes:
        if node == "root_node":
            nx.set_node_attributes(graph, {node: "root_node"}, "name")
        elif show_constraints:
            nx.set_node_attributes(graph, {node: (node.value, node.constraints)}, "name")
        else:
            nx.set_node_attributes(graph, {node: node.value}, "name")
    nx.draw_networkx_labels(graph, pos, labels=nx.get_node_attributes(graph, "name"), font_size=7)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, "order"), font_size=7)
    if print:
        plt.show()
    else:
        plt.savefig(filename, dpi=600)
    plt.close()

def test_neural_net(lang: Language, n_gen=5, n_items=3, device_str="cpu", num_tests=40,
                    lr=0.0001, num_epochs=100, batch_size=2**16):
    print(f"Starting neural network metric test ({num_tests} tests)")
    total_metrics = {"train": {"loss": 0, "accuracy": 0, "AUC": 0},
                     "test": {"loss": 0, "accuracy": 0, "AUC": 0}}
    for i in range(num_tests):
        print(f"\rCurrently running test {i + 1}", end="")
        neural_network = NeuralNet(lang, n_gen=n_gen, n_items=n_items, device_str=device_str,
                                   lr=lr, num_epochs=num_epochs, batch_size=batch_size)
        curr_metrics = neural_network.metrics
        for each_set in total_metrics:
            for metric in total_metrics[each_set]:
                total_metrics[each_set][metric] += curr_metrics[each_set][metric]
    print()
    avg_metrics = {set_name: {metric: (value / num_tests) for metric, value in total_metrics[set_name].items()} for set_name in total_metrics}
    print("Averaged results:")
    for set_name in avg_metrics:
        print(f"{set_name.capitalize()} set:\t", end="")
        for metric, value in avg_metrics[set_name].items():
            print(f"{metric}: {value:.3f}   ", end="")
        print()
