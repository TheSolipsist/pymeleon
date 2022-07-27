from time import perf_counter
import functools
import networkx as nx
from networkx import DiGraph
from matplotlib import pyplot as plt
from language.language import Language
from neural_net.neural_net import NeuralNet, NeuralNetError
from neural_net.metrics import Metrics
import torch


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
                    lr=0.0001, num_epochs=100, batch_size=2**16, **kwargs):
    print(f"Starting neural network metric test ({num_tests} tests)")
    general_metrics = Metrics().metric_funcs
    total_metrics = {"train": {metric_str: 0 for metric_str in general_metrics},
                     "test": {metric_str: 0 for metric_str in general_metrics}}
    bad_tests = 0
    for i in range(num_tests):
        print(f"\rCurrently running test {i + 1}", end="")
        try:
            neural_network = NeuralNet(lang, n_gen=n_gen, n_items=n_items, device_str=device_str,
                                       lr=lr, num_epochs=num_epochs, batch_size=batch_size, **kwargs)
        except NeuralNetError as e:
            print(f"\nERROR: {e}\nThis test will not be counted, continuing with next one")
            bad_tests += 1
            continue
        metrics_epoch = neural_network.metrics_epoch
        _plot_results(metrics_epoch, i)
        curr_metrics = {dataset_str: {metric_str: metric[-1] for metric_str, metric in metrics.items()}
                                      for dataset_str, metrics in metrics_epoch.items()}
        for dataset_str in total_metrics:
            for metric in total_metrics[dataset_str]:
                total_metrics[dataset_str][metric] += curr_metrics[dataset_str][metric]
    print()
    avg_metrics = {set_name: {metric: (value / (num_tests - bad_tests)) for metric, value in total_metrics[set_name].items()} for set_name in total_metrics}
    to_print = f"Averaged results from {num_tests} tests:\n"
    for set_name in avg_metrics:
        to_print += f"{set_name.capitalize()} set:\t"
        for metric, value in avg_metrics[set_name].items():
            to_print += f"{metric}: {value:.3f}   "
        to_print += "\n"
    with open("results/final_results.txt", "w") as results_file:
        results_file.write(to_print)
        print(to_print)

def _plot_results(metrics_epoch: dict[str, dict[str, torch.Tensor]], i: int = None):
    """
    Plots the metrics that were recorded during the training of the neural network

    Args:
        ``metrics_epoch`` (dict[str, dict[str, torch.Tensor]]): Dictionary containing the recorded metrics
                Example: metrics_epoch["train"]["loss"][3] gives the training loss at the 4th epoch
    """
    i = str(i) if i is not None else ""
    general_metrics = Metrics().metric_funcs
    for metric_str in general_metrics:
        fig, ax = plt.subplots()
        ax.set_title(f"{metric_str}")
        ax.set_ylabel(f"{metric_str}")
        ax.set_xlabel(f"Epoch")
        for dataset_str in metrics_epoch:
            ax.plot(metrics_epoch[dataset_str][metric_str], label=f"{dataset_str} set")
        ax.legend()
        # plt.show()
        fig.savefig(f"results/{metric_str}_{i}.png", dpi=150)
        plt.close(fig)
    
