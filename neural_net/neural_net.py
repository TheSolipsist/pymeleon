"""
Neural network implementation module
"""

from time import perf_counter
import itertools
# pymeleon modules
from language.language import Language
from language.parser import Node
from language.rule import Rule
from neural_net.training_generation import TrainingGenerationRandom, TrainingGenerationExhaustive
from neural_net.dataset import SequenceDataset
from neural_net.metrics import Metrics
# torch modules
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
# networkx modules
from networkx import DiGraph


class NeuralNetError(Exception):
    pass


def node_representation(node: Node, constraint_types: dict) -> list:
    """
    Returns a node representation

    A node is represented by a (len(language.types) + 1)-int vector of its constraint types and its order (which is
    added in the dfs_component_representation_rec function)
    """
    representation = []
    for constraint_type in constraint_types:
        representation.append(int(constraint_type in node.constraints))
    return representation


def dfs_component_representation_rec(graph: DiGraph, root_node: Node, constraint_types: dict,
                                     visited_nodes: list, representation: list) -> None:
    """
    DFS representation for connected components (recursive)

    WARNING: Assuming that all sibling nodes are connected to their predecessor with an "order" label that is valid
    (numbers that collectively comprise the list [1, 2, ..., num_siblings])
    """
    suc_nodes = sorted(graph.successors(root_node), key=lambda suc_node: graph[root_node][suc_node]["order"])
    for node in suc_nodes:
        if node in visited_nodes:
            continue
        visited_nodes.append(node)
        representation.append(graph[root_node][node]["order"])
        representation.extend(node_representation(node, constraint_types))
        dfs_component_representation_rec(graph, node, constraint_types, visited_nodes, representation)


def dfs_representation(graph: DiGraph, language: Language) -> tuple:
    """
    Returns a Depth First Search representation of a graph

    In order to assure uniqueness of each representation, a hash of the connected components of the graph is generated
    and the order in which the representation of each component is added to the final representation is dictated by the
    order of the hashes
    """
    
    if graph is None:
        return tuple()
    constraint_types = language.types
    components = []
    max_num = float("-inf")
    for node in graph.successors("root_node"):
        # Successors of the root_node are to be considered to have order "0"
        component_representation = [0] + node_representation(node, constraint_types)
        visited_nodes = []
        dfs_component_representation_rec(graph, node, constraint_types, visited_nodes, component_representation)
        max_num = max(max_num, max(component_representation))
        components.append(component_representation)
    components.sort(key=lambda component: hash_graph_representation(component, max_num + 2))
    dfs_tuple = tuple(itertools.chain.from_iterable(components))
    return dfs_tuple


def hash_graph_representation(representation: list, base: int) -> int:
    """
    Returns a unique hash for a connected graph's representation
    """
    graph_hash = 0
    for exponent, i in zip(range(len(representation)), representation):
        graph_hash += (base ** exponent) * (i + 1)
    return graph_hash


def create_rule_representations(language_rules: list[Rule]) -> dict:
    """
    Creates the Rule representations dictionary

    A Rule is represented as a one-hot vector in the domain of a language's rules
    """
    representations = dict()
    for lang_rule_i in language_rules:
        rule_representation = []
        for lang_rule_to_check in language_rules:
            rule_representation.append(int(lang_rule_i is lang_rule_to_check))
        representations[lang_rule_i] = rule_representation
    return representations


def max_len_training_data(data: list[tuple[list[int]]]) -> int:
    """
    Finds the max length of the graph representations in the training data
    """
    max_len = float("-inf")
    for sample in data:
        for repr_tuple in sample:
            curr_max_len = len(max(repr_tuple, key=len))
            max_len = max(curr_max_len, max_len)
    # return max(len(max(repr_tuple, key=len) for repr_tuple in sample) for sample in data)
    return max_len
    
    
def fix_len_training_data(data: list[tuple[tuple[int]]], repr_len: int) -> None:
    """
    Fixes the graph representations in the training data to have a fixed length by padding zeros to the right of 
    each graph representation.
    """
    for i in range(len(data)):
        data[i] = tuple(tuple(dfs_repr + (0,) * (repr_len - len(dfs_repr)) for dfs_repr in graph_tuple) 
                        for graph_tuple in data[i])


def remove_duplicates(data: list[tuple[list[int]]]):
    """
    Removes duplicates from training data
    """
    indices_to_delete = []
    found_samples = dict()
    for i, training_sample in enumerate(data):
        if training_sample in found_samples:
            indices_to_delete.append(i)
        else:
            found_samples[training_sample] = {"index": i}
    for index in indices_to_delete[::-1]:
        # Deleting in reverse order to not shift the position of the actual elements of the rest of the indices
        del data[index]
        
        
class NeuralNet:
    """
    ### Neural network implementation for usage with the Genetic Viewer as its fitness function

    #### Parameters
        ``language``: Language to be used
        ``n_gen``: Number of consecutive rules to be applied to the initial graphs when generating
            the training data
        ``n_items``: Maximum number of items to create initial graphs from when generating the training data
        ``lr``: Learning rate
        ``num_epochs``: Number of epochs to iterate through while training the network
        ``batch_size``: The batch size to use while iterating through the training and testing data
        ``num_classes``: The number of labels for each data instance
        ``device_str``: The name of the device on which to keep the model and do the training
        ``training_generation``: The method to use for training example generation ("random", "exhaustive")
        
    #### Methods
        ``predict``(graph_before, graph_after, graph_final): Returns a prediction on the fitness of the
        (graph_before, graph_after, graph_final) sequence
    """
    def __init__(
                 self,
                 language: Language,
                 n_gen: int = None,
                 n_items: int = None,
                 lr: float = 0.0001, 
                 num_epochs: int = 400,
                 batch_size: int = 2**16,
                 num_classes: int = 1,
                 device_str: str = "cpu",
                 training_generation: str = "random"
                 ) -> None:
        self.language = language
        self.device = torch.device(device_str)
        self.batch_size = batch_size
        self.metric_funcs = Metrics(num_classes=num_classes).metric_funcs
        if training_generation == "random":
            train_gen_obj = TrainingGenerationRandom(n_gen, n_items)
        elif training_generation == "exhaustive":
            train_gen_obj = TrainingGenerationExhaustive(n_gen, n_items)
        else:
            raise NeuralNetError("Training generation argument must be 'random' or 'exhaustive'")
        self._data = train_gen_obj.generate_training_data(language)
        self._prepare_data()
        print("Training data ready, initializing network")
        datasets = self._init_net(lr)
        # self._train(datasets, num_epochs)

    def _init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def _prepare_data(self) -> None:
        """
        Transforms the training graphs to their DFS representations and removes duplicates
        """
        dfs_sample = lambda sample: tuple(tuple(dfs_representation(graph, self.language) for graph in graph_tuple)
                                          for graph_tuple in sample)
        self._data = list(map(dfs_sample, self._data))
        self._graph_len = max_len_training_data(self._data)
        fix_len_training_data(self._data, self._graph_len)
        self._data = [tuple(g_target + g_final for g_target, g_final in sample) for sample in self._data]
        remove_duplicates(self._data)
        
    def _init_net(self, lr: float) -> None:
        """
        Initializes the network for training
        """
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self._graph_len * 2, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),
        ).to(self.device)
        self.model.apply(NeuralNet._init_weights)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        train_size = int(0.99 * len(self._data))
        test_size = len(self._data) - train_size
        self._data = SequenceDataset(self._data)
        train_set, test_set = random_split(self._data, [train_size, test_size])
        # x_val, x_test = train_test_split(x_test, train_size=0.5)
        # validation_set = SequenceDataset(x_val, y_val, device=self.device)
        return {"train": train_set, 
                "test": test_set}
    
    def _calculate_metrics(self, datasets: dict[str, SequenceDataset]):
        """
        Returns the metrics of the model for the given datasets
        """
        metrics = {dataset_str: dict() for dataset_str in datasets}
        for dataset_str, dataset in datasets.items():
            y_hat = self.model(dataset.x)
            for metric_str, metric in self.metric_funcs.items():
                metrics[dataset_str][metric_str] = metric(y_hat, dataset.y)
        return metrics
    
    def _train(self, datasets: dict[str, SequenceDataset], num_epochs: int) -> None:
        """
        Starts training the neural network

        Args:
            ``datasets`` (dict[str, SequenceDataset]): Dictionary mapping the name of the split dataset to the dataset
                    Example: datasets = {"train": train_set,
                                         "validation": validation_set,
                                         "test": test_set}
        """
        train_loader = DataLoader(datasets["train"], batch_size=min(len(datasets["train"]), self.batch_size), shuffle=True)
        # validation_loader = DataLoader(datasets["validation"], batch_size=min(len(datasets["validation"]), self.batch_size), shuffle=False)
        model = self.model
        optimizer = self.optimizer
        metrics_epoch = {dataset_str: {metric : torch.empty(num_epochs, dtype=torch.float32, requires_grad=False)
                                       for metric in self.metric_funcs}
                         for dataset_str in datasets}
        for epoch in range(num_epochs):
            model.train()
            print(f"\rEpoch: {epoch + 1}/{num_epochs}", end='')
            for x, y in train_loader:
                optimizer.zero_grad()
                y_hat = model(x)
                loss = self.metric_funcs["loss"](y_hat, y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                for dataset_str, metrics in self._calculate_metrics(datasets).items():
                    for metric_str, metric in metrics.items():
                        metrics_epoch[dataset_str][metric_str][epoch] = metric
        print()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.metrics_epoch = metrics_epoch 
        
    
    def predict(self, graph_before: DiGraph, graph_after: DiGraph, graph_final: DiGraph) -> float:
        representation = []
        graphs = [graph_before, graph_after, graph_final]
        for graph in graphs:
            graph_repr = dfs_representation(graph, self.language)
            if len(graph_repr) > self._graph_len:
                raise NeuralNetError(f"Graph {graph} has more than allowed nodes ({len(graph_repr)}"\
                                     f", maximum allowed are {self._graph_len})")
            representation.extend(graph_repr + (self._graph_len - len(graph_repr)) * (0,))
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(representation, dtype=torch.float32, device=self.device)).item()
