"""
Neural network implementation module
"""

import itertools
import pathlib
# pymeleon modules
from language.language import Language
from language.parser import Node
from language.rule import Rule
from neural_net.training_generation import TrainingGenerationRandom, TrainingGenerationExhaustive
from neural_net.dataset import SequenceDataset
from neural_net.metrics import Metrics
from neural_net import pretrained_models_path
# torch modules
import torch
from torch.utils.data import DataLoader, random_split
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
    Neural network implementation for usage with the Genetic Viewer as its fitness function

    Parameters
        ``language``: Language to be used.
        ``hyperparams``: Dictionary holding any of the following hyperparameters
            ``n_gen``: Number of consecutive rules to be applied to the initial graphs when generating \
                the training data. Defaults to 20.
            ``n_items``: Maximum number of items to create initial graphs from when generating the training \
                data. Defaults to len(language.types).
            ``lr``: Learning rate. Defaults to 0.0001.
            ``num_epochs``: Number of epochs to iterate through while training the network. Defaults to 1000.
            ``batch_size``: The batch size to use while iterating through the training and testing data. \
                Defaults to 2**16.
        ``device_str``: The name of the device on which to keep the model and do the training
        ``training_generation``: The method to use for training example generation ("random", "exhaustive")
        
    Methods
        ``predict``(graph_before, graph_after, graph_final): Returns a prediction on the fitness of the
        (graph_before, graph_after, graph_final) sequence
    """
    DEFAULT_HYPERPARAMS = {
        "n_gen": 20,
        "n_items": None, # Initialized to len(language.types) for each NeuralNet instance
        "lr": 0.0001,
        "prev_reg": 0.1,
        "num_epochs": 1000,
        "batch_size": 2**16,
        "weight_decay": 1.e-4
    }
    
    def __init__(
                 self,
                 language: Language,
                 hyperparams: dict = None,
                 device_str: str = "cpu",
                 training_generation: str = "random",
                 use_pretrained: bool = True
                 ) -> None:
        self.language = language
        self.device = torch.device(device_str)
        model_path = pretrained_models_path / f"__pymeleon_pretrained_model_{self.language.name}.pt"
        if use_pretrained:
            if self.load_pretrained_model(model_path):
                return
        if hyperparams is None:
            hyperparams = dict()
        self.hyperparams = NeuralNet.DEFAULT_HYPERPARAMS | {"n_items": len(language.types)} | hyperparams
        self.metric_funcs = Metrics(loss_func=self.loss_function).metric_funcs
        if training_generation == "random":
            train_gen_obj = TrainingGenerationRandom(self.hyperparams["n_gen"], 
                                                     self.hyperparams["n_items"])
        elif training_generation == "exhaustive":
            train_gen_obj = TrainingGenerationExhaustive(self.hyperparams["n_gen"], 
                                                         self.hyperparams["n_items"])
        else:
            raise NeuralNetError("Training generation argument must be 'random' or 'exhaustive'")
        data = self._prepare_data(train_gen_obj.generate_training_data(language))
        dataloaders = self._init_net(data)
        self._train(dataloaders, model_path)

    def load_pretrained_model(self, model_path: pathlib.Path):
        if self.language.name == "default_lang__pym":
            print("WARNING: Loading pretrained model for default language name")
            if model_path.is_file():
                self.model = torch.load(model_path)
                self._graph_len = self.model[0].in_features // 2
                return True

    def loss_function(self, 
                      data_tensors: torch.Tensor
                      ) -> torch.Tensor:
        """
        Loss function for the neural network, discriminator
        """
        before_tensor = data_tensors[:, 0]
        after_tensor = data_tensors[:, 1]
        neg_tensor = data_tensors[:, 2]
        after_pred = self.model(after_tensor)
        loss = (torch.sigmoid(self.model(neg_tensor) - after_pred) + 
                self.hyperparams["prev_reg"] * torch.sigmoid((self.model(before_tensor) - after_pred)))
        return loss.sum()
        
    def _init_weights(m) -> None:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def _prepare_data(self, data: list) -> list:
        """
        Transforms the training graphs to their DFS representations
        """
        print(f"\rPreparing data for training", end="")
        dfs_sample = lambda sample: tuple(tuple(dfs_representation(graph, self.language) for graph in graph_tuple)
                                          for graph_tuple in sample)
        data = list(map(dfs_sample, data))
        self._graph_len = max_len_training_data(data)
        fix_len_training_data(data, self._graph_len)
        data = [tuple(g_target + g_final for g_target, g_final in sample) for sample in data]
        # remove_duplicates(data)
        print(f"\r{' ' * 60}", end="")
        return data
        
    def _init_net(self, data: list) -> dict[str, DataLoader]:
        """
        Initializes the network for training
        """
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self._graph_len * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        ).to(self.device)
        self.model.apply(NeuralNet._init_weights)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), 
                                          lr=self.hyperparams["lr"], 
                                          weight_decay=self.hyperparams["weight_decay"])
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        data = SequenceDataset(data, device=self.device)
        train_set, test_set = random_split(data, [train_size, test_size])
        # x_val, x_test = train_test_split(x_test, train_size=0.5)
        # validation_set = SequenceDataset(x_val, y_val, device=self.device)
        batch_size = self.hyperparams["batch_size"]
        return {"train": DataLoader(train_set, batch_size=min(train_size, batch_size), shuffle=True), 
                "test": DataLoader(test_set, batch_size=min(test_size, batch_size), shuffle=False)}
    
    def _calculate_metrics(self, dataloaders: dict[str, DataLoader]):
        """
        Returns the metrics of the model for the given dataloaders
        """
        metrics = {dataset_str: {metric_str: 0 for metric_str in self.metric_funcs}
                   for dataset_str in dataloaders}
        for dataset_str, dataloader in dataloaders.items():
            for metric_str, metric in self.metric_funcs.items():
                for data in dataloader:
                    metrics[dataset_str][metric_str] += metric(data)
                metrics[dataset_str][metric_str] /= len(dataloader)
        return metrics
    
    def _train(self, dataloaders: dict[str, DataLoader], model_path: pathlib.Path) -> None:
        """
        Trains the neural network and saves the trained model

        Args:
            ``dataloaders`` (dict[str, DataLoader]): Dictionary mapping the name of the split dataset to the dataloader
                    Example: dataloaders = {"train": train_loader,
                                            "validation": validation_loader,
                                            "test": test_loader}
        """
        num_epochs = self.hyperparams["num_epochs"]
        model = self.model
        optimizer = self.optimizer
        metrics_epoch = {dataset_str: {metric : torch.zeros(num_epochs, dtype=torch.float32, requires_grad=False, device=self.device)
                                       for metric in self.metric_funcs}
                         for dataset_str in dataloaders}
        for epoch in range(num_epochs):
            model.train()
            print(f"\rEpoch: {epoch + 1}/{num_epochs}", end="")
            for data_tensors in dataloaders["train"]:
                optimizer.zero_grad()
                loss = self.metric_funcs["loss"](data_tensors)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                metrics_epoch["train"]["loss"][epoch] = loss
                for test_data in dataloaders["test"]:
                    metrics_epoch["test"]["loss"][epoch] += self.metric_funcs["loss"](test_data)
                metrics_epoch["test"]["loss"][epoch] /= len(dataloaders["test"])
            # with torch.no_grad():
            #     for dataset_str, metrics in self._calculate_metrics(dataloaders).items():
            #         for metric_str, metric in metrics.items():
            #             metrics_epoch[dataset_str][metric_str][epoch] = metric
        print(f"\rTraining complete {' ' * 50}", end="")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.model = self.model.to(device=torch.device("cpu"))
        self.metrics_epoch = metrics_epoch 
        torch.save(self.model, model_path)
    
    def predict(self, graph_after: DiGraph, graph_final: DiGraph) -> float:
        representation = []
        graphs = [graph_after, graph_final]
        for graph in graphs:
            graph_repr = dfs_representation(graph, self.language)
            if len(graph_repr) > self._graph_len:
                raise NeuralNetError(f"Graph {graph} has more than allowed nodes ({len(graph_repr)}"\
                                     f", maximum allowed are {self._graph_len})")
            representation.extend(graph_repr + (self._graph_len - len(graph_repr)) * (0,))
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(representation, dtype=torch.float32)).item()
