"""
Neural network implementation module
"""

# pymeleon modules
from language.language import Language
from neural_net.training_generation import TrainingGeneration, TrainingGenerationRandom, dfs_representation
from neural_net.dataset import SequenceDataset
from neural_net.metrics import Metrics
# torch modules
import torch
from torch.utils.data import DataLoader
# scikit-learn modules
from sklearn.model_selection import train_test_split
# networkx modules
from networkx import DiGraph


class NeuralNetError(Exception):
    pass


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
        ``training_generation_class``: The class to use for training example generation
        
    #### Methods
        ``predict``(graph_before, graph_after, graph_final): Returns a prediction on the fitness of the
        (graph_before, graph_after, graph_final) sequence
    """
    def __init__(
                 self,
                 language: Language,
                 n_gen: int = 5,
                 n_items: int = None,
                 lr: float = 0.0001, 
                 num_epochs: int = 400,
                 batch_size: int = 2**16,
                 num_classes: int = 1,
                 device_str: str = "cpu",
                 training_generation_class: TrainingGeneration = TrainingGenerationRandom
                 ) -> None:
        self.language = language
        train_gen_obj = training_generation_class()
        self._data, self._labels, self._input_len = train_gen_obj.generate_training_examples(language, n_gen, n_items)
        self.device = torch.device(device_str)
        self.batch_size = batch_size
        self.metric_funcs = Metrics(num_classes=num_classes).metric_funcs
        datasets = self._prepare_for_training(lr)
        self._train(datasets, num_epochs)

    def _init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def _prepare_for_training(self, lr: float) -> None:
        """
        Generates training examples and initializes the network for training
        """
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self._input_len * 3, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid()
        ).to(self.device)
        self.model.apply(NeuralNet._init_weights)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        x_train, x_test, y_train, y_test = train_test_split(self._data, self._labels, train_size=0.8)
        # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5)
        train_set = SequenceDataset(x_train, y_train, device=self.device)
        test_set = SequenceDataset(x_test, y_test, device=self.device)
        if ((train_set.y.all() or not train_set.y.any()) or
            (test_set.y.all() or not test_set.y.any())):
            raise NeuralNetError("Only 1 label exists in either the training or the test set")
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
            # print(f"\rEpoch: {epoch + 1}/{num_epochs}", end='')
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
        # print()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.metrics_epoch = metrics_epoch 
        
    
    def predict(self, graph_before: DiGraph, graph_after: DiGraph, graph_final: DiGraph) -> float:
        representation = []
        graphs = [graph_before, graph_after, graph_final]
        for graph in graphs:
            graph_repr = dfs_representation(graph, self.language)
            if len(graph_repr) > self._input_len:
                raise NeuralNetError(f"Graph {graph} has more than allowed nodes ({len(graph_repr)}"\
                                     f", maximum allowed are {self._input_len})")
            representation.extend(graph_repr + (self._input_len - len(graph_repr)) * [0])
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(representation, dtype=torch.float32, device=self.device)).item()
