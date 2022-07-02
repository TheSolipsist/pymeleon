"""
Neural network implementation module
"""
from neural_net.training_generation import generate_training_examples
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from neural_net.dataset import SequenceDataset


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)


class NeuralNet:
    """
    Neural network implementation for usage with the Genetic Viewer as its fitness function

    -- Parameters --
        language: Language to be used
        n_gen: Number of consecutive rules to be applied to the initial graphs when generating
            the training data

    -- Methods --
        predict(representation): Returns a prediction on the fitness of the Graph, Rule, Graph sequence
    """
    def __init__(self, language, n_gen, n_items=None, lr=0.01, num_epochs=3000, device_str=None):
        self.language = language
        self.n_gen = n_gen
        self.n_items = n_items
        self.lr = lr
        self.num_epochs = num_epochs
        if device_str is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_str)
        self._data, self._labels, self._max_length_dict = generate_training_examples(language, n_gen, n_items)
        input_len = len(self._data[0])
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_len, input_len // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(input_len // 2, 1),
            torch.nn.Sigmoid()
        )
        self._train()

    def _train(self):
        """
        Starts training the neural network on the generated training sample
        """
        x_train, x_test, y_train, y_test = train_test_split(self._data, self._labels, train_size=0.8)
        # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5)
        train_loader = DataLoader(SequenceDataset(x_train, y_train, device=self.device),
                                  batch_size=min(len(y_train), 1024), shuffle=True)
        # validation_loader = DataLoader(SequenceDataset(x_val, y_val, device=self.device),
        #                                batch_size=min(len(y_val), 1024), shuffle=False)
        test_loader = DataLoader(SequenceDataset(x_test, y_test, device=self.device),
                                 batch_size=min(len(y_test), 1024), shuffle=False)

        net = self.net
        net.to(self.device)
        net.apply(init_weights)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(params=net.parameters(), lr=self.lr)

        net.train()
        for epoch in range(self.num_epochs):
            print(f"\rEpoch: {epoch + 1}/{self.num_epochs}", end='')
            for x, y in train_loader:
                optimizer.zero_grad()
                y_hat = net(x)
                loss = criterion(y_hat.squeeze().to(torch.float32), y.squeeze().to(torch.float32))
                loss.backward()
                optimizer.step()
        print()

        net.eval()
        with torch.no_grad():
            correct_predictions = 0
            total_predictions = 0
            losses = []
            for x, y in train_loader:
                y_hat = net(x)
                losses.append(criterion(y_hat.squeeze().to(torch.float32), y.squeeze().to(torch.float32)))
                predictions = (y_hat.squeeze() > 0.5) == y.squeeze()
                correct_predictions += predictions.sum()
                total_predictions += predictions.numel()
        training_loss = sum(losses) / len(losses)
        training_accuracy = correct_predictions / total_predictions
        print(f"Training set loss: {training_loss}")
        print(f"Training set accuracy: {training_accuracy}")

        with torch.no_grad():
            correct_predictions = 0
            total_predictions = 0
            losses = []
            for x, y in test_loader:
                y_hat = net(x)
                losses.append(criterion(y_hat.squeeze().to(torch.float32), y.squeeze().to(torch.float32)))
                predictions = (y_hat.squeeze() > 0.5) == y.squeeze()
                correct_predictions += predictions.sum()
                total_predictions += predictions.numel()
        test_loss = sum(losses) / len(losses)
        test_accuracy = correct_predictions / total_predictions
        print(f"Test set loss: {test_loss}")
        print(f"Test set accuracy: {test_accuracy}")

    def predict(self, representation):
        return self.net(representation)


