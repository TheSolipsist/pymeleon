from torch.utils.data import Dataset
import torch


class SequenceDataset(Dataset):
    """
    Custom Dataset class inheriting the PyTorch dataset class for usage with graph representation sequences
    """
    def __init__(self, data, labels, device=None):
        if device is None:
            device = torch.device("cpu")
        self.x = torch.tensor(data, dtype=torch.float32, device=device).reshape(len(data), len(data[0]))
        self.y = torch.tensor(labels, dtype=torch.float32, device=device).reshape(len(labels), 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
