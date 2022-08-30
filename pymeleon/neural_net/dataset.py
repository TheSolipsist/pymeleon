from torch.utils.data import Dataset
import torch


class SequenceDataset(Dataset):
    """
    Custom Dataset class inheriting the PyTorch dataset class for usage with graph representation sequences
    """
    def __init__(self, data, device=None):
        if device is None:
            device = torch.device("cpu")
        self.data = torch.tensor(data, dtype=torch.float32, device=device).reshape(len(data), len(data[0]), len(data[0][0]))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
