import torch
from torch.utils.data import Dataset
import math
import random

class HarmonicDataset(Dataset):
    """Generate synthetic multi-dimensional harmonic signals."""
    def __init__(self, length=1024, num_samples=10000, freq_range=(50, 100)):
        self.length = length
        self.num_samples = num_samples
        self.freq_range = freq_range
        self.t = torch.linspace(0, 1, length)

    def __len__(self):
        return self.num_samples

    def _gen_signal(self):
        f1 = random.uniform(*self.freq_range)
        f2 = random.uniform(*self.freq_range)
        a1 = random.uniform(0.5, 1.5)
        a2 = random.uniform(0.5, 1.5)
        p1 = random.uniform(0, 2*math.pi)
        p2 = random.uniform(0, 2*math.pi)
        s1 = a1 * torch.sin(2*math.pi*f1*self.t + p1)
        s2 = a2 * torch.sin(2*math.pi*f2*self.t + p2)
        return torch.stack([s1, s2], dim=0)

    def __getitem__(self, idx):
        signal = self._gen_signal()
        return signal.float(), 0
