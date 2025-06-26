import numpy as np
import torch
from torch.utils.data import Dataset

class HarmonicDataset(Dataset):
    """Dataset generating multi-dimensional harmonic time series."""
    def __init__(self, length=512, channels=2, num_samples=10000, num_components=3,
                 freq_range=(1.0, 20.0)):
        self.length = length
        self.channels = channels
        self.num_samples = num_samples
        self.num_components = num_components
        self.freq_range = freq_range
        self.t = np.linspace(0, 1, length, endpoint=False)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        signal = np.zeros((self.channels, self.length), dtype=np.float32)
        for c in range(self.channels):
            for _ in range(self.num_components):
                freq = np.random.uniform(*self.freq_range)
                amp = np.random.uniform(0.5, 1.0)
                phase = np.random.uniform(0, 2 * np.pi)
                signal[c] += amp * np.sin(2 * np.pi * freq * self.t + phase)
        return torch.from_numpy(signal)
