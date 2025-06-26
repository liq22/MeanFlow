import torch
from torch.utils.data import Dataset
import numpy as np

class HarmonicDataset(Dataset):
    """Generate multi-dimensional harmonic sequences."""
    def __init__(self, length=1024, n_samples=10000, freq_range=(5,50), n_channels=2, sample_rate=200):
        self.length = length
        self.n_samples = n_samples
        self.freq_range = freq_range
        self.n_channels = n_channels
        self.sample_rate = sample_rate

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t = np.arange(self.length) / self.sample_rate
        freqs = np.random.uniform(*self.freq_range, size=self.n_channels)
        phases = np.random.uniform(0, 2*np.pi, size=self.n_channels)
        signal = [np.sin(2*np.pi*f*t + p) for f,p in zip(freqs, phases)]
        signal = np.stack(signal, axis=0).astype(np.float32)
        return torch.from_numpy(signal)
