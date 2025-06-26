import torch
from torch.utils.data import Dataset
import numpy as np

class HarmonicDataset(Dataset):
    """Generate multi-dimensional harmonic signals."""

    def __init__(self, num_samples=1000, seq_len=1024, channels=2,
                 num_harmonics=3, freq_range=(50, 500), sample_rate=2000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.channels = channels
        self.num_harmonics = num_harmonics
        self.freq_range = freq_range
        self.sample_rate = sample_rate
        self.t = torch.linspace(0, seq_len / sample_rate, seq_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        signal = torch.zeros(self.channels, self.seq_len)
        for c in range(self.channels):
            for _ in range(self.num_harmonics):
                freq = np.random.uniform(*self.freq_range)
                phase = np.random.uniform(0, 2 * np.pi)
                amp = np.random.uniform(0.5, 1.0)
                signal[c] += amp * torch.sin(2 * np.pi * freq * self.t + phase)
        return signal
