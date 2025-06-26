import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from harmonic_dataset import HarmonicDataset
from vibration_model import VibrationNet
from meanflow_1d import MeanFlow1D


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = HarmonicDataset(length=1024, num_samples=2000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = VibrationNet(channels=2, hidden_dim=64).to(device)
    meanflow = MeanFlow1D(channels=2, seq_len=1024)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in tqdm(range(1000)):
        x, _ = next(iter(loader))
        x = x.to(device)
        loss, _ = meanflow.loss(model, x)
        loss.backward()
        optim.step()
        optim.zero_grad()
        if (step + 1) % 200 == 0:
            print('step', step+1, 'loss', loss.item())

    samples = meanflow.sample(model, 4, device=device)
    print('sample shape', samples.shape)


if __name__ == '__main__':
    main()
