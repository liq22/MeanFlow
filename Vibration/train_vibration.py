import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from harmonic_dataset import HarmonicDataset
from time_dit import TimeDiT
from meanflow_time import TimeMeanFlow


def main():
    length = 512
    channels = 2
    batch_size = 32
    steps = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = HarmonicDataset(length=length, channels=channels, num_samples=10000)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_iter = iter(loader)

    model = TimeDiT(input_size=length, patch_size=1, in_channels=channels,
                    dim=256, depth=6, num_heads=8, num_classes=None).to(device)
    meanflow = TimeMeanFlow(channels=channels, length=length, num_classes=None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    pbar = tqdm(range(steps))
    for step in pbar:
        try:
            x = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x = next(loader_iter)
        x = x.to(device)
        loss, _ = meanflow.loss(model, x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f"loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'vibration_model.pt')
    samples = meanflow.sample(model, num_samples=4, device=device)
    torch.save(samples.cpu(), 'vibration_samples.pt')


if __name__ == '__main__':
    main()
