import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from harmonic_dataset import HarmonicDataset
from model_1d import MFDiT1D
from meanflow_1d import MeanFlow1D


if __name__ == '__main__':
    steps = 1000
    batch_size = 16
    seq_len = 1024

    accelerator = Accelerator()

    dataset = HarmonicDataset(num_samples=10000, seq_len=seq_len, channels=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader = iter(dataloader)

    model = MFDiT1D(input_size=seq_len, patch_size=16, in_channels=2, dim=256, depth=6, num_heads=8, num_classes=None)
    meanflow = MeanFlow1D(channels=2, signal_length=seq_len, num_classes=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model, optimizer = accelerator.prepare(model, optimizer)

    for step in tqdm(range(steps), desc='training'):
        x = next(dataloader).to(accelerator.device)
        loss, _ = meanflow.loss(model, x)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 200 == 0 and accelerator.is_main_process:
            with torch.no_grad():
                samples = meanflow.sample(model, 4, device=accelerator.device)
                torch.save(samples.cpu(), f'samples_step_{step+1}.pt')
