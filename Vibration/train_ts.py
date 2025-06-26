import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ts_dit import TSDiT
from meanflow_ts import MeanFlowTS
from signal_dataset import HarmonicDataset


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = HarmonicDataset(length=1024, n_samples=10000, n_channels=2)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = TSDiT(input_size=1024, patch_size=16, in_channels=2, dim=256, depth=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    meanflow = MeanFlowTS(channels=2, seq_len=1024)

    for step in tqdm(range(1000)):
        x = next(iter(dataloader)).to(device)
        loss, mse = meanflow.loss(model, x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} mse {mse.item():.4f}")
    torch.save(model.state_dict(), 'ts_model.pt')


if __name__ == '__main__':
    train()
