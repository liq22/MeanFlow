import torch
import torch.nn as nn

class VibrationNet(nn.Module):
    """Simple 1D convolutional network for time-series modeling."""
    def __init__(self, channels=2, hidden_dim=64, num_layers=4, num_classes=None):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes

        layers = []
        in_ch = channels + 2  # for t and r concatenation
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_ch, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_ch = hidden_dim
        layers.append(nn.Conv1d(hidden_dim, channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes + 1, 1)
        else:
            self.label_emb = None

    def forward(self, x, t, r, y=None):
        bs, c, l = x.shape
        t_feat = t.view(bs,1,1).expand(bs,1,l)
        r_feat = r.view(bs,1,1).expand(bs,1,l)
        inp = torch.cat([x, t_feat, r_feat], dim=1)
        out = self.net(inp)
        if self.label_emb is not None and y is not None:
            cond = self.label_emb(y).view(bs,1,1)
            out = out + cond
        return out
