import torch
import torch.nn.functional as F
from functools import partial
import numpy as np


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    delta_sq = torch.mean(error ** 2, dim=(1, 2))
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()


class Normalizer:
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std']
        self.mode = mode
        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError('mean and std must be provided for mean_std mode')
            self.mean = torch.tensor(mean).view(-1, 1)
            self.std = torch.tensor(std).view(-1, 1)

    @classmethod
    def from_list(cls, config):
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        else:
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == 'minmax':
            x = x.clip(-1, 1)
            return (x + 1) * 0.5
        else:
            return x * self.std.to(x.device) + self.mean.to(x.device)


class MeanFlowTS:
    def __init__(self, channels=2, seq_len=1024, normalizer=['minmax', None, None],
                 flow_ratio=0.5, time_dist=['lognorm', -0.4, 1.0]):
        self.channels = channels
        self.seq_len = seq_len
        self.normer = Normalizer.from_list(normalizer)
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.jvp_fn = torch.autograd.functional.jvp

    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        else:
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def loss(self, model, x):
        batch_size = x.shape[0]
        device = x.device
        t, r = self.sample_t_r(batch_size, device)
        t_ = t.view(-1, 1, 1)
        r_ = r.view(-1, 1, 1)
        e = torch.randn_like(x)
        x = self.normer.norm(x)
        z = (1 - t_) * x + t_ * e
        v = e - x
        model_partial = partial(model, y=None)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v, torch.ones_like(t), torch.zeros_like(r)),
        )
        u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        u_tgt = v - (t_ - r_) * dudt
        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        mse_val = (stopgrad(error) ** 2).mean()
        return loss, mse_val

    @torch.no_grad()
    def sample(self, model, n_samples=1, sample_steps=5, device='cuda'):
        model.eval()
        z = torch.randn(n_samples, self.channels, self.seq_len, device=device)
        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i+1], device=device)
            t_ = t.view(-1,1,1)
            r_ = r.view(-1,1,1)
            v = model(z, t, r, None)
            z = z - (t_-r_) * v
        z = self.normer.unnorm(z)
        return z
