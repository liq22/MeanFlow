import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np

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
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == 'minmax':
            x = x.clip(-1,1)
            return (x + 1) * 0.5
        return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()


class MeanFlow1D:
    def __init__(
        self,
        channels=2,
        seq_len=1024,
        num_classes=None,
        normalizer=['minmax', None, None],
        flow_ratio=0.50,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=None,
        cfg_scale=None,
        cfg_uncond='u',
        jvp_api='autograd',
    ):
        self.channels = channels
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.use_cond = num_classes is not None

        self.normer = Normalizer.from_list(normalizer)

        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale

        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api

        assert jvp_api in ['funtorch', 'autograd']
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        else:
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        else:
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))
        t_np = np.maximum(samples[:,0], samples[:,1])
        r_np = np.minimum(samples[:,0], samples[:,1])
        num_selected = int(self.flow_ratio * batch_size)
        idx = np.random.permutation(batch_size)[:num_selected]
        r_np[idx] = t_np[idx]
        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def loss(self, model, x, c=None):
        batch_size = x.shape[0]
        device = x.device
        t, r = self.sample_t_r(batch_size, device)
        t_ = t.view(batch_size,1,1).detach().clone()
        r_ = r.view(batch_size,1,1).detach().clone()
        e = torch.randn_like(x)
        x = self.normer.norm(x)
        z = (1 - t_) * x + t_ * e
        v = e - x
        if c is not None:
            assert self.cfg_ratio is not None
            uncond = torch.ones_like(c) * self.num_classes
            cfg_mask = torch.rand_like(c.float()) < self.cfg_ratio
            c = torch.where(cfg_mask, uncond, c)
            if self.w is not None:
                with torch.no_grad():
                    u_t = model(z, t, t, uncond)
                v_hat = self.w * v + (1 - self.w) * u_t
                if self.cfg_uncond == 'v':
                    cfg_mask = cfg_mask.view(batch_size,1,1).bool()
                    v_hat = torch.where(cfg_mask, v, v_hat)
            else:
                v_hat = v
        else:
            v_hat = v
        model_partial = partial(model, y=c)
        jvp_args = (
            lambda z,t,r: model_partial(z,t,r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )
        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)
        u_tgt = v_hat - (t_-r_) * dudt
        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        mse_val = (stopgrad(error) ** 2).mean()
        return loss, mse_val

    @torch.no_grad()
    def sample(self, model, n_samples, sample_steps=5, device='cuda'):
        model.eval()
        z = torch.randn(n_samples, self.channels, self.seq_len, device=device)
        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        for i in range(sample_steps):
            t = torch.full((n_samples,), t_vals[i], device=device)
            r = torch.full((n_samples,), t_vals[i+1], device=device)
            t_ = t.view(n_samples,1,1)
            r_ = r.view(n_samples,1,1)
            v = model(z, t, r)
            z = z - (t_-r_) * v
        z = self.normer.unnorm(z)
        return z
