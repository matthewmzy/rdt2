# ...existing code...
import torch
import numpy as np
from typing import Optional, Union


class LinearNormalizer:
    """
    Minimal, flexible linear normalizer that supports loading common checkpoint key names.
    It looks for pairs like:
      - "mean" / "std"
      - "action_mean" / "action_std"
      - "state_mean" / "state_std"
    If not found, you can pass explicit mean/std arrays when constructing.

    Methods accept torch.Tensor or numpy.ndarray and return the same type as input.
    """
    def __init__(self, mean: Optional[Union[torch.Tensor, np.ndarray]] = None,
                 std: Optional[Union[torch.Tensor, np.ndarray]] = None,
                 eps: float = 1e-8):
        self.eps = eps
        self.mean = None
        self.std = None
        if mean is not None:
            self.mean = torch.as_tensor(mean) if not isinstance(mean, torch.Tensor) else mean.clone()
        if std is not None:
            self.std = torch.as_tensor(std) if not isinstance(std, torch.Tensor) else std.clone()

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x_t = torch.from_numpy(x)
        else:
            x_t = x
        if self.mean is None or self.std is None:
            # no-op if no stats available
            return x if not is_numpy else x.copy()
        out = (x_t - self.mean) / (self.std + self.eps)
        return out.numpy() if is_numpy else out

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x_t = torch.from_numpy(x)
        else:
            x_t = x
        if self.mean is None or self.std is None:
            return x if not is_numpy else x.copy()
        out = x_t * (self.std + self.eps) + self.mean
        return out.numpy() if is_numpy else out

    def to(self, device):
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self

    @staticmethod
    def _find_pair(ckpt: dict, prefix_opts=('action', 'state', '')):
        # return tuple (mean, std, prefix_used) where mean/std are tensors or (None,None, None)
        keys = set(ckpt.keys())
        candidates = []
        for p in prefix_opts:
            if p == '':
                mean_k = 'mean'
                std_k = 'std'
            else:
                mean_k = f'{p}_mean'
                std_k = f'{p}_std'
            if mean_k in keys and std_k in keys:
                candidates.append((mean_k, std_k, p or 'mean'))
        # prefer 'action' if present (common), else first match
        for m, s, p in candidates:
            if 'action' in p:
                return ckpt[m], ckpt[s], p
        if candidates:
            m, s, p = candidates[0]
            return ckpt[m], ckpt[s], p
        # fallback: accept single 'mean'/'std' if present
        if 'mean' in keys and 'std' in keys:
            return ckpt['mean'], ckpt['std'], 'mean'
        return None, None, None

    @classmethod
    def from_checkpoint(cls, path: str):
        """
        Load checkpoint and attempt to extract mean/std automatically.
        If multiple candidate pairs exist, prioritizes 'action' pair.
        """
        ckpt = torch.load(path, map_location='cpu')
        if not isinstance(ckpt, dict):
            raise ValueError("Loaded object is not a dict. Got type: {}".format(type(ckpt)))
        mean, std, prefix = cls._find_pair(ckpt)
        if mean is None or std is None:
            # Could be that keys are nested, so do a shallow search of tensor-like entries
            # Print keys for user to inspect and raise helpful error.
            raise ValueError(
                "No mean/std pair found in checkpoint. Available keys:\n  {}".format(
                    "\n  ".join(f"{k}: {type(v)}{getattr(v,'shape', '')}" for k, v in ckpt.items())
                )
            )
        # ensure tensors
        mean_t = mean.clone() if isinstance(mean, torch.Tensor) else torch.as_tensor(mean)
        std_t = std.clone() if isinstance(std, torch.Tensor) else torch.as_tensor(std)
        inst = cls(mean=mean_t, std=std_t)
        return inst

    @staticmethod
    def inspect(path: str, top_n: int = 10):
        """
        Print a compact summary of checkpoint keys, types, shapes and basic stats.
        """
        ckpt = torch.load(path, map_location='cpu')
        print(f"Loaded checkpoint: {path}")
        if isinstance(ckpt, dict):
            for k, v in ckpt.items():
                typ = type(v)
                shape = getattr(v, 'shape', None)
                s = f"{k} : {typ.__name__}"
                if shape is not None:
                    s += f" shape={tuple(shape)}"
                    if isinstance(v, torch.Tensor) and v.numel() > 0:
                        arr = v.detach().cpu().float().flatten()
                        # compute a few stats
                        vals = arr.numpy()
                        s += f" min={vals.min():.6g} max={vals.max():.6g} mean={vals.mean():.6g} std={vals.std():.6g}"
                print(s)
        else:
            # tensor or other object
            typ = type(ckpt)
            print(f"Checkpoint is a {typ.__name__} of type; value summary not printed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect or load a normalizer .pt")
    parser.add_argument("--path", default="normalizer.pt", help="Path to normalizer .pt")
    parser.add_argument("--inspect-only", action="store_true", help="Only print keys/stats and exit")
    args = parser.parse_args()

    try:
        LinearNormalizer.inspect(args.path)
        if not args.inspect_only:
            try:
                norm = LinearNormalizer.from_checkpoint(args.path)
                print("\nAuto-loaded mean/std shapes:")
                print(" mean.shape =", tuple(norm.mean.shape) if norm.mean is not None else None)
                print(" std.shape  =", tuple(norm.std.shape) if norm.std is not None else None)
            except Exception as e:
                print("\nCould not auto-load mean/std pair:", e)
    except Exception as e:
        print("Error loading checkpoint:", e)