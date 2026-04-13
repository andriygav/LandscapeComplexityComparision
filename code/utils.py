from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import hessian as func_hessian
from torch.func import vmap


def configure_torch_cpu(num_threads: Optional[int] = None) -> None:
    if num_threads is None:
        num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
    torch.set_num_threads(max(1, num_threads))


configure_torch_cpu()


def figures_dir() -> Path:
    cwd = Path.cwd().resolve()
    root = cwd.parent if cwd.name == "code" else cwd
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def mu_f_linear(X: np.ndarray) -> float:
    m, n = X.shape
    if m == 0:
        return float("nan")
    h_bar = np.zeros((n, n), dtype=np.float64)
    for i in range(m):
        h_bar += np.outer(X[i], X[i])
    h_bar /= m
    total = 0.0
    for i in range(m):
        diff = np.outer(X[i], X[i]) - h_bar
        total += np.linalg.norm(diff, ord=2)
    return total / m


def rademacher_linear_empirical(
    X: np.ndarray,
    W: float,
    n_mc: int = 4096,
    rng: Optional[np.random.Generator] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng()
    m, _ = X.shape
    acc = 0.0
    for _ in range(n_mc):
        sigma = rng.choice([-1.0, 1.0], size=m)
        svec = (sigma[:, None] * X).sum(axis=0)
        acc += np.linalg.norm(svec, ord=2)
    return (W / m) * (acc / n_mc)


def twolayer_forward(theta: torch.Tensor, x: torch.Tensor, n: int, h: int) -> torch.Tensor:
    hn = h * n
    W1 = theta[:hn].view(h, n)
    w2 = theta[hn:]
    z = F.relu(W1 @ x)
    return (w2 * z).sum()


def twolayer_forward_batch(
    theta: torch.Tensor, X: torch.Tensor, n: int, h: int
) -> torch.Tensor:
    hn = h * n
    W1 = theta[:hn].view(h, n)
    w2 = theta[hn:]
    Z = F.relu(X @ W1.T)
    return (Z * w2.unsqueeze(0)).sum(dim=1)


def _per_example_hessians_twolayer_vmap(
    theta: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    n: int,
    h: int,
) -> torch.Tensor:
    def ell_i(th: torch.Tensor, xi: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
        pred = twolayer_forward(th, xi, n, h)
        return 0.5 * (pred - yi) ** 2

    hess_one = func_hessian(ell_i)
    H = vmap(hess_one, in_dims=(None, 0, 0))(theta, X, y)
    return 0.5 * (H + H.transpose(-1, -2))


def mu_f_twolayer(
    theta: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    n: int,
    h: int,
    *,
    hessian_chunk_size: int = 1024,
) -> float:
    m = X.shape[0]
    H_sum = None
    for start in range(0, m, hessian_chunk_size):
        end = min(start + hessian_chunk_size, m)
        H_chunk = _per_example_hessians_twolayer_vmap(theta, X[start:end], y[start:end], n, h)
        chunk_sum = H_chunk.sum(dim=0)
        H_sum = chunk_sum if H_sum is None else H_sum + chunk_sum
    assert H_sum is not None
    H_bar = H_sum / m
    norm_sum = 0.0
    for start in range(0, m, hessian_chunk_size):
        end = min(start + hessian_chunk_size, m)
        H_chunk = _per_example_hessians_twolayer_vmap(theta, X[start:end], y[start:end], n, h)
        diff = H_chunk - H_bar.unsqueeze(0)
        norms = torch.linalg.matrix_norm(diff, ord=2, dim=(-2, -1))
        norm_sum += float(norms.sum().item())
    return norm_sum / m


def twolayer_empirical_mse_grad_norm(
    theta: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    n: int,
    h: int,
) -> float:
    th = theta.detach().clone().requires_grad_(True)
    pred = twolayer_forward_batch(th, X, n, h)
    loss = 0.5 * torch.mean((pred - y) ** 2)
    g = torch.autograd.grad(loss, th, create_graph=False)[0]
    return float(torch.linalg.norm(g).item())


def _project_twolayer_entrywise(
    theta: torch.Tensor, n: int, h: int, W_bound: float
) -> torch.Tensor:
    del n, h
    return torch.clamp(theta, min=-W_bound, max=W_bound)


def rademacher_twolayer_mc_entrywise(
    theta_init: torch.Tensor,
    X: torch.Tensor,
    n: int,
    h: int,
    W_entry: float,
    n_sigma: int = 128,
    pgd_steps: int = 50,
    pgd_lr: float = 0.1,
    seed: int = 0,
) -> float:
    rng = torch.Generator(device=X.device)
    rng.manual_seed(seed)
    m = X.shape[0]
    acc = 0.0
    for _ in range(n_sigma):
        sigma = torch.randint(0, 2, (m,), generator=rng, device=X.device, dtype=torch.int64)
        sigma = (2 * sigma - 1).to(dtype=X.dtype)
        theta = theta_init.clone().detach().requires_grad_(True)
        for _ in range(pgd_steps):
            pred = twolayer_forward_batch(theta, X, n, h)
            inner = (sigma * pred).mean()
            grad = torch.autograd.grad(inner, theta, create_graph=False)[0]
            with torch.no_grad():
                theta = theta + pgd_lr * grad
                theta = _project_twolayer_entrywise(theta, n, h, W_entry)
            theta = theta.detach().requires_grad_(True)
        pred = twolayer_forward_batch(theta, X, n, h)
        acc += float((sigma * pred).mean().item())
    return acc / n_sigma


def twolayer_weight_entrywise_bounds(theta: torch.Tensor, n: int, h: int) -> Tuple[float, float]:
    hn = h * n
    W1 = theta[:hn].view(h, n).detach().abs()
    w2 = theta[hn:].detach().abs()
    return float(W1.max().item()), float(w2.max().item())


def perturb_twolayer_theta_entrywise(
    theta_np: np.ndarray,
    W_bound: float,
    *,
    inward_scale: float = 0.05,
    noise_scale: float = 0.01,
    seed: int = 0,
    X_reference: Optional[np.ndarray] = None,
    n: Optional[int] = None,
    h: Optional[int] = None,
    max_relative_output_perturbation: Optional[float] = None,
    max_backtracks: int = 16,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta_base = np.array(theta_np, dtype=np.float64, copy=True)
    scale = W_bound / np.sqrt(max(theta_base.size, 1))
    delta = np.zeros_like(theta_base)
    if inward_scale != 0.0:
        delta -= inward_scale * scale * np.sign(theta_base)
    if noise_scale != 0.0:
        delta += noise_scale * scale * rng.standard_normal(theta_base.shape)
    candidate = np.clip(theta_base + delta, -W_bound, W_bound)
    if (
        X_reference is None
        or max_relative_output_perturbation is None
        or n is None
        or h is None
        or max_relative_output_perturbation <= 0.0
    ):
        return candidate
    X_t = torch.tensor(X_reference, dtype=torch.float64)
    theta_t = torch.tensor(theta_base, dtype=torch.float64)
    with torch.no_grad():
        pred_base = twolayer_forward_batch(theta_t, X_t, n, h)
    base_rms = float(torch.sqrt(torch.mean(pred_base.square())).item())
    if base_rms <= 1e-12:
        return candidate
    alpha = 1.0
    for _ in range(max_backtracks + 1):
        theta_try = np.clip(theta_base + alpha * delta, -W_bound, W_bound)
        theta_try_t = torch.tensor(theta_try, dtype=torch.float64)
        with torch.no_grad():
            pred_try = twolayer_forward_batch(theta_try_t, X_t, n, h)
        rel_shift = float(
            torch.sqrt(torch.mean((pred_try - pred_base).square())).item() / base_rms
        )
        if rel_shift <= max_relative_output_perturbation:
            return theta_try
        alpha *= 0.5
    return np.clip(theta_base + alpha * delta, -W_bound, W_bound)


def sample_inputs_worstcase_clusters(
    n: int,
    m: int,
    *,
    seed: int = 0,
    noise_scale: float = 0.03,
    floor: float = 1e-3,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k_mid = max(1, n // 4)
    templates = np.stack(
        [
            np.ones(n, dtype=np.float64),
            np.concatenate(
                [
                    np.ones(k_mid, dtype=np.float64),
                    np.full(n - k_mid, floor, dtype=np.float64),
                ]
            ),
            np.concatenate(
                [
                    np.array([1.0], dtype=np.float64),
                    np.full(n - 1, floor, dtype=np.float64),
                ]
            ),
        ],
        axis=0,
    )
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)
    X = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        x = templates[i % templates.shape[0]] + noise_scale * rng.standard_normal(n)
        x = np.maximum(x, floor)
        X[i] = x
    rng.shuffle(X, axis=0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X *= np.sqrt(n)
    return X


def _aligned_twolayer_theta_entrywise(n: int, h: int, W_bound: float) -> np.ndarray:
    k_mid = max(1, n // 4)
    W1 = np.zeros((h, n), dtype=np.float64)
    for j in range(h):
        mode = j % 3
        if mode == 0:
            W1[j, :] = W_bound
        elif mode == 1:
            W1[j, :k_mid] = W_bound
            W1[j, k_mid:] = 0.25 * W_bound
        else:
            W1[j, 0] = W_bound
            W1[j, 1:] = 0.10 * W_bound
    w2 = np.full(h, W_bound, dtype=np.float64)
    return np.concatenate([W1.reshape(-1), w2], axis=0)


def make_synthetic_twolayer_worstcase_in_ball(
    n: int,
    m: int,
    h_teacher: int,
    W_bound: float,
    noise_std: float = 0.1,
    seed: int = 0,
    *,
    return_theta_teacher: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = sample_inputs_worstcase_clusters(n, m, seed=seed)
    theta_np = _aligned_twolayer_theta_entrywise(n, h_teacher, W_bound)
    hn = h_teacher * n
    W1 = theta_np[:hn].reshape(h_teacher, n)
    w2 = theta_np[hn:]
    rng = np.random.default_rng(seed + 1)
    z = np.maximum(0.0, X @ W1.T)
    y = z @ w2 + noise_std * rng.standard_normal(m)
    if return_theta_teacher:
        return X, y, theta_np
    return X, y


@dataclass
class SyntheticLinearDataset:
    n: int
    m: int
    noise_std: float = 0.1
    seed: int = 0
    row_radius_log_sigma: float = 0.0

    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        w_star = rng.standard_normal(self.n)
        w_star = w_star / np.linalg.norm(w_star) * 1.0
        X = rng.standard_normal((self.m, self.n))
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / row_norms
        if self.row_radius_log_sigma > 0:
            scales = np.exp(rng.normal(0.0, self.row_radius_log_sigma, size=(self.m, 1)))
        else:
            scales = np.ones((self.m, 1))
        X = X * (np.sqrt(self.n) * scales)
        y = X @ w_star + self.noise_std * rng.standard_normal(self.m)
        return X, y, w_star
