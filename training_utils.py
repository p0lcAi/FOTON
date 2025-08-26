"""
Lightweight training utilities (BP, PEPITA, FOTON) with Weights & Biases integration
via local wandb_helpers.py (provides maybe_init_wandb / maybe_log / maybe_finish).

Notes:
- Accuracies stored in results are percentages (0–100).
- FOTON supports a flag `use_bp_reference` to compare its updates against BP gradients.
- Only scalar metrics are logged to W&B by default; per-layer vectors are returned in `results`
  but not logged (to avoid clutter). You can add your own logging if desired.
  
Best checkpoint:
- Tracks the best `test_acc_pct` seen across epochs.
- Saves to disk when it improves (if `save_best_to` is provided).
- Optionally restores the best weights at the end (`restore_best=True`).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from timeit import default_timer as timer

from wandb_helpers import maybe_init_wandb, maybe_log, maybe_finish


# -----------------------------
# Small helpers
# -----------------------------

def print_train_time(
    start: float,
    end: float,
    device: Optional[torch.device] = None,
) -> float:
    """
    Print and return elapsed time.

    Parameters
    ----------
    start, end : float
        Start/end timestamps (e.g., from timeit.default_timer()).
    device : torch.device | None
        Device used (purely for printing).

    Returns
    -------
    float
        Elapsed seconds.
    """
    total_time = float(end - start)
    dev_str = str(device) if device is not None else "unknown-device"
    print(f"Train time on {dev_str}: {total_time:.3f} seconds")
    return total_time

def _batch_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Accuracy in [0,1] without softmax."""
    return (logits.argmax(dim=1) == y).float().mean().item()


def _should_early_stop(epoch: int,
                       threshold: Optional[Tuple[int, float, float]],
                       train_acc_pct: float,
                       test_acc_pct: float) -> bool:
    """
    threshold = (epoch_index, test_acc_min, train_acc_min)
    Note: epoch_index is zero-based to match the loop index.
    """
    if not threshold:
        return False
    stop_epoch, min_test, min_train = threshold
    return (epoch == stop_epoch) and (test_acc_pct < min_test) and (train_acc_pct < min_train)


def _avg_list(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def _num_effective_layers(model: torch.nn.Module) -> int:
    """
    Expected length of updates returned by model.modulated_forward.
    Matches prior convention: len(params) - 1 (F_Te counted among params).
    """
    return max(len(list(model.parameters())) - 1, 0)


# -----------------------------
# Best checkpoint keeper
# -----------------------------
class BestKeeper:
    """
    Tracks the best metric (max) and holds deep copies of the best model/optimizer states.
    """
    def __init__(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        self.model = model
        self.optimizer = optimizer
        self.best_metric = float("-inf")
        self.best_epoch = -1
        self._best_model_sd: Optional[Dict[str, torch.Tensor]] = None
        self._best_opt_sd: Optional[Dict] = None

    def update(self, metric: float, epoch: int) -> bool:
        """Return True if this epoch is a new best."""
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_epoch = epoch
            self._best_model_sd = copy.deepcopy(self.model.state_dict())
            if self.optimizer is not None:
                self._best_opt_sd = copy.deepcopy(self.optimizer.state_dict())
            return True
        return False

    def restore(self) -> None:
        """Load best weights back into the model (and optimizer if provided)."""
        if self._best_model_sd is not None:
            self.model.load_state_dict(self._best_model_sd)
        if self.optimizer is not None and self._best_opt_sd is not None:
            self.optimizer.load_state_dict(self._best_opt_sd)

    def save(self, path: str | Path) -> None:
        """Save best checkpoint to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
                "model_state": self._best_model_sd,
                "optimizer_state": self._best_opt_sd,
            },
            path,
        )


# ===========================
# BP TRAINING STEP
# ===========================
def train_step_BP(model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += _batch_accuracy(logits, y)

    n = len(data_loader)
    return total_loss / n, total_acc / n


# ===========================
# TEST STEP
# ===========================
@torch.inference_mode()
def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              num_classes: Optional[int] = None,
              adversarial_attack: Optional[str] = None,
              adversarial_test: Optional[bool] = None,
              adv_eps: Optional[float] = None,
              device: Optional[torch.device] = None) -> Tuple[float, float]:
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        total_loss += loss.item()
        total_acc += _batch_accuracy(logits, y)

    n = len(data_loader)
    return total_loss / n, total_acc / n


# ===========================
# BP TRAIN+TEST LOOP (with best tracking)
# ===========================
def train_test_BP(model: torch.nn.Module,
                  train_dataloader: torch.utils.data.DataLoader,
                  test_dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  epochs: int,
                  device: torch.device,
                  config: dict = None,
                  sweep: bool = False,
                  threshold: Optional[Tuple[int, float, float]] = None,
                  *,
                  save_best_to: Optional[str] = None,
                  restore_best: bool = True) -> Dict[str, List]:
    model.to(device)

    run = maybe_init_wandb(config)
    t0 = timer()

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best = BestKeeper(model, optimizer)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step_BP(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(test_dataloader, model, loss_fn, device=device)

        train_acc_pct = 100.0 * train_acc
        test_acc_pct  = 100.0 * test_acc

        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc_pct:.2f}% | "
              f"test_loss: {test_loss:.4f} | test_acc: {test_acc_pct:.2f}%")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc_pct)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc_pct)

        # keep the best
        if best.update(test_acc_pct, epoch):
            if save_best_to:
                best.save(save_best_to)
            maybe_log({"best_test_acc_pct": best.best_metric, "best_epoch": best.best_epoch + 1})

        maybe_log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc_pct": train_acc_pct,
            "test_loss": test_loss,
            "test_acc_pct": test_acc_pct,
        })

        if _should_early_stop(epoch, threshold, train_acc_pct, test_acc_pct):
            break

    if restore_best:
        best.restore()

    results["best_test_acc_pct"] = best.best_metric
    results["best_epoch"] = best.best_epoch + 1

    print_train_time(t0, timer(), device)
    maybe_finish()
    return results


# ===========================
# PEPITA TRAIN STEP
# ===========================
def train_step_PEPITA(model: torch.nn.Module,
                      data_loader: torch.utils.data.DataLoader,
                      loss_fn: torch.nn.Module,
                      num_classes: int,
                      lr: float,
                      adversarial_attack: Optional[str] = None,
                      adversarial_train: Optional[bool] = None,
                      adv_eps: Optional[float] = None,
                      device: Optional[torch.device] = None) -> Tuple[float, float]:
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        # Forward (training path)
        y_pred = model(x)
        target = F.one_hot(y, num_classes)

        # PEPITA update
        model.modulated_forward(x=x, y=y_pred, target=target, lr=lr)

        # Evaluate without dropout/etc.
        model.eval()
        with torch.inference_mode():
            logits_eval = model(x)
            total_loss += loss_fn(logits_eval, y).item()
            total_acc  += _batch_accuracy(logits_eval, y)
        model.train()

    n = len(data_loader)
    return total_loss / n, total_acc / n


# ===========================
# PEPITA TRAIN+TEST LOOP (with best tracking)
# ===========================
def train_test_PEPITA(model: torch.nn.Module,
                      train_dataloader: torch.utils.data.DataLoader,
                      test_dataloader: torch.utils.data.DataLoader,
                      loss_fn: torch.nn.Module,
                      lr: float,
                      lr_sched: List,   # [factor, [epochs_to_decay]]
                      epochs: int,
                      adversarial_attack: Optional[str] = None,
                      adversarial_train: Optional[bool] = None,
                      adversarial_test: Optional[bool] = None,
                      adv_eps: Optional[float] = None,
                      device: Optional[torch.device] = None,
                      config: dict = None,
                      sweep: bool = False,
                      threshold: Optional[Tuple[int, float, float]] = None,
                      *,
                      save_best_to: Optional[str] = None,
                      restore_best: bool = True) -> Dict[str, List]:
    model.to(device)

    run = maybe_init_wandb(config)
    t0 = timer()

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    num_classes = len(train_dataloader.dataset.classes)
    best = BestKeeper(model, optimizer=None)

    for epoch in tqdm(range(epochs)):
        if lr_sched and epoch in lr_sched[1]:
            lr = lr / lr_sched[0]

        train_loss, train_acc = train_step_PEPITA(
            model, train_dataloader, loss_fn, num_classes, lr,
            adversarial_attack, adversarial_train, adv_eps, device
        )
        test_loss, test_acc = test_step(test_dataloader, model, loss_fn, device=device)

        train_acc_pct = 100.0 * train_acc
        test_acc_pct  = 100.0 * test_acc

        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc_pct:.2f}% | "
              f"test_loss: {test_loss:.4f} | test_acc: {test_acc_pct:.2f}%")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc_pct)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc_pct)

        # keep the best
        if best.update(test_acc_pct, epoch):
            if save_best_to:
                best.save(save_best_to)
            maybe_log({"best_test_acc_pct": best.best_metric, "best_epoch": best.best_epoch + 1})

        maybe_log({
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc_pct": train_acc_pct,
            "test_loss": test_loss,
            "test_acc_pct": test_acc_pct,
        })

        if _should_early_stop(epoch, threshold, train_acc_pct, test_acc_pct):
            break

    if restore_best:
        best.restore()

    results["best_test_acc_pct"] = best.best_metric
    results["best_epoch"] = best.best_epoch + 1

    print_train_time(t0, timer(), device)
    maybe_finish()
    return results


# ===========================
# FOTON TRAIN STEP (with optional BP reference)
# ===========================
def train_step_FOTON(model: torch.nn.Module,
                     data_loader: torch.utils.data.DataLoader,
                     loss_fn: torch.nn.Module,
                     num_classes: int,
                     lr: float,
                     device: torch.device,
                     optimizer: Optional[torch.optim.Optimizer] = None,  # used only to zero grads if reference on
                     use_bp_reference: bool = True
                     ) -> Tuple[float, float, Optional[List[float]], Optional[List[float]], List[float], Optional[List[float]]]:
    """
    If use_bp_reference=True:
        - compute BP grads (no optimizer step),
        - compare to FOTON updates: return avg cosine, avg ||grad||, avg ||updates||, avg ||grad - updates|| per layer.
    If False:
        - skip BP grads; return only avg ||updates|| (others are None).
    """
    model.train()
    # --- before the loop: initialize accumulators once ---
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    cos_acc = None          # Optional[List[List[float]]]
    norm_grad_acc = None    # Optional[List[List[float]]]
    norm_diff_acc = None    # Optional[List[List[float]]]
    norm_updates_acc = None # List[List[float]]

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        # 1) Forward for logging only (no grad)
        with torch.inference_mode():
            logits = model(x)
            loss = loss_fn(logits, y).mean()
            total_loss += loss.item()

        # 2) FOTON update (no autograd graph)
        target = F.one_hot(y, num_classes=num_classes).to(x.device, dtype=torch.float32)
        with torch.no_grad():
            updates = model.modulated_forward(x=x, y=logits, target=target, lr=lr)  # should return list/tuple
        assert updates is not None, "modulated_forward must return per-parameter updates"
        updates = list(updates)
        n_upd = len(updates)

        # 3) (Re)size metric containers based on n_upd (persist across batches)
        if norm_updates_acc is None or len(norm_updates_acc) != n_upd:
            norm_updates_acc = [[] for _ in range(n_upd)]

        if use_bp_reference:
            if (cos_acc is None) or len(cos_acc) != n_upd:
                cos_acc = [[] for _ in range(n_upd)]
                norm_grad_acc = [[] for _ in range(n_upd)]
                norm_diff_acc = [[] for _ in range(n_upd)]

            # Fresh BP pass on a fresh graph
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

            logits_ref = model(x)  # fresh forward builds graph
            bp_loss = loss_fn(logits_ref, y).mean()
            bp_loss.backward()

        # 4) Fill metrics
        for i, (p, u) in enumerate(zip(model.parameters(), updates)):
            if u is None:
                norm_updates_acc[i].append(float("nan"))
                if use_bp_reference:
                    cos_acc[i].append(float("nan"))
                    norm_grad_acc[i].append(float("nan"))
                    norm_diff_acc[i].append(float("nan"))
                continue

            # always track update norm
            norm_updates_acc[i].append(u.detach().norm().item())

            # optional BP comparison
            if use_bp_reference and (p.grad is not None) and (p.data.shape == u.shape):
                g = p.grad.detach()
                cos_acc[i].append(F.cosine_similarity(g.flatten(), u.flatten(), dim=0).item())
                norm_grad_acc[i].append(g.norm().item())
                norm_diff_acc[i].append((g - u).norm().item())

        # 5) Eval accuracy with dropout off
        model.eval()
        with torch.inference_mode():
            logits_eval = model(x)
            total_acc += _batch_accuracy(logits_eval, y)
        model.train()

        n_batches += 1

    # --- after the loop: average metrics across batches ---
    def _avg_list(v: list[float]) -> float:
        return float(sum(v) / len(v)) if len(v) > 0 else float("nan")

    if use_bp_reference:
        cos_out       = [_avg_list(v) for v in cos_acc]        # type: ignore[arg-type]
        norm_grad_out = [_avg_list(v) for v in norm_grad_acc]  # type: ignore[arg-type]
        norm_diff_out = [_avg_list(v) for v in norm_diff_acc]  # type: ignore[arg-type]
    else:
        cos_out = norm_grad_out = norm_diff_out = None

    norm_updates_out = [_avg_list(v) for v in norm_updates_acc]

    return (total_loss / n_batches, total_acc / n_batches,
            cos_out, norm_grad_out, norm_updates_out, norm_diff_out)


# ===========================
# FOTON TRAIN+TEST LOOP (with best tracking + optional BP reference)
# ===========================
def train_test_FOTON(model: torch.nn.Module,
                     train_dataloader: torch.utils.data.DataLoader,
                     test_dataloader: torch.utils.data.DataLoader,
                     loss_fn: torch.nn.Module,
                     lr: float,
                     lr_sched: List,   # [factor, [epochs_to_decay]]
                     epochs: int,
                     device: torch.device,
                     optimizer: Optional[torch.optim.Optimizer] = None,
                     config: dict = None,
                     sweep: bool = False,
                     threshold: Optional[Tuple[int, float, float]] = None,
                     use_bp_reference: bool = True,
                     *,
                     save_best_to: Optional[str] = None,
                     restore_best: bool = True) -> Dict[str, List]:
    model.to(device)

    run = maybe_init_wandb(config)
    t0 = timer()

    results: Dict[str, List] = {
        "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
        "cos": [], "norm_grad": [], "norm_updates": [], "norm_diff": []
    }
    num_classes = len(train_dataloader.dataset.classes)
    best = BestKeeper(model, optimizer)

    for epoch in tqdm(range(epochs)):
        if lr_sched and epoch in lr_sched[1]:
            lr = lr / lr_sched[0]

        train_loss, train_acc, cos, norm_grad, norm_updates, norm_diff = train_step_FOTON(
            model, train_dataloader, loss_fn, num_classes, lr, device, optimizer, use_bp_reference
        )
        test_loss, test_acc = test_step(test_dataloader, model, loss_fn, device=device)

        train_acc_pct = 100.0 * train_acc
        test_acc_pct  = 100.0 * test_acc

        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc_pct:.2f}% | "
              f"test_loss: {test_loss:.4f} | test_acc: {test_acc_pct:.2f}%")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc_pct)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc_pct)  # ensure pct stored
        results["cos"].append(cos)
        results["norm_grad"].append(norm_grad)
        results["norm_updates"].append(norm_updates)
        results["norm_diff"].append(norm_diff)

        # keep the best
        if best.update(test_acc_pct, epoch):
            if save_best_to:
                best.save(save_best_to)
            maybe_log({"best_test_acc_pct": best.best_metric, "best_epoch": best.best_epoch + 1})

        maybe_log({
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc_pct": train_acc_pct,
            "test_loss": test_loss,
            "test_acc_pct": test_acc_pct,
            # Optionally add per-layer summaries:
            # "cos_mean": None if cos is None else float(sum(cos)/len(cos)),
        })

        if _should_early_stop(epoch, threshold, train_acc_pct, test_acc_pct):
            break

    if restore_best:
        best.restore()

    results["best_test_acc_pct"] = best.best_metric
    results["best_epoch"] = best.best_epoch + 1

    print_train_time(t0, timer(), device)
    maybe_finish()
    return results
