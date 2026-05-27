"""
Influence-function unlearning with an EKFAC inverse-Fisher approximation.

The EKFAC paper is an optimizer paper, not an unlearning paper. This module
uses its inverse empirical-Fisher preconditioner as the curvature inverse in
the standard influence approximation for removing a forget set:

    theta_retrained ~= theta + step_size * G_EKFAC^-1 grad L_forget(theta)

Supported layers are nn.Linear and nn.Conv2d. Other parameters fall back to a
diagonal empirical-Fisher preconditioner so residual BatchNorm parameters still
receive a curvature-scaled update.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseUnlearning


@dataclass
class _LayerState:
    module: nn.Module
    name: str
    activation: Optional[torch.Tensor] = None
    grad_output: Optional[torch.Tensor] = None
    ua: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    weight_scaling: Optional[torch.Tensor] = None
    bias_scaling: Optional[torch.Tensor] = None


class EKFACInfluenceUnlearning(BaseUnlearning):
    """Approximate influence-function unlearning using EKFAC preconditioning."""

    name = "ekfac_influence"

    def __init__(
        self,
        model,
        step_size: float = 0.01,
        damping: float = 1e-3,
        update_norm_clip: Optional[float] = None,
        max_curvature_batches: Optional[int] = None,
        max_forget_batches: Optional[int] = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.step_size = step_size
        self.damping = damping
        self.update_norm_clip = update_norm_clip
        self.max_curvature_batches = max_curvature_batches
        self.max_forget_batches = max_forget_batches
        self.device = device
        self._states: Dict[nn.Module, _LayerState] = {}
        self._handles = []
        self._diag_fisher: Dict[str, torch.Tensor] = {}

    def unlearn(self, forget_loader, train_loader, **kwargs) -> dict:
        logger = kwargs.get("logger")
        model = self.model.model.to(self.device)

        if logger:
            logger.print("\n[bold yellow]=== EKFAC Influence Unlearning ===[/bold yellow]")
            logger.info(f"  Step size: [cyan]{self.step_size}[/cyan]")
            logger.info(f"  Damping: [cyan]{self.damping}[/cyan]")
            logger.info(f"  Update norm clip: [cyan]{self.update_norm_clip or 'disabled'}[/cyan]")
            logger.info(f"  Curvature batches: [cyan]{self.max_curvature_batches or 'all'}[/cyan]")
            logger.info(f"  Forget batches: [cyan]{self.max_forget_batches or 'all'}[/cyan]")

        self._register_hooks()
        try:
            curvature_stats = self._estimate_ekfac_curvature(train_loader, logger)
            forget_grads, forget_stats = self._compute_forget_gradient(forget_loader, logger)
        finally:
            self._remove_hooks()

        update_stats = self._apply_influence_update(forget_grads)

        if logger:
            logger.info("  Influence update applied")
            logger.info(f"    - Updated parameters: {update_stats['updated_params']:,}")
            logger.info(f"    - Raw update norm: {update_stats['raw_update_norm']:.6f}")
            logger.info(f"    - Applied update norm: {update_stats['update_norm']:.6f}")
            logger.info(f"    - Clip coefficient: {update_stats['clip_coef']:.6f}")
            logger.log_metrics({
                "curvature_batches": float(curvature_stats["batches"]),
                "forget_batches": float(forget_stats["batches"]),
                "forget_grad_norm": forget_stats["grad_norm"],
                "raw_update_norm": update_stats["raw_update_norm"],
                "update_norm": update_stats["update_norm"],
                "clip_coef": update_stats["clip_coef"],
                "nonfinite_update_tensors": float(update_stats["nonfinite_update_tensors"]),
            }, step=0, prefix="unlearning/")

        return {
            "method": self.name,
            "step_size": self.step_size,
            "damping": self.damping,
            "update_norm_clip": self.update_norm_clip,
            "curvature_stats": curvature_stats,
            "forget_stats": forget_stats,
            "update_stats": update_stats,
        }

    def _register_hooks(self) -> None:
        for name, module in self.model.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self._states[module] = _LayerState(module=module, name=name)
                self._handles.append(module.register_forward_hook(self._save_activation))
                self._handles.append(module.register_full_backward_hook(self._save_grad_output))

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _save_activation(self, module, inputs, output) -> None:
        self._states[module].activation = inputs[0].detach()

    def _save_grad_output(self, module, grad_input, grad_output) -> None:
        self._states[module].grad_output = grad_output[0].detach()

    def _estimate_ekfac_curvature(self, dataloader, logger=None) -> dict:
        model = self.model.model
        model.eval()
        criterion = nn.CrossEntropyLoss()

        factor_sums = {}
        scaling_sums = {}
        diag_sums = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        total = 0
        batches = 0

        iterator = tqdm(dataloader, desc="Estimating EKFAC curvature", unit="batch")
        for inputs, targets in iterator:
            if self.max_curvature_batches is not None and batches >= self.max_curvature_batches:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size = inputs.size(0)

            model.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()

            batch_factors = {}
            for module, state in self._states.items():
                if state.activation is None or state.grad_output is None:
                    continue
                a, b, h_proj, delta_proj = self._compute_batch_factors_and_projection(state)
                key = state.name
                batch_factors[key] = (h_proj, delta_proj)
                if key not in factor_sums:
                    factor_sums[key] = {
                        "a": torch.zeros_like(a),
                        "b": torch.zeros_like(b),
                        "count": 0,
                        "module": module,
                    }
                count = h_proj.size(0)
                factor_sums[key]["a"] += a * count
                factor_sums[key]["b"] += b * count
                factor_sums[key]["count"] += count

            for name, param in model.named_parameters():
                if param.grad is not None:
                    diag_sums[name] += param.grad.detach().pow(2) * batch_size

            total += batch_size
            batches += 1

        if batches == 0 or total == 0:
            raise ValueError("No batches were available for EKFAC curvature estimation.")

        for state in self._states.values():
            factor = factor_sums.get(state.name)
            if factor is None:
                continue
            a = factor["a"] / factor["count"]
            b = factor["b"] / factor["count"]
            _, state.ua = torch.linalg.eigh(a + self.damping * torch.eye(a.size(0), device=a.device))
            _, state.ub = torch.linalg.eigh(b + self.damping * torch.eye(b.size(0), device=b.device))

        # Recompute projected second moments once the KFE bases are known.
        batches_for_scaling = 0
        iterator = tqdm(dataloader, desc="Estimating EKFAC scalings", unit="batch")
        for inputs, targets in iterator:
            if self.max_curvature_batches is not None and batches_for_scaling >= self.max_curvature_batches:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            model.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()

            for module, state in self._states.items():
                if state.ua is None or state.ub is None:
                    continue
                h_proj, delta_proj = self._project_saved_batch(state)
                key = state.name
                weight_scaling = delta_proj.pow(2).t().matmul(h_proj.pow(2)) / h_proj.size(0)
                bias_scaling = delta_proj.pow(2).mean(dim=0) if module.bias is not None else None

                if key not in scaling_sums:
                    scaling_sums[key] = {
                        "weight": torch.zeros_like(weight_scaling),
                        "bias": torch.zeros_like(bias_scaling) if bias_scaling is not None else None,
                        "count": 0,
                    }
                scaling_sums[key]["weight"] += weight_scaling
                if bias_scaling is not None:
                    scaling_sums[key]["bias"] += bias_scaling
                scaling_sums[key]["count"] += 1

            batches_for_scaling += 1

        for state in self._states.values():
            scaling = scaling_sums.get(state.name)
            if scaling is None:
                continue
            state.weight_scaling = scaling["weight"] / scaling["count"]
            if scaling["bias"] is not None:
                state.bias_scaling = scaling["bias"] / scaling["count"]

        self._diag_fisher = {name: value / total for name, value in diag_sums.items()}
        return {"batches": batches, "samples": total, "layers": len(scaling_sums)}

    def _compute_batch_factors_and_projection(self, state: _LayerState):
        h, delta = self._flatten_layer_io(state)
        a = h.t().matmul(h) / h.size(0)
        b = delta.t().matmul(delta) / delta.size(0)
        return a, b, h, delta

    def _project_saved_batch(self, state: _LayerState):
        h, delta = self._flatten_layer_io(state)
        return h.matmul(state.ua), delta.matmul(state.ub)

    def _flatten_layer_io(self, state: _LayerState):
        module = state.module
        activation = state.activation
        grad_output = state.grad_output

        if isinstance(module, nn.Linear):
            h = activation.reshape(-1, activation.shape[-1])
            delta = grad_output.reshape(-1, grad_output.shape[-1])
            return h, delta

        if isinstance(module, nn.Conv2d):
            patches = F.unfold(
                activation,
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            h = patches.transpose(1, 2).reshape(-1, patches.size(1))
            delta = grad_output.permute(0, 2, 3, 1).reshape(-1, grad_output.size(1))
            return h, delta

        raise TypeError(f"Unsupported EKFAC layer type: {type(module)}")

    def _compute_forget_gradient(self, dataloader, logger=None):
        model = self.model.model
        model.eval()
        criterion = nn.CrossEntropyLoss()
        grads = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        total = 0
        batches = 0

        iterator = tqdm(dataloader, desc="Computing forget gradient", unit="batch")
        for inputs, targets in iterator:
            if self.max_forget_batches is not None and batches >= self.max_forget_batches:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size = inputs.size(0)

            model.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grads[name] += param.grad.detach() * batch_size

            total += batch_size
            batches += 1

        if total == 0:
            raise ValueError("No samples were available for forget-gradient estimation.")

        grads = {name: grad / total for name, grad in grads.items()}
        grad_norm = torch.sqrt(sum(grad.pow(2).sum() for grad in grads.values())).item()
        return grads, {"batches": batches, "samples": total, "grad_norm": grad_norm}

    def _apply_influence_update(self, forget_grads: Dict[str, torch.Tensor]) -> dict:
        module_by_param = {}
        for module, state in self._states.items():
            for param_name, _ in module.named_parameters(recurse=False):
                module_by_param[f"{state.name}.{param_name}"] = state

        updates = {}
        raw_update_norm_sq = 0.0
        nonfinite_update_tensors = 0

        with torch.no_grad():
            for name, param in self.model.model.named_parameters():
                if not param.requires_grad or name not in forget_grads:
                    continue

                update = self._precondition_gradient(name, forget_grads[name], module_by_param.get(name))
                update = self.step_size * update
                if not torch.isfinite(update).all():
                    nonfinite_update_tensors += 1
                    update = torch.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)

                updates[name] = update
                raw_update_norm_sq += update.pow(2).sum().item()

            raw_update_norm = raw_update_norm_sq ** 0.5
            clip_coef = 1.0
            if self.update_norm_clip is not None and raw_update_norm > self.update_norm_clip:
                clip_coef = self.update_norm_clip / (raw_update_norm + 1e-12)

            update_norm_sq = 0.0
            updated_params = 0
            for name, param in self.model.model.named_parameters():
                if name not in updates:
                    continue

                update = updates[name] * clip_coef
                param.add_(update)
                update_norm_sq += update.pow(2).sum().item()
                updated_params += param.numel()

        return {
            "updated_params": updated_params,
            "raw_update_norm": raw_update_norm,
            "update_norm": update_norm_sq ** 0.5,
            "clip_coef": clip_coef,
            "nonfinite_update_tensors": nonfinite_update_tensors,
        }

    def _precondition_gradient(
        self,
        name: str,
        grad: torch.Tensor,
        state: Optional[_LayerState],
    ) -> torch.Tensor:
        if state is None or state.ua is None or state.ub is None:
            fisher = self._diag_fisher.get(name)
            if fisher is None:
                return grad
            return grad / (fisher + self.damping)

        if name.endswith(".weight") and state.weight_scaling is not None:
            matrix_grad = grad.reshape(grad.size(0), -1)
            projected = state.ub.t().matmul(matrix_grad).matmul(state.ua)
            projected = projected / (state.weight_scaling + self.damping)
            preconditioned = state.ub.matmul(projected).matmul(state.ua.t())
            return preconditioned.reshape_as(grad)

        if name.endswith(".bias") and state.bias_scaling is not None:
            projected = state.ub.t().matmul(grad)
            projected = projected / (state.bias_scaling + self.damping)
            return state.ub.matmul(projected)

        fisher = self._diag_fisher.get(name)
        if fisher is None:
            return grad
        return grad / (fisher + self.damping)

    def save_unlearned_model(self, save_path: str) -> None:
        torch.save({
            "model_state_dict": self.model.model.state_dict(),
            "method": self.name,
            "step_size": self.step_size,
            "damping": self.damping,
            "update_norm_clip": self.update_norm_clip,
        }, save_path)
