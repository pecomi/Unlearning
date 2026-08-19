"""
Influence-function unlearning with an EKFAC inverse-Fisher approximation.

The EKFAC paper is an optimizer paper, not an unlearning paper. This module
uses its inverse empirical-Fisher preconditioner as the curvature inverse in
the standard influence approximation for removing a forget set:

    theta_retrained ~= theta + step_size * G_EKFAC^-1 grad L_forget(theta)

Supported layers are nn.Linear and nn.Conv2d. Other parameters are frozen by
default because a batch-mean squared gradient is not a valid per-example
empirical-Fisher estimate. The legacy diagonal fallback is opt-in.
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
    inverse_damping: Optional[torch.Tensor] = None


class EKFACInfluenceUnlearning(BaseUnlearning):
    """Approximate influence-function unlearning using EKFAC preconditioning."""

    name = "ekfac_influence"

    def __init__(
        self,
        model,
        step_size: float = 0.007,
        damping_mode: str = "absolute",
        damping: float = 0.05,
        damping_ratio: float = 0.1,
        damping_floor: float = 1e-8,
        regularization_curvature: float = 0.0,
        update_unsupported_params: bool = False,
        num_unlearn_steps: int = 1,
        recompute_curvature_each_step: bool = False,
        unlearn_eval_interval: int = 1,
        max_eval_batches: Optional[int] = None,
        max_curvature_batches: Optional[int] = None,
        max_forget_batches: Optional[int] = None,
        forget_update_mode: str = "full_batch",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.step_size = float(step_size)
        self.damping_mode = str(damping_mode)
        self.damping = float(damping)
        self.damping_ratio = float(damping_ratio)
        self.damping_floor = float(damping_floor)
        self.regularization_curvature = float(regularization_curvature)
        self.update_unsupported_params = bool(update_unsupported_params)
        self.num_unlearn_steps = max(1, int(num_unlearn_steps))
        self.recompute_curvature_each_step = recompute_curvature_each_step
        self.unlearn_eval_interval = max(0, int(unlearn_eval_interval))
        self.max_eval_batches = max_eval_batches
        self.max_curvature_batches = max_curvature_batches
        self.max_forget_batches = max_forget_batches
        self.forget_update_mode = forget_update_mode
        self.device = device
        self._states: Dict[nn.Module, _LayerState] = {}
        self._handles = []
        self._diag_fisher: Dict[str, torch.Tensor] = {}
        if self.forget_update_mode not in {"full_batch", "minibatch"}:
            raise ValueError(
                "forget_update_mode must be either 'full_batch' or 'minibatch' "
                f"(got {self.forget_update_mode!r})."
            )
        if self.forget_update_mode != "full_batch":
            raise ValueError(
                "EKFAC influence unlearning now supports only full_batch updates. "
                "Mini-batch updates are sequential parameter updates and are not "
                "equivalent to the intended single influence/Newton correction."
            )
        if self.damping_mode not in {"absolute", "relative"}:
            raise ValueError(
                "damping_mode must be either 'absolute' or 'relative' "
                f"(got {self.damping_mode!r})."
            )

    def unlearn(self, forget_loader, train_loader, **kwargs) -> dict:
        logger = kwargs.get("logger")
        eval_loaders = kwargs.get("eval_loaders", {})
        model = self.model.model.to(self.device)

        if logger:
            logger.print("\n[bold yellow]=== EKFAC Influence Unlearning ===[/bold yellow]")
            logger.info(f"  Update step size: [cyan]{self.step_size}[/cyan]")
            logger.info(f"  Damping mode: [cyan]{self.damping_mode}[/cyan]")
            logger.info(f"  Absolute damping: [cyan]{self.damping}[/cyan]")
            logger.info(f"  Relative damping ratio: [cyan]{self.damping_ratio}[/cyan]")
            logger.info(f"  Damping floor: [cyan]{self.damping_floor}[/cyan]")
            logger.info(f"  Regularization curvature: [cyan]{self.regularization_curvature}[/cyan]")
            logger.info(f"  Update unsupported parameters: [cyan]{self.update_unsupported_params}[/cyan]")
            logger.info(f"  Unlearn steps: [cyan]{self.num_unlearn_steps}[/cyan]")
            logger.info(f"  Recompute curvature each step: [cyan]{self.recompute_curvature_each_step}[/cyan]")
            logger.info(f"  Eval interval: [cyan]{self.unlearn_eval_interval or 'disabled'}[/cyan]")
            logger.info(f"  Max eval batches: [cyan]{self.max_eval_batches or 'all'}[/cyan]")
            logger.info(f"  Curvature batches: [cyan]{self.max_curvature_batches or 'all'}[/cyan]")
            logger.info(f"  Forget batches: [cyan]{self.max_forget_batches or 'all'}[/cyan]")
            logger.info(f"  Forget update mode: [cyan]{self.forget_update_mode}[/cyan]")

        step_results = []
        eval_history = []
        self._register_hooks()
        try:
            curvature_stats = self._estimate_ekfac_curvature(train_loader, logger)

            for unlearn_step in range(self.num_unlearn_steps):
                if self.recompute_curvature_each_step and unlearn_step > 0:
                    curvature_stats = self._estimate_ekfac_curvature(train_loader, logger)

                forget_grads, forget_stats = self._compute_forget_gradient(forget_loader, logger)
                forget_samples = len(forget_loader.dataset)
                retain_samples = len(train_loader.dataset)
                if retain_samples <= 0:
                    raise ValueError("Retain set must contain at least one sample.")
                removal_scale = forget_samples / retain_samples
                effective_step_size = self.step_size
                update_stats = self._apply_influence_update(
                    forget_grads,
                    step_size=effective_step_size,
                )

                step_result = {
                    "step": unlearn_step + 1,
                    "effective_step_size": effective_step_size,
                    "removal_scale": removal_scale,
                    "curvature_stats": curvature_stats,
                    "forget_stats": forget_stats,
                    "update_stats": update_stats,
                }
                step_results.append(step_result)

                if logger:
                    logger.info(f"  Influence update applied ({unlearn_step + 1}/{self.num_unlearn_steps})")
                    logger.info(f"    - Effective step size: {effective_step_size:.6f}")
                    logger.info(f"    - Removal scale |Df|/|Dr|: {removal_scale:.6f}")
                    logger.info(f"    - Updated parameters: {update_stats['updated_params']:,}")
                    logger.info(f"    - Raw update norm: {update_stats['raw_update_norm']:.6f}")
                    logger.info(f"    - Applied update norm: {update_stats['update_norm']:.6f}")
                    logger.log_metrics({
                        "curvature_batches": float(curvature_stats["batches"]),
                        "effective_step_size": effective_step_size,
                        "forget_batches": float(forget_stats["batches"]),
                        "forget_grad_norm": forget_stats["grad_norm"],
                        "raw_update_norm": update_stats["raw_update_norm"],
                        "update_norm": update_stats["update_norm"],
                        "nonfinite_update_tensors": float(update_stats["nonfinite_update_tensors"]),
                    }, step=unlearn_step + 1, prefix="unlearning/")

                    if self.unlearn_eval_interval and (unlearn_step + 1) % self.unlearn_eval_interval == 0:
                        eval_metrics = self._evaluate_loaders(eval_loaders)
                        if eval_metrics:
                            eval_history.append({
                                "step": unlearn_step + 1,
                                **eval_metrics,
                            })
                            step_result["eval_metrics"] = eval_metrics
                            logger.log_metrics(eval_metrics, step=unlearn_step + 1, prefix="unlearning_eval/")
        finally:
            self._remove_hooks()

        final_result = step_results[-1]
        if logger and eval_history:
            self._log_validation_history_plot(eval_history, logger)

        return {
            "method": self.name,
            "step_size": self.step_size,
            "damping_mode": self.damping_mode,
            "damping": self.damping,
            "damping_ratio": self.damping_ratio,
            "damping_floor": self.damping_floor,
            "regularization_curvature": self.regularization_curvature,
            "update_unsupported_params": self.update_unsupported_params,
            "num_unlearn_steps": self.num_unlearn_steps,
            "recompute_curvature_each_step": self.recompute_curvature_each_step,
            "unlearn_eval_interval": self.unlearn_eval_interval,
            "max_eval_batches": self.max_eval_batches,
            "forget_update_mode": self.forget_update_mode,
            "curvature_stats": final_result["curvature_stats"],
            "forget_stats": final_result["forget_stats"],
            "update_stats": final_result["update_stats"],
            "step_results": step_results,
            "eval_history": eval_history,
        }

#layer별 h와 delta를 저장하기 위한 hook
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

#activation=h, grad_output=delta로 저장. EKFAC에서 h는 입력의 패치화된 버전, delta는 출력에 대한 그라디언트의 플랫된 버전. 
    def _save_activation(self, module, inputs, output) -> None:
        self._states[module].activation = inputs[0].detach()

    def _save_grad_output(self, module, grad_input, grad_output) -> None:
        self._states[module].grad_output = grad_output[0].detach() * grad_output[0].size(0)

    def _estimate_ekfac_curvature(self, dataloader, logger=None) -> dict:
        model = self.model.model
        model.eval()
        criterion = nn.CrossEntropyLoss()
        #ekfac A,B factor / EKFAC projected gradient second moment / diagonal Fisher 누적 값
        factor_sums = {}
        scaling_sums = {}
        diag_sums = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        } if self.update_unsupported_params else {}
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
                #h와 delta에 대한 연산. A와 B를 구하게 됨
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
                if name in diag_sums and param.grad is not None:
                    diag_sums[name] += param.grad.detach().pow(2) * batch_size

            total += batch_size
            batches += 1

        if batches == 0 or total == 0:
            raise ValueError("No batches were available for EKFAC curvature estimation.")
        #EKFAC의 KFE 기반 역행렬 근사에 필요한 고유값과 고유벡터 계산. A와 B는 각각 입력과 출력의 공분산 행렬로, 이들의 고유분해를 통해 KFE의 좌우 고유벡터(ua, ub)를 구함.
        for state in self._states.values():
            factor = factor_sums.get(state.name)
            if factor is None:
                continue
            a = factor["a"] / factor["count"]
            b = factor["b"] / factor["count"]
            _, state.ua = torch.linalg.eigh(
                a + self.damping_floor * torch.eye(a.size(0), device=a.device)
            )
            _, state.ub = torch.linalg.eigh(
                b + self.damping_floor * torch.eye(b.size(0), device=b.device)
            )

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
                weight_scaling = self._compute_scaling(state)
                key = state.name

                if key not in scaling_sums:
                    scaling_sums[key] = {
                        "weight": torch.zeros_like(weight_scaling),
                        "count": 0,
                    }
                batch_count = state.activation.size(0)
                scaling_sums[key]["weight"] += weight_scaling * batch_count
                scaling_sums[key]["count"] += batch_count

            batches_for_scaling += 1

        for state in self._states.values():
            scaling = scaling_sums.get(state.name)
            if scaling is None:
                continue
            state.weight_scaling = scaling["weight"] / scaling["count"]
            state.inverse_damping = self._inverse_damping(state.weight_scaling)

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

    def _compute_scaling(self, state: _LayerState) -> torch.Tensor:
        h, delta = self._layer_io_by_sample(state)
        module = state.module

        if isinstance(module, nn.Linear):
            h_proj = h.matmul(state.ua)
            delta_proj = delta.matmul(state.ub)
            return delta_proj.pow(2).t().matmul(h_proj.pow(2)) / h_proj.size(0)

        if isinstance(module, nn.Conv2d):
            batch_size, num_locations, in_dim = h.shape
            out_dim = delta.shape[-1]
            h_proj = h.reshape(-1, in_dim).matmul(state.ua).view(batch_size, num_locations, -1)
            delta_proj = delta.reshape(-1, out_dim).matmul(state.ub).view(batch_size, num_locations, -1)
            per_sample_grad = torch.einsum("nlo,nli->noi", delta_proj, h_proj)
            return per_sample_grad.pow(2).mean(dim=0)

        raise TypeError(f"Unsupported EKFAC layer type: {type(module)}")

    def _flatten_layer_io(self, state: _LayerState):
        h, delta = self._layer_io_by_sample(state)
        return h.reshape(-1, h.shape[-1]), delta.reshape(-1, delta.shape[-1])

    def _layer_io_by_sample(self, state: _LayerState):
        module = state.module
        activation = state.activation
        grad_output = state.grad_output

        if isinstance(module, nn.Linear):
            h = activation.reshape(-1, activation.shape[-1])
            delta = grad_output.reshape(-1, grad_output.shape[-1])
            if module.bias is not None:
                h = torch.cat([h, torch.ones_like(h[:, :1])], dim=1)
            return h, delta

        if isinstance(module, nn.Conv2d):
            patches = F.unfold(
                activation,
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            h = patches.transpose(1, 2)
            if module.bias is not None:
                h = torch.cat([h, torch.ones_like(h[:, :, :1])], dim=2)
            delta = grad_output.flatten(2).transpose(1, 2)
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

    def _compute_forget_batch_gradient(self, inputs, targets) -> Dict[str, torch.Tensor]:
        model = self.model.model
        criterion = nn.CrossEntropyLoss()

        model.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()

        grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                grads[name] = (
                    torch.zeros_like(param, device=self.device)
                    if param.grad is None
                    else param.grad.detach().clone()
                )
        return grads

    def _run_minibatch_forget_updates(self, dataloader, step_size: float) -> tuple:
        model = self.model.model
        model.eval()

        total = 0
        batches = 0
        grad_norm_sum = 0.0
        grad_norm_sq_sum = 0.0
        raw_update_norm_sq = 0.0
        update_norm_sq = 0.0
        nonfinite_update_tensors = 0
        updated_params = 0

        iterator = tqdm(dataloader, desc="Mini-batch forget updates", unit="batch")
        for inputs, targets in iterator:
            if self.max_forget_batches is not None and batches >= self.max_forget_batches:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size = inputs.size(0)

            batch_grads = self._compute_forget_batch_gradient(inputs, targets)
            batch_grad_norm = torch.sqrt(
                sum(grad.pow(2).sum() for grad in batch_grads.values())
            ).item()
            batch_update_stats = self._apply_influence_update(batch_grads, step_size=step_size)

            total += batch_size
            batches += 1
            grad_norm_sum += batch_grad_norm
            grad_norm_sq_sum += batch_grad_norm ** 2
            raw_update_norm_sq += batch_update_stats["raw_update_norm"] ** 2
            update_norm_sq += batch_update_stats["update_norm"] ** 2
            nonfinite_update_tensors += batch_update_stats["nonfinite_update_tensors"]
            updated_params = batch_update_stats["updated_params"]

        if total == 0:
            raise ValueError("No samples were available for mini-batch forget updates.")

        forget_stats = {
            "batches": batches,
            "samples": total,
            "grad_norm": grad_norm_sum / batches,
            "mean_batch_grad_norm": grad_norm_sum / batches,
            "rms_batch_grad_norm": (grad_norm_sq_sum / batches) ** 0.5,
        }
        update_stats = {
            "updated_params": updated_params,
            "raw_update_norm": raw_update_norm_sq ** 0.5,
            "update_norm": update_norm_sq ** 0.5,
            "nonfinite_update_tensors": nonfinite_update_tensors,
            "updates_applied": batches,
        }
        return forget_stats, update_stats

    def _evaluate_loaders(self, eval_loaders: dict) -> dict:
        metrics = {}
        if not eval_loaders:
            return metrics

        model = self.model.model
        model.eval()
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for loader_name, dataloader in eval_loaders.items():
                if dataloader is None or len(dataloader.dataset) == 0:
                    continue

                total_loss = 0.0
                total_correct = 0
                total_samples = 0
                batches = 0

                for inputs, targets in dataloader:
                    if self.max_eval_batches is not None and batches >= self.max_eval_batches:
                        break

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    total_loss += loss.item() * inputs.size(0)
                    total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
                    total_samples += inputs.size(0)
                    batches += 1

                if total_samples == 0:
                    continue

                metrics[f"{loader_name}_loss"] = total_loss / total_samples
                metrics[f"{loader_name}_accuracy"] = total_correct / total_samples
                metrics[f"{loader_name}_samples"] = float(total_samples)

        return metrics

    def _log_validation_history_plot(self, eval_history, logger=None) -> None:
        try:
            import matplotlib.pyplot as plt
            import wandb
        except ImportError:
            if logger:
                logger.warning("Skipping validation history plot because matplotlib or wandb is unavailable.")
            return

        if wandb.run is None:
            return

        steps = [record["step"] for record in eval_history]
        loss_metrics = ["retain_val_loss", "forget_val_loss"]
        accuracy_metrics = ["retain_val_accuracy", "forget_val_accuracy"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for metric_name in loss_metrics:
            values = [record.get(metric_name) for record in eval_history]
            if any(value is not None for value in values):
                axes[0].plot(steps, values, marker="o", label=metric_name.replace("_", " "))
        axes[0].set_title("Validation Loss During Unlearning")
        axes[0].set_xlabel("Unlearn step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        for metric_name in accuracy_metrics:
            values = [record.get(metric_name) for record in eval_history]
            if any(value is not None for value in values):
                axes[1].plot(steps, values, marker="o", label=metric_name.replace("_", " "))
        axes[1].set_title("Validation Accuracy During Unlearning")
        axes[1].set_xlabel("Unlearn step")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        wandb.log({"unlearning_eval/validation_history": wandb.Image(fig)}, step=steps[-1])
        plt.close(fig)

    def _apply_influence_update(self, forget_grads: Dict[str, torch.Tensor], step_size: float) -> dict:
        named_params = dict(self.model.model.named_parameters())
        updates = {}
        raw_update_norm_sq = 0.0
        nonfinite_update_tensors = 0

        def add_update(param_name: str, param: torch.Tensor, update: torch.Tensor) -> None:
            nonlocal raw_update_norm_sq, nonfinite_update_tensors

            update = step_size * update
            if not torch.isfinite(update).all():
                nonfinite_update_tensors += 1
                update = torch.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)

            updates[param_name] = update
            raw_update_norm_sq += update.pow(2).sum().item()

        with torch.no_grad():
            handled_params = set()
            for state in self._states.values():
                weight_name = f"{state.name}.weight"
                bias_name = f"{state.name}.bias"

                if weight_name not in forget_grads or weight_name not in named_params:
                    continue

                weight_param = named_params[weight_name]
                bias_param = named_params.get(bias_name)
                bias_grad = forget_grads.get(bias_name) if bias_param is not None else None
                weight_update, bias_update = self._precondition_module_gradient(
                    state=state,
                    weight_grad=forget_grads[weight_name],
                    bias_grad=bias_grad,
                )

                if weight_update is None:
                    continue
                add_update(weight_name, weight_param, weight_update)
                handled_params.add(weight_name)
                if bias_param is not None and bias_update is not None:
                    add_update(bias_name, bias_param, bias_update)
                    handled_params.add(bias_name)

            for name, param in self.model.model.named_parameters():
                if not param.requires_grad or name not in forget_grads or name in handled_params:
                    continue

                if not self.update_unsupported_params:
                    continue

                update = self._precondition_gradient(name, forget_grads[name])
                add_update(name, param, update)

            raw_update_norm = raw_update_norm_sq ** 0.5

            update_norm_sq = 0.0
            updated_params = 0
            for name, param in self.model.model.named_parameters():
                if name not in updates:
                    continue

                update = updates[name]
                param.add_(update)
                update_norm_sq += update.pow(2).sum().item()
                updated_params += param.numel()

        return {
            "updated_params": updated_params,
            "raw_update_norm": raw_update_norm,
            "update_norm": update_norm_sq ** 0.5,
            "nonfinite_update_tensors": nonfinite_update_tensors,
        }

    def _precondition_module_gradient(
        self,
        state: _LayerState,
        weight_grad: torch.Tensor,
        bias_grad: Optional[torch.Tensor],
    ) -> tuple:
        if (
            state.ua is None
            or state.ub is None
            or state.weight_scaling is None
            or state.inverse_damping is None
        ):
            return None, None

        matrix_grad = weight_grad.reshape(weight_grad.size(0), -1)
        has_bias = bias_grad is not None
        if has_bias:
            matrix_grad = torch.cat([matrix_grad, bias_grad.reshape(-1, 1)], dim=1)

        projected = state.ub.t().matmul(matrix_grad).matmul(state.ua)
        projected = projected / (state.weight_scaling + state.inverse_damping)
        preconditioned = state.ub.matmul(projected).matmul(state.ua.t())

        if has_bias:
            bias_update = preconditioned[:, -1].contiguous().view_as(bias_grad)
            weight_update = preconditioned[:, :-1].contiguous().view_as(weight_grad)
            return weight_update, bias_update

        return preconditioned.contiguous().view_as(weight_grad), None

    def _precondition_gradient(
        self,
        name: str,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        fisher = self._diag_fisher.get(name)
        if fisher is None:
            return grad
        damping = self._inverse_damping(fisher)
        return grad / (fisher + damping)

    def _inverse_damping(self, scaling: torch.Tensor) -> torch.Tensor:
        if self.damping_mode == "absolute":
            base = scaling.new_tensor(max(self.damping, self.damping_floor))
        else:
            base = torch.clamp(
                self.damping_ratio * scaling.mean(),
                min=self.damping_floor,
            )
        return base + self.regularization_curvature

    def save_unlearned_model(self, save_path: str) -> None:
        torch.save({
            "model_state_dict": self.model.model.state_dict(),
            "method": self.name,
            "step_size": self.step_size,
            "damping_mode": self.damping_mode,
            "damping": self.damping,
            "damping_ratio": self.damping_ratio,
            "damping_floor": self.damping_floor,
            "regularization_curvature": self.regularization_curvature,
            "update_unsupported_params": self.update_unsupported_params,
            "num_unlearn_steps": self.num_unlearn_steps,
            "recompute_curvature_each_step": self.recompute_curvature_each_step,
            "unlearn_eval_interval": self.unlearn_eval_interval,
            "max_eval_batches": self.max_eval_batches,
            "forget_update_mode": self.forget_update_mode,
        }, save_path)
