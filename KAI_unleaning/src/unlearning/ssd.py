"""
Paper reference:
"Selective Synaptic Dampening" by Foster et al., AAAI 2024
https://github.com/if-loops/selective-synaptic-dampening
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from .base import BaseUnlearning


class SSDUnlearning(BaseUnlearning):

    name = "ssd"

    def __init__(self, model, dampening_constant: float = 1.0,
                 selection_weighting: float = 10.0, device: str = "cuda",
                 **kwargs):
        """
        Args:
            model: BaseModel instance
            dampening_constant: λ — controls dampening strength (default: 1.0, 논문 권장)
            selection_weighting: α — synapse selection threshold (default: 10.0, 논문 권장)
            device: Device to compute on
        """
        super().__init__(model, **kwargs)
        self.dampening_constant = dampening_constant
        self.selection_weighting = selection_weighting
        self.device = device

    def unlearn(self, forget_loader, train_loader, **kwargs) -> dict:
        logger = kwargs.get("logger")

        if logger:
            logger.print("\n[bold yellow]╔══════════════════════════════════════════╗[/bold yellow]")
            logger.print("[bold yellow]║       SSD Unlearning Algorithm              ║[/bold yellow]")
            logger.print("[bold yellow]╚══════════════════════════════════════════╝[/bold yellow]\n")
            logger.info(f"Configuration:")
            logger.info(f"  • Dampening constant (λ): [cyan]{self.dampening_constant}[/cyan]")
            logger.info(f"  • Selection weighting (α): [cyan]{self.selection_weighting}[/cyan]")
            logger.info(f"  • Forget set size: [cyan]{len(forget_loader.dataset)}[/cyan]")
            logger.info(f"  • Full train set size: [cyan]{len(train_loader.dataset)}[/cyan]\n")

        total_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        if logger:
            logger.info(f"  • Total parameters: [cyan]{total_params:,}[/cyan]\n")

        # Step 1: Compute Fisher on full training set (F_full)
        if logger:
            logger.print("[bold cyan]▶ Step 1/3: Computing Fisher on Full Training Set (F_full)[/bold cyan]")

        fisher_full = self._compute_fisher(train_loader, logger=logger, name="F_full")
        full_fisher_stats = self._get_fisher_stats(fisher_full)

        if logger:
            logger.info(f"  ✓ F_full computed")
            logger.info(f"    - Min: {full_fisher_stats['min']:.6f}")
            logger.info(f"    - Max: {full_fisher_stats['max']:.6f}")
            logger.info(f"    - Mean: {full_fisher_stats['mean']:.6f}")
            logger.info(f"    - Std: {full_fisher_stats['std']:.6f}\n")

        # Step 2: Compute Fisher on forget set (F_forget)
        if logger:
            logger.print("[bold cyan]▶ Step 2/3: Computing Fisher on Forget Set (F_forget)[/bold cyan]")

        fisher_forget = self._compute_fisher(forget_loader, logger=logger, name="F_forget")
        forget_fisher_stats = self._get_fisher_stats(fisher_forget)

        if logger:
            logger.info(f"  ✓ F_forget computed")
            logger.info(f"    - Min: {forget_fisher_stats['min']:.6f}")
            logger.info(f"    - Max: {forget_fisher_stats['max']:.6f}")
            logger.info(f"    - Mean: {forget_fisher_stats['mean']:.6f}")
            logger.info(f"    - Std: {forget_fisher_stats['std']:.6f}\n")

        # Step 3: Apply selective synaptic dampening
        if logger:
            logger.print("[bold cyan]▶ Step 3/3: Applying Selective Synaptic Dampening[/bold cyan]")
            logger.info("  Formula: β = min(λ * F_full / F_forget, 1)")
            logger.info("  Condition: F_forget > α * F_full\n")

        dampening_stats = self._apply_dampening(fisher_full, fisher_forget, logger)

        if logger:
            logger.info(f"  ✓ Dampening applied")
            logger.info(f"    - Parameters selected (F_forget > α*F_full): {dampening_stats['selected']:,}")
            logger.info(f"    - Parameters modified: {dampening_stats['modified']:,}")
            logger.info(f"    - Parameters preserved: {dampening_stats['preserved']:,}")
            logger.info(f"    - Average change: {dampening_stats['avg_change']:.4%}\n")

        # Log to WandB
        if logger:
            logger.log_metrics({
                "forget_fisher_mean": forget_fisher_stats["mean"],
                "forget_fisher_std": forget_fisher_stats["std"],
                "full_fisher_mean": full_fisher_stats["mean"],
                "full_fisher_std": full_fisher_stats["std"],
                "selected_params": dampening_stats["selected"],
                "modified_params": dampening_stats["modified"],
                "preserved_params": dampening_stats["preserved"],
            }, step=0, prefix="unlearning/")

        # Summary
        if logger:
            logger.print("[bold green]╔══════════════════════════════════════════╗[/bold green]")
            logger.print("[bold green]║       SSD Unlearning Completed!             ║[/bold green]")
            logger.print("[bold green]╚══════════════════════════════════════════╝[/bold green]\n")

        return {
            "method": "ssd",
            "dampening_constant": self.dampening_constant,
            "selection_weighting": self.selection_weighting,
            "forget_fisher_stats": forget_fisher_stats,
            "full_fisher_stats": full_fisher_stats,
            "dampening_stats": dampening_stats,
        }

    def _compute_fisher(self, dataloader, logger=None, name="Fisher") -> dict:
        self.model.model.eval()
        self.model.model.to(self.device)
        criterion = nn.CrossEntropyLoss()

        params = {n: p for n, p in self.model.model.named_parameters() if p.requires_grad}
        fisher_diag = {n: torch.zeros_like(p) for n, p in params.items()}
        num_batches = 0

        pbar = tqdm(total=len(dataloader), desc=f"Computing {name}", unit="batch")

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.model.zero_grad()
            outputs = self.model.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for n, p in params.items():
                if p.grad is not None:
                    fisher_diag[n] += p.grad.detach().pow(2)

            num_batches += 1
            pbar.update(1)

        pbar.close()

        fisher_diag = {n: f / num_batches for n, f in fisher_diag.items()}
        return fisher_diag

    def _get_fisher_stats(self, fisher_diag: dict) -> dict:
        all_fisher = torch.cat([f.flatten() for f in fisher_diag.values()])
        return {
            "min": all_fisher.min().item(),
            "max": all_fisher.max().item(),
            "mean": all_fisher.mean().item(),
            "std": all_fisher.std().item(),
        }

    def _apply_dampening(self, fisher_full: dict, fisher_forget: dict,
                         logger=None) -> dict:
        selected_params = 0
        modified_params = 0
        preserved_params = 0
        total_params = 0
        total_change = 0.0

        with torch.no_grad():
            for n, p in self.model.model.named_parameters():
                if not p.requires_grad:
                    continue

                f_full = fisher_full[n]
                f_forget = fisher_forget[n]

                # Synapse Selection: F_forget > α * F_full
                locations = torch.where(f_forget > self.selection_weighting * f_full)
                selected_params += locations[0].numel()

                # Synapse Dampening: β = min(λ * F_full / F_forget, 1)
                # Guard against division by zero
                safe_f_forget = f_forget[locations].clamp(min=1e-10)
                beta = (self.dampening_constant * f_full[locations] / safe_f_forget).clamp(max=1.0)

                modified_params += locations[0].numel()

                # Apply dampening: θ_i ← β * θ_i
                p[locations] = p[locations] * beta

                # Statistics
                total_params += p.numel()
                if locations[0].numel() > 0:
                    total_change += (1 - beta).sum().item()

        preserved_params = total_params - modified_params

        return {
            "selected": selected_params,
            "modified": modified_params,
            "preserved": preserved_params,
            "total": total_params,
            "avg_change": total_change / total_params if total_params > 0 else 0,
        }

    def save_unlearned_model(self, save_path: str) -> None:
        torch.save({
            "model_state_dict": self.model.model.state_dict(),
            "dampening_constant": self.dampening_constant,
            "selection_weighting": self.selection_weighting,
        }, save_path)
