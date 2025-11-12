"""Training loop for the Hyena2DForecaster."""

import os
import sys
import warnings
from datetime import datetime
from types import SimpleNamespace
import gc
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import autocast, GradScaler

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
except NameError:
    current_dir = os.getcwd()
    while not os.path.exists(os.path.join(current_dir, "data_provider")):
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            project_root = r"C:\\Users\\USER\\Desktop\\baseline\\icepixel"
            break
        current_dir = parent
    else:
        project_root = current_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root: {project_root}")

from data_provider import data_provider
from src import (
    calc_metrics,
    plot_spatial_comparison,
    plot_timeseries,
    create_metric_table,
)
from models.hyena_operator import create_hyena2d_model

warnings.filterwarnings("ignore")


def get_config() -> SimpleNamespace:
    """Default experiment configuration."""
    config = SimpleNamespace(
        root_path="C:/Users/USER/Desktop/ice/data/NSIDC_Data",
        train_years=(2013, 2020),
        val_years=(2021, 2021),
        test_years=(2022, 2022),
        seq_input=30,
        output_lens=[7, 14, 21],
        model_name="Hyena2D",
        n_layers=3,
        hidden_dim=32,
        temporal_dim=256,
        hyena_order=2,
        hyena_filter_order=64,
        hyena_dropout=0.1,
        hyena_filter_dropout=0.1,
        input_size=(448, 304),
        batch_size=1,
        num_workers=2,
        Epoch=30,
        lr=0.001,
        use_amp=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_model=True,
        verbose=True,
        cache_in_memory=True,
        plot_interval=10,
    )
    return config


def train_epoch(model, train_loader, optimizer, device, use_amp=False, scaler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        batch_x = batch["input"].to(device)
        batch_y = batch["target"].to(device)
        mask = batch["mask"].to(device=device, dtype=torch.bool)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast():
                output = model(batch_x)
                mask_expanded = mask.unsqueeze(1).unsqueeze(1).expand_as(output)
                valid_count = mask_expanded.sum().item()
                if valid_count == 0:
                    continue
                output_valid = output.masked_select(mask_expanded)
                target_valid = batch_y.masked_select(mask_expanded)
                loss = F.mse_loss(output_valid, target_valid)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(batch_x)
            mask_expanded = mask.unsqueeze(1).unsqueeze(1).expand_as(output)
            valid_count = mask_expanded.sum().item()
            if valid_count == 0:
                continue
            output_valid = output.masked_select(mask_expanded)
            target_valid = batch_y.masked_select(mask_expanded)
            loss = F.mse_loss(output_valid, target_valid)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, test_loader, device):
    model.eval()

    metrics_accum = defaultdict(float)
    batch_count = 0
    sample = None

    with torch.no_grad():
        for batch in test_loader:
            batch_x = batch["input"].to(device)
            batch_y = batch["target"].to(device)
            mask_cpu = batch["mask"].clone()
            mask = mask_cpu.to(device=device, dtype=torch.bool)

            output = model(batch_x)

            batch_metrics = calc_metrics(
                output.detach().cpu(),
                batch_y.detach().cpu(),
                mask_cpu.numpy(),
                pixel_area_km2=625.0,
            )

            for key, value in batch_metrics.items():
                metrics_accum[key] += float(value)
            batch_count += 1

            if sample is None:
                dates_out = None
                if "dates_out" in batch and batch["dates_out"]:
                    dates_out = batch["dates_out"][0][-1]
                sample = {
                    "pred": output[0].detach().cpu(),
                    "true": batch_y[0].detach().cpu(),
                    "mask": mask_cpu[0].detach().cpu(),
                    "date": dates_out,
                }

    if batch_count == 0:
        avg_metrics = {
            "RMSE": float("inf"),
            "MAE": float("inf"),
            "R2": -float("inf"),
            "SIE_pred": 0.0,
            "SIE_true": 0.0,
            "SIE_error_pct": 0.0,
        }
    else:
        avg_metrics = {key: metrics_accum[key] / batch_count for key in metrics_accum}

    return avg_metrics, sample


def main():
    config = get_config()

    print("=" * 80)
    print("Hyena 2D Forecaster Training")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Input sequence: {config.seq_input} days")
    print(f"Output sequences: {config.output_lens}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.Epoch}")
    print(f"Learning rate: {config.lr}")
    print("=" * 80)

    all_results = {}

    for seq_output in config.output_lens:
        print(f"\n{'='*80}")
        print(f"Experiment: Input={config.seq_input} days → Output={seq_output} days")
        print(f"{'='*80}\n")

        print("[1] Creating Dataloaders...")
        args = SimpleNamespace(
            root_path=config.root_path,
            seq_len=config.seq_input,
            pred_len=seq_output,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            train_years=config.train_years,
            val_years=config.val_years,
            test_years=config.test_years,
            verbose=config.verbose,
            cache_in_memory=config.cache_in_memory,
        )

        train_dataset, train_loader = data_provider(args, split="train")
        test_dataset, test_loader = data_provider(args, split="test")

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")

        print("\n[2] Creating Model...")
        model = create_hyena2d_model(
            input_size=config.input_size,
            seq_input=config.seq_input,
            seq_output=seq_output,
            n_layers=config.n_layers,
            hidden_dim=config.hidden_dim,
            temporal_dim=config.temporal_dim,
            hyena_order=config.hyena_order,
            hyena_filter_order=config.hyena_filter_order,
            hyena_dropout=config.hyena_dropout,
            hyena_filter_dropout=config.hyena_filter_dropout,
            device=config.device,
        )

        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        use_amp = config.use_amp and str(config.device).startswith("cuda")
        scaler = GradScaler(enabled=use_amp)

        best_metrics = None
        best_rmse = float("inf")

        now = datetime.now()
        save_dir = os.path.join(
            project_root,
            "results",
            config.model_name,
            f"seq_{seq_output}_{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}",
        )
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)

        print(f"\n[3] Training for {config.Epoch} epochs...")
        print(f"  Results will be saved to: {save_dir}")

        for epoch in range(config.Epoch):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                config.device,
                use_amp=use_amp,
                scaler=scaler if use_amp else None,
            )

            metrics, sample = evaluate(model, test_loader, config.device)

            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_metrics = metrics.copy()
                best_metrics["best_epoch"] = epoch

                if config.save_model:
                    model_path = os.path.join(save_dir, "best_model.pt")
                    torch.save(model.state_dict(), model_path)

            print(f"\n[Epoch {epoch+1}/{config.Epoch}]")
            print(f"  Train Loss: {train_loss:.6f}")
            best_rmse_display = best_metrics["RMSE"] if best_metrics else float("inf")
            best_mae_display = best_metrics["MAE"] if best_metrics else float("inf")
            best_r2_display = best_metrics["R2"] if best_metrics else -float("inf")

            print(f"  RMSE: {metrics['RMSE']:.6f}  |  Best: {best_rmse_display:.6f}")
            print(f"  MAE:  {metrics['MAE']:.6f}  |  Best: {best_mae_display:.6f}")
            print(f"  R²:   {metrics['R2']:.6f}  |  Best: {best_r2_display:.6f}")

            if sample and (epoch % config.plot_interval == 0 or epoch == config.Epoch - 1):
                print("  Saving visualizations...")

                plot_spatial_comparison(
                    sample["pred"][-1, 0],
                    sample["true"][-1, 0],
                    date=sample.get("date") or test_dataset.file_list[0][0],
                    save_path=os.path.join(save_dir, "plots", f"epoch_{epoch:03d}_spatial.png"),
                )

                plot_timeseries(
                    sample["pred"][:, 0],
                    sample["true"][:, 0],
                    model_name=config.model_name,
                    seq_output=seq_output,
                    mask=sample["mask"],
                    save_path=os.path.join(save_dir, "plots", f"epoch_{epoch:03d}_timeseries.png"),
                )

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        if best_metrics:
            print(f"Best Epoch: {best_metrics.get('best_epoch', 'N/A')}")
            print(f"Best RMSE: {best_metrics['RMSE']:.6f}")
            print(f"Best MAE: {best_metrics['MAE']:.6f}")
            print(f"Best R²: {best_metrics['R2']:.6f}")
        else:
            print("Best metrics not computed (no evaluation batches).")
        print(f"{'='*60}\n")

        if best_metrics:
            create_metric_table(
                best_metrics,
                save_path=os.path.join(save_dir, "best_metrics.csv"),
            )
            print("Metrics saved to:", os.path.join(save_dir, "best_metrics.csv"))

        all_results[seq_output] = best_metrics if best_metrics else {}

        del model, optimizer, train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("Final Summary - All Experiments")
    print("=" * 80)
    for seq_output, metrics in all_results.items():
        print(f"\nOutput Length: {seq_output} days")
        if metrics:
            print(f"  RMSE: {metrics['RMSE']:.6f}")
            print(f"  MAE:  {metrics['MAE']:.6f}")
            print(f"  R²:   {metrics['R2']:.6f}")
        else:
            print("  Metrics unavailable (no evaluation data).")
    print("=" * 80)


if __name__ == "__main__":
    main()
