"""
Run the additional loss-ablation experiments and print the rows for the paper's loss table.

    python run_loss_ablation.py                      # sfmi, focal_a75, hard_afm_l0u1
    python run_loss_ablation.py --losses sfmi        # only the most important one
    python run_loss_ablation.py --summary_only       # print the table from finished runs

Every loss gets its own output folder (benchmark.py's result file name does not contain the
loss, so sharing a folder would overwrite earlier results). Finished combinations are skipped,
so an interrupted run (for example on Colab) can simply be started again.
"""
import argparse
import json
import os
import subprocess
import sys

DEFAULT_LOSSES = ["sfmi", "focal_a75", "hard_afm_l0u1"]
DATASETS = ["floodkaggle", "floodscene"]
LABELS = {
    "sfmi": r"$1-\mathrm{sFMI}$ (no gate)",
    "focal_a75": r"Focal ($\alpha=0.75$)",
    "hard_afm_l0u1": r"HFM ($l=0$, $u=1$)",
    "hard_afm": r"HFM (Ours)",
    "focal": r"Focal ($\alpha=0.25$)",
    "dice": "Dice",
    "bce": "BCE",
}


def result_path(args, loss, dataset):
    return os.path.join(args.output_path, loss, f"{args.model}_{dataset}_multiseed.json")


def finished(args, loss, dataset):
    """The stored result, only if it was produced with exactly the requested settings."""
    path = result_path(args, loss, dataset)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    cfg = data["config"]
    same = (cfg["model"], cfg["dataset"], cfg["loss"], cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg["size"]) == \
        (args.model, dataset, loss, args.epochs, args.batch_size, args.lr, args.size)
    return data if same and data["seeds"] == args.seeds else None


def main():
    ap = argparse.ArgumentParser(description="Loss ablation runs")
    ap.add_argument("--losses", nargs="+", default=DEFAULT_LOSSES, choices=list(LABELS))
    ap.add_argument("--model", default="ours")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024])
    ap.add_argument("--output_path", default="outputs/loss_ablation")
    ap.add_argument("--summary_only", action="store_true")
    args = ap.parse_args()

    if not args.summary_only:
        for loss in args.losses:
            for dataset in DATASETS:
                if finished(args, loss, dataset):
                    print(f"[SKIP] {loss} / {dataset}: already done")
                    continue
                print(f"\n########## {loss} / {dataset} ##########", flush=True)
                cmd = [sys.executable, "benchmark.py", "--dataset", dataset, "--model", args.model,
                       "--size", str(args.size), "--loss", loss, "--epochs", str(args.epochs),
                       "--batch_size", str(args.batch_size), "--lr", str(args.lr),
                       "--output_path", os.path.join(args.output_path, loss), "--multiseed",
                       "--seeds", *map(str, args.seeds), "--download"]
                if subprocess.run(cmd).returncode != 0:
                    sys.exit(f"[ERROR] run failed: {loss} / {dataset}")

    print("\n" + "=" * 90)
    print("LOSS ABLATION (mean +- std over seeds)   IoU / DSC on Flood Area Segmentation | AIFloodSense")
    print("=" * 90)
    rows = []
    for loss in args.losses:
        cells, latex = [], []
        for dataset in DATASETS:
            data = finished(args, loss, dataset)
            for metric in ("miou", "dice"):            # 'miou' is the pooled flood-class IoU
                if data is None:
                    cells.append("      --       ")
                    latex.append("--")
                else:
                    s = data["statistics"][metric]
                    cells.append(f"{s['mean']:.4f} +- {s['std']:.4f}")
                    latex.append(f"${s['mean']:.4f} \\pm {s['std']:.4f}$")
        print(f"{loss:14s} {cells[0]}  {cells[1]} | {cells[2]}  {cells[3]}")
        rows.append(f"{LABELS[loss]} & " + " & ".join(latex) + r" \\")
    print("\nLaTeX rows:")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
