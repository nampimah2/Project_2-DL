"""Train and compare CIFAR-100 CNNs.

Example: python main.py --output-dir results --epochs 100 --patience 8
"""
import argparse
import json
import os
import random
from pathlib import Path

# Many shared HPC nodes expose far more CPUs than a user process is allowed
# to spawn threads for. Set these before importing TensorFlow.
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import tensorflow as tf
from model import build_model
from util import (load_cifar100, make_dataset, evaluate_model,
                  save_normalized_confusion, save_ig_figure,
                  write_results_csv, save_training_plot)

CONFIGS = [
    {"architecture": "compact", "learning_rate": 1e-3, "dropout": .30, "l2_strength": 1e-4},
    {"architecture": "compact", "learning_rate": 3e-4, "dropout": .50, "l2_strength": 5e-4},
    {"architecture": "deep", "learning_rate": 1e-3, "dropout": .30, "l2_strength": 1e-4},
    {"architecture": "deep", "learning_rate": 3e-4, "dropout": .50, "l2_strength": 5e-4},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="results")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=2,
                   help="Limit TensorFlow/data-pipeline threads for shared HPC nodes.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ig-steps", type=int, default=64)
    p.add_argument("--limit-configs", type=int, default=4,
                   help="Use 4 for the full required grid; lower values are for smoke tests only.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.patience < 5:
        raise ValueError("Assignment requirement: early-stopping patience must be at least 5.")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); tf.keras.utils.set_random_seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_cifar100(seed=args.seed, threads=args.threads)
    train_ds = make_dataset((train_x, train_y), args.batch_size, training=True,
                            seed=args.seed, threads=args.threads)
    val_ds = make_dataset((val_x, val_y), args.batch_size, threads=args.threads)
    summaries, histories, retained_models = [], {}, []
    for config in CONFIGS[:args.limit_configs]:
        run_name = f"{config['architecture']}_lr{config['learning_rate']}_drop{config['dropout']}"
        print(f"\n=== {run_name} ===", flush=True)
        model = build_model(**config)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                             min_delta=1e-4, restore_best_weights=True, mode="min"),
            tf.keras.callbacks.ModelCheckpoint(str(out / f"{run_name}.keras"), monitor="val_loss",
                                               save_best_only=True, mode="min"),
            tf.keras.callbacks.CSVLogger(str(out / f"{run_name}_epochs.csv")),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=max(3,args.patience//2),
                                                 min_lr=1e-6, verbose=1),
        ]
        hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)
        histories[run_name] = hist.history
        model.load_weights(str(out / f"{run_name}.keras"))
        ev = evaluate_model(model, test_x, test_y, out, run_name)
        record = {**config, **{k:v for k,v in ev.items() if k not in ("true","pred","probs")},
                  "best_epoch": int(np.argmin(hist.history["val_loss"]) + 1)}
        summaries.append(record); retained_models.append((record, model, ev))
        print(json.dumps(record, indent=2), flush=True)
    # Candidate selection uses validation loss only; test set is reported for comparison.
    best_i = int(np.argmin([min(h["val_loss"]) for h in histories.values()]))
    best_record, best_model, best_eval = retained_models[best_i]
    selected = best_record["run"]
    save_normalized_confusion(best_eval["true"], best_eval["pred"], out / "normalized_confusion_matrix.png")
    ig_selection = save_ig_figure(best_model, test_x, best_eval["true"], best_eval["pred"],
                                  best_eval["probs"], out / "integrated_gradients_examples.png", args.ig_steps)
    write_results_csv(summaries, out / "results.csv")
    (out / "ig_example_selection.json").write_text(json.dumps({"selected_model": selected,
        "selection_rule": "Within one shared true class, select highest-confidence correct, lowest-confidence correct, and highest-confidence incorrect examples when available.",
        "examples": ig_selection}, indent=2), encoding="utf-8")
    save_training_plot(histories, out / "training_curves.png")
    print(f"\nSelected by minimum validation loss: {selected}")
    print(f"Artifacts written to: {out.resolve()}")

if __name__ == "__main__":
    main()
