"""Train and compare CIFAR-100 CNNs.

Run the full experiment with:
    python main.py

If training is interrupted, run the same command again. Completed configurations
are reused, and an interrupted configuration resumes from its latest epoch backup.
"""
import argparse
import csv
import json
import os
import random
import shutil
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
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=2,
                   help="Limit TensorFlow/data-pipeline threads for shared HPC nodes.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ig-steps", type=int, default=64)
    p.add_argument("--limit-configs", type=int, default=4,
                   help="Use 4 for the full required grid; lower values are for smoke tests only.")
    return p.parse_args()


def load_history_csv(csv_path):
    """Load the complete epoch history from a CSVLogger file."""
    history = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        raise RuntimeError(f"Epoch log is empty: {csv_path}")

    history["_epoch"] = [
        int(float(row["epoch"]))
        for row in rows
    ]

    for key in fieldnames:
        if key == "epoch":
            continue

        values = []

        for row in rows:
            value = row.get(key, "")

            if value not in ("", None):
                values.append(float(value))

        history[key] = values

    return history


def build_record(config, history, evaluation):
    """Create one summary record from a completed run."""
    best_index = int(np.argmin(history["val_loss"]))
    best_epoch = int(
        history["_epoch"][best_index] + 1
    )

    return {
        **config,
        **{
            k: v
            for k, v in evaluation.items()
            if k not in ("true", "pred", "probs")
        },
        "best_epoch": best_epoch,
    }


def main():
    args = parse_args()

    if args.patience < 5:
        raise ValueError(
            "Assignment requirement: early-stopping patience must be at least 5."
        )

    if args.threads < 1:
        raise ValueError(
            "--threads must be at least 1"
        )

    if args.limit_configs < 1 or args.limit_configs > len(CONFIGS):
        raise ValueError(
            f"--limit-configs must be between 1 and {len(CONFIGS)}"
        )

    tf.config.threading.set_intra_op_parallelism_threads(
        args.threads
    )

    tf.config.threading.set_inter_op_parallelism_threads(
        args.threads
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    (out / "run_config.json").write_text(
        json.dumps(
            vars(args),
            indent=2
        ),
        encoding="utf-8"
    )

    (
        (train_x, train_y),
        (val_x, val_y),
        (test_x, test_y)
    ) = load_cifar100(
        seed=args.seed,
        threads=args.threads
    )

    train_ds = make_dataset(
        (train_x, train_y),
        args.batch_size,
        training=True,
        seed=args.seed,
        threads=args.threads
    )

    val_ds = make_dataset(
        (val_x, val_y),
        args.batch_size,
        threads=args.threads
    )

    summaries = []
    histories = {}
    retained_models = []

    for config in CONFIGS[:args.limit_configs]:
        run_name = (
            f"{config['architecture']}_"
            f"lr{config['learning_rate']}_"
            f"drop{config['dropout']}"
        )

        print(
            f"\n=== {run_name} ===",
            flush=True
        )

        model = build_model(
            **config
        )

        checkpoint_path = (
            out /
            f"{run_name}.weights.h5"
        )

        epoch_log_path = (
            out /
            f"{run_name}_epochs.csv"
        )

        completion_path = (
            out /
            f"{run_name}_complete.json"
        )

        backup_dir = (
            out /
            "resume_backups" /
            run_name
        )

        backup_exists = (
            backup_dir.exists()
            and any(backup_dir.iterdir())
        )

        # A completed configuration does not need to be trained again.
        if completion_path.exists():
            if (
                not checkpoint_path.exists()
                or not epoch_log_path.exists()
            ):
                raise RuntimeError(
                    f"{run_name} is marked complete, but its "
                    "checkpoint or epoch log is missing."
                )

            print(
                f"{run_name} is already complete. "
                "Reusing saved results.",
                flush=True
            )

            history = load_history_csv(
                epoch_log_path
            )

            histories[run_name] = history

            model.load_weights(
                str(checkpoint_path)
            )

            ev = evaluate_model(
                model,
                test_x,
                test_y,
                out,
                run_name
            )

            record = build_record(
                config,
                history,
                ev
            )

            summaries.append(
                record
            )

            retained_models.append(
                (record, model, ev)
            )

            print(
                json.dumps(
                    record,
                    indent=2
                ),
                flush=True
            )

            continue

        # If an interrupted run is present, both the recovery backup and
        # the associated epoch/checkpoint files must be available.
        if backup_exists:
            if (
                not epoch_log_path.exists()
                or not checkpoint_path.exists()
            ):
                raise RuntimeError(
                    f"A resume backup exists for {run_name}, "
                    "but its epoch log or best-model checkpoint "
                    "is missing."
                )

        elif (
            epoch_log_path.exists()
            or checkpoint_path.exists()
        ):
            raise RuntimeError(
                f"Partial files exist for {run_name}, but no "
                "resume backup was found. For a completely fresh "
                "experiment, delete the results directory and run "
                "python main.py again."
            )

        initial_best_val_loss = None

        if backup_exists:
            existing_history = load_history_csv(
                epoch_log_path
            )

            completed_epochs = int(
                existing_history["_epoch"][-1] + 1
            )

            if (
                "val_loss" in existing_history
                and existing_history["val_loss"]
            ):
                initial_best_val_loss = float(
                    min(
                        existing_history[
                            "val_loss"
                        ]
                    )
                )

            print(
                f"Interrupted run detected for {run_name}. "
                f"Resuming after epoch {completed_epochs}.",
                flush=True
            )

        backup_callback = (
            tf.keras.callbacks.BackupAndRestore(
                backup_dir=str(
                    backup_dir
                ),
                save_freq="epoch",
                delete_checkpoint=False
            )
        )

        early_stopping = (
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                min_delta=1e-4,
                restore_best_weights=True,
                mode="min"
            )
        )

        callbacks = [
            backup_callback,

            early_stopping,

            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                mode="min",
                initial_value_threshold=(
                    initial_best_val_loss
                )
            ),

            tf.keras.callbacks.CSVLogger(
                str(epoch_log_path),
                append=backup_exists
            ),

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=.5,
                patience=max(
                    3,
                    args.patience // 2
                ),
                min_lr=1e-6,
                verbose=1
            ),
        ]

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=2
        )

        history = load_history_csv(
            epoch_log_path
        )

        histories[run_name] = history

        epochs_completed = int(
            history["_epoch"][-1] + 1
        )

        # If the model reaches the maximum epoch count without
        # early stopping, the run is not considered complete.
        if (
            early_stopping.stopped_epoch == 0
            and epochs_completed >= args.epochs
        ):
            raise RuntimeError(
                f"{run_name} reached the maximum of "
                f"{args.epochs} epochs before early stopping "
                "completed. Increase --epochs and run "
                "python main.py again. The saved recovery "
                "state will be reused."
            )

        best_index = int(
            np.argmin(
                history["val_loss"]
            )
        )

        completion_path.write_text(
            json.dumps(
                {
                    "run": run_name,
                    "epochs_completed": (
                        epochs_completed
                    ),
                    "best_epoch": int(
                        history["_epoch"][
                            best_index
                        ] + 1
                    ),
                    "minimum_validation_loss": float(
                        min(
                            history[
                                "val_loss"
                            ]
                        )
                    ),
                },
                indent=2
            ),
            encoding="utf-8"
        )

        # This run is complete, so its temporary recovery
        # checkpoint is no longer needed.
        shutil.rmtree(
            backup_dir,
            ignore_errors=True
        )

        # Restore the weights from the best validation-loss checkpoint.
        model.load_weights(
            str(checkpoint_path)
        )

        ev = evaluate_model(
            model,
            test_x,
            test_y,
            out,
            run_name
        )

        record = build_record(
            config,
            history,
            ev
        )

        summaries.append(
            record
        )

        retained_models.append(
            (record, model, ev)
        )

        print(
            json.dumps(
                record,
                indent=2
            ),
            flush=True
        )

    # Candidate selection uses validation loss only;
    # test set is reported for comparison.
    best_i = int(
        np.argmin(
            [
                min(
                    histories[
                        record["run"]
                    ]["val_loss"]
                )
                for record, _, _
                in retained_models
            ]
        )
    )

    (
        best_record,
        best_model,
        best_eval
    ) = retained_models[best_i]

    selected = best_record["run"]

    # Save the overall selected-model confusion matrix.
    save_normalized_confusion(
        best_eval["true"],
        best_eval["pred"],
        out /
        "normalized_confusion_matrix.png"
    )

    # Select the best run within each architecture
    # using validation loss.
    best_by_architecture = {}

    for record, model, ev in retained_models:
        architecture = (
            record["architecture"]
        )

        run_name = record["run"]

        minimum_validation_loss = min(
            histories[
                run_name
            ]["val_loss"]
        )

        if (
            architecture
            not in best_by_architecture
            or minimum_validation_loss
            < best_by_architecture[
                architecture
            ]["val_loss"]
        ):
            best_by_architecture[
                architecture
            ] = {
                "record": record,
                "model": model,
                "evaluation": ev,
                "val_loss": (
                    minimum_validation_loss
                ),
            }

    # Save one normalized confusion matrix for the
    # best Compact model and one for the best Deep model.
    if "compact" in best_by_architecture:
        compact_eval = (
            best_by_architecture[
                "compact"
            ]["evaluation"]
        )

        save_normalized_confusion(
            compact_eval["true"],
            compact_eval["pred"],
            out /
            "compact_normalized_confusion_matrix.png"
        )

    if "deep" in best_by_architecture:
        deep_eval = (
            best_by_architecture[
                "deep"
            ]["evaluation"]
        )

        save_normalized_confusion(
            deep_eval["true"],
            deep_eval["pred"],
            out /
            "deep_normalized_confusion_matrix.png"
        )

    ig_selection = save_ig_figure(
        best_model,
        test_x,
        best_eval["true"],
        best_eval["pred"],
        best_eval["probs"],
        out /
        "integrated_gradients_examples.png",
        args.ig_steps
    )

    write_results_csv(
        summaries,
        out /
        "results.csv"
    )

    (
        out /
        "ig_example_selection.json"
    ).write_text(
        json.dumps(
            {
                "selected_model": (
                    selected
                ),
                "selection_rule": (
                    "Within one shared true class, "
                    "select highest-confidence correct, "
                    "lowest-confidence correct, and "
                    "highest-confidence incorrect "
                    "examples when available."
                ),
                "examples": (
                    ig_selection
                )
            },
            indent=2
        ),
        encoding="utf-8"
    )

    save_training_plot(
        histories,
        out /
        "training_curves.png"
    )

    print(
        f"\nSelected by minimum "
        f"validation loss: {selected}"
    )

    if "compact" in best_by_architecture:
        print(
            "Best compact model by "
            "validation loss: "
            f"{best_by_architecture['compact']['record']['run']}"
        )

    if "deep" in best_by_architecture:
        print(
            "Best deep model by "
            "validation loss: "
            f"{best_by_architecture['deep']['record']['run']}"
        )

    print(
        f"Artifacts written to: "
        f"{out.resolve()}"
    )


if __name__ == "__main__":
    main()