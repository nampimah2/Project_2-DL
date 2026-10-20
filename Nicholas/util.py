"""Data, evaluation, confidence interval, and Integrated Gradients helpers."""
from pathlib import Path
import csv
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", " pear", " pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", " rocket", "rose",
    "sea", "seal", "shark", "shrew", "skunk", " skyscraper", "snail", "snake", "spider", "squirrel", "streetcar",
    "sunflower", "sweet_pepper", "table", "tank", "telephone", " television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]
# TFDS label order is canonical; clean incidental whitespace in display names.
CLASS_NAMES = [n.strip() for n in CLASS_NAMES]


def _limit_dataset_threads(dataset, threads=2):
    options = tf.data.Options()
    options.threading.private_threadpool_size = threads
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options)


def load_cifar100(validation_fraction=0.1, seed=42, threads=2):
    """Load CIFAR-100 and make a deterministic, stratified train/validation split."""
    train_ds, test_ds = tfds.load("cifar100", split=["train", "test"],
                                  as_supervised=True, shuffle_files=False)
    # Bound TFDS iteration threads as well as model-training threads. This is
    # important on shared HPC/login nodes with strict per-process thread caps.
    train_ds = _limit_dataset_threads(train_ds, threads)
    test_ds = _limit_dataset_threads(test_ds, threads)
    def collect(ds):
        images, labels = [], []
        for image, label in tfds.as_numpy(ds):
            images.append(image)
            labels.append(int(label))
        x = np.asarray(images, dtype=np.float32) / 255.0
        labels = np.asarray(labels, dtype=np.int64)
        y = tf.keras.utils.to_categorical(labels, 100).astype(np.float32)
        return x, y, labels
    train_x, train_y, train_labels = collect(train_ds)
    test_x, test_y, _ = collect(test_ds)
    indices = np.arange(len(train_x))
    train_idx, val_idx = train_test_split(
        indices, test_size=validation_fraction, random_state=seed,
        stratify=train_labels, shuffle=True,
    )
    return (train_x[train_idx], train_y[train_idx]), (train_x[val_idx], train_y[val_idx]), (test_x, test_y)


def make_dataset(data, batch_size, training=False, seed=42, threads=2):
    x, y = data
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(min(len(x), 50000), seed=seed, reshuffle_each_iteration=True)
        def augment(image, label):
            image = tf.image.resize_with_crop_or_pad(image, 40, 40)
            image = tf.image.random_crop(image, [32, 32, 3], seed=seed)
            image = tf.image.random_flip_left_right(image, seed=seed)
            return image, label
        ds = ds.map(augment, num_parallel_calls=threads)
    return _limit_dataset_threads(ds.batch(batch_size).prefetch(1), threads)


def wilson_interval(successes, n, z=1.959963984540054):
    """95% Wilson score interval (returns lower, upper) for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * np.sqrt((p*(1-p) + z*z/(4*n))/n) / denom
    return float(max(0, center-half)), float(min(1, center+half))


def evaluate_model(model, x, y, output_dir, run_name):
    probs = model.predict(x, batch_size=256, verbose=0)
    true = np.argmax(y, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = int(np.sum(true == pred))
    n = len(true)
    lo, hi = wilson_interval(correct, n)
    err_lo, err_hi = wilson_interval(n-correct, n)
    top5 = float(np.mean([t in np.argpartition(p, -5)[-5:] for t, p in zip(true, probs)]))
    return {
        "run": run_name, "n_test": n, "test_accuracy": correct/n,
        "accuracy_ci95_low": lo, "accuracy_ci95_high": hi,
        "generalization_error": 1-correct/n,
        "error_ci95_low": err_lo, "error_ci95_high": err_hi,
        "top5_accuracy": top5,
        "test_loss": float(model.evaluate(x, y, batch_size=256, verbose=0)[0]),
        "true": true, "pred": pred, "probs": probs,
    }


def save_normalized_confusion(true, pred, output_path):
    cm = np.zeros((100, 100), dtype=np.float64)
    np.add.at(cm, (true, pred), 1)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(normalized, cmap="magma", vmin=0, vmax=max(.2, normalized.max()))
    ax.set(title="Row-normalized test confusion matrix", xlabel="Predicted class", ylabel="True class")
    ax.set_xticks(np.arange(100)); ax.set_yticks(np.arange(100))
    ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=4)
    ax.set_yticklabels(CLASS_NAMES, fontsize=4)
    fig.colorbar(im, ax=ax, label="Fraction of true-class examples")
    fig.tight_layout(); fig.savefig(output_path, dpi=220); plt.close(fig)
    return normalized


def integrated_gradients(model, image, target_class, steps=64):
    """Integrated Gradients from a black baseline to an image for target_class."""
    image = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    baseline = tf.zeros_like(image)
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    interpolated = baseline + alphas[:, None, None, None] * (image - baseline)
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        predictions = model(interpolated, training=False)
        score = predictions[:, target_class]
    grads = tape.gradient(score, interpolated)
    avg_grads = tf.reduce_mean((grads[:-1] + grads[1:]) / 2.0, axis=0)
    attr = (image[0] - baseline[0]) * avg_grads
    return attr.numpy()


def choose_ig_examples(true, pred, probs):
    """For one class, choose high-confidence correct, low-confidence correct,
    and high-confidence incorrect examples. The class is chosen to have all 3.
    If none has all 3, use the class with the most available categories.
    """
    confidence = probs.max(axis=1)
    true_conf = probs[np.arange(len(true)), true]
    correct = true == pred
    best = None
    for class_id in range(100):
        correct_ids = np.where((true == class_id) & correct)[0]
        wrong_ids = np.where((true == class_id) & ~correct)[0]
        candidates = {}
        if len(correct_ids) >= 2:
            candidates["high_confidence_correct"] = correct_ids[np.argmax(true_conf[correct_ids])]
            candidates["low_confidence_correct"] = correct_ids[np.argmin(true_conf[correct_ids])]
        if len(wrong_ids):
            candidates["high_confidence_misclassified"] = wrong_ids[np.argmax(confidence[wrong_ids])]
        if len(candidates) == 3:
            return class_id, candidates
        if best is None or len(candidates) > len(best[1]):
            best = (class_id, candidates)
    return best


def save_ig_figure(model, x, true, pred, probs, output_path, steps=64):
    class_id, examples = choose_ig_examples(true, pred, probs)
    if not examples:
        raise RuntimeError("No eligible test examples found for Integrated Gradients.")
    fig, axes = plt.subplots(len(examples), 3, figsize=(10, 3.2*len(examples)), squeeze=False)
    for row, (criterion, idx) in enumerate(examples.items()):
        target = int(pred[idx])
        attr = integrated_gradients(model, x[idx], target, steps=steps)
        heat = np.sum(np.abs(attr), axis=-1)
        heat /= max(float(heat.max()), 1e-12)
        axes[row, 0].imshow(x[idx]); axes[row, 0].set_title(f"{criterion.replace('_',' ')}\ntrue: {CLASS_NAMES[true[idx]]}")
        axes[row, 1].imshow(heat, cmap="inferno"); axes[row, 1].set_title(f"IG magnitude\ntarget: {CLASS_NAMES[target]}")
        axes[row, 2].imshow(x[idx]); axes[row, 2].imshow(heat, cmap="inferno", alpha=.48); axes[row, 2].set_title(f"overlay | p={probs[idx,target]:.2f}")
        for ax in axes[row]: ax.axis("off")
    fig.suptitle(f"Integrated Gradients examples (shared true class: {CLASS_NAMES[class_id]})", y=1.01)
    fig.tight_layout(); fig.savefig(output_path, dpi=220, bbox_inches="tight"); plt.close(fig)
    return {k: {"index": int(i), "true_class": CLASS_NAMES[int(true[i])],
                "predicted_class": CLASS_NAMES[int(pred[i])],
                "confidence": float(probs[i, pred[i]])} for k, i in examples.items()}


def write_results_csv(records, output_path):
    fields = ["run", "architecture", "learning_rate", "dropout", "l2_strength", "best_epoch",
              "test_loss", "test_accuracy", "accuracy_ci95_low", "accuracy_ci95_high",
              "generalization_error", "error_ci95_low", "error_ci95_high", "top5_accuracy"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)


def save_training_plot(histories, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        axes[0].plot(h["loss"], label=f"{name} train")
        axes[0].plot(h["val_loss"], linestyle="--", label=f"{name} val")
        axes[1].plot(h["accuracy"], label=f"{name} train")
        axes[1].plot(h["val_accuracy"], linestyle="--", label=f"{name} val")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for ax in axes: ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output_path, dpi=200); plt.close(fig)
