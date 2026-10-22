"""23CSE301 Lab Session 07 — Decision Trees on project data (Audelle/GTZAN).

Dataset (own project data): GTZAN subset from the same Hugging Face mirror
used by scripts/train_genre_classifier.py (storylinez/gtzan-music-genre-dataset).
4 genres x 6 clips, 5 librosa features per clip, genre label as the target.

Covers A1 (entropy), A2 (Gini), A3 (root node via information gain),
A4 (binning with defaults), A5 (own decision-tree module), A6 (tree plot),
A7 (2-feature decision boundary), A8 (GridSearchCV tuning).

Coding instructions followed: modular functions, prints only in main,
proper variable names, inline comments. Run:
    python scripts/lab07_decision_tree.py
Outputs: scripts/lab07_fig_tree.png, scripts/lab07_fig_boundary.png,
         scripts/lab07_metrics.json
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# --- Project dataset configuration (same mirror as train_genre_classifier) ---
DATASET_BASE = (
    "https://huggingface.co/datasets/storylinez"
    "/gtzan-music-genre-dataset/resolve/main"
)
PROJECT_GENRES = ["disco", "hiphop", "metal", "classical"]
CLIPS_PER_GENRE = 6
SAMPLE_RATE = 22050
CLIP_SECONDS = 10.0
FEATURE_NAMES = [
    "centroid_mean",
    "zcr_mean",
    "rms_mean",
    "flatness_mean",
    "tempo",
]


# ---------------------------------------------------------------------------
# Data loading (project data)
# ---------------------------------------------------------------------------
def download_project_subset(root_dir: Path, clips_per_genre: int = 6):
    """Download clips_per_genre clips for each project genre into root_dir."""
    downloaded = []
    session = requests.Session()
    for genre in PROJECT_GENRES:
        genre_dir = root_dir / genre
        genre_dir.mkdir(parents=True, exist_ok=True)
        collected = 0
        file_index = 0
        while collected < clips_per_genre:
            file_name = f"{genre}.{file_index:05d}.wav"
            url = f"{DATASET_BASE}/{genre}/{file_name}"
            dest = genre_dir / file_name
            if not dest.exists():
                response = session.get(url, timeout=60)
                if response.status_code == 404:
                    file_index += 1
                    continue
                response.raise_for_status()
                dest.write_bytes(response.content)
            downloaded.append((dest, genre))
            collected += 1
            file_index += 1
            if file_index > 500:
                raise RuntimeError(f"could not find clips for {genre}")
    return downloaded


def extract_feature_row(audio_path: Path):
    """Extract one fixed-length numeric feature row from an audio clip."""
    signal, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True,
                             duration=CLIP_SECONDS)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=SAMPLE_RATE)[0]
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    rms = librosa.feature.rms(y=signal)[0]
    flatness = librosa.feature.spectral_flatness(y=signal)[0]
    tempo, _ = librosa.beat.beat_track(y=signal, sr=SAMPLE_RATE)
    return [
        float(np.mean(centroid)),
        float(np.mean(zcr)),
        float(np.mean(rms)),
        float(np.mean(flatness)),
        float(np.atleast_1d(tempo)[0]),
    ]


# ---------------------------------------------------------------------------
# A1. Entropy (+ equal-width binning helper for continuous outcomes)
# ---------------------------------------------------------------------------
def equal_width_binning(values, n_bins: int = 4):
    """Map continuous values to bin ids 0..n_bins-1 with equal-width edges."""
    arr = np.asarray(values, dtype=float)
    bin_ids = np.zeros(len(arr), dtype=int)
    if len(arr) == 0:
        return bin_ids.tolist()
    if float(np.max(arr)) == float(np.min(arr)):
        return bin_ids.tolist()
    edges = np.linspace(float(np.min(arr)), float(np.max(arr)), n_bins + 1)
    bin_ids = np.clip(np.digitize(arr, edges[1:-1]), 0, n_bins - 1)
    return bin_ids.tolist()


def calculate_entropy(labels):
    """Shannon entropy H = -sum(p_i * log2(p_i)) over label occurrences."""
    label_list = list(labels)
    total = len(label_list)
    if total == 0:
        return 0.0
    entropy = 0.0
    for outcome in set(label_list):
        prob = label_list.count(outcome) / total
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return float(entropy)


# ---------------------------------------------------------------------------
# A2. Gini index
# ---------------------------------------------------------------------------
def calculate_gini(labels):
    """Gini impurity Gini = 1 - sum(p_j^2) over label occurrences."""
    label_list = list(labels)
    total = len(label_list)
    if total == 0:
        return 0.0
    return float(1.0 - sum((label_list.count(o) / total) ** 2
                           for o in set(label_list)))


# ---------------------------------------------------------------------------
# A4. Binning with type + bins parameters (defaults = overloading substitute)
# ---------------------------------------------------------------------------
def bin_feature(values, n_bins: int = 4, binning_type: str = "equal_width"):
    """Discretize continuous values; binning_type in {equal_width, equal_frequency}."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return []
    if binning_type == "equal_frequency":
        order = np.argsort(arr, kind="stable")
        bin_ids = np.zeros(len(arr), dtype=int)
        for rank, idx in enumerate(order):
            bin_ids[idx] = min(n_bins - 1, rank * n_bins // len(arr))
        return bin_ids.tolist()
    return equal_width_binning(arr, n_bins)


# ---------------------------------------------------------------------------
# A3. Root-node selection via information gain
# ---------------------------------------------------------------------------
def information_gain(parent_labels, feature_bins):
    """Parent entropy minus weighted child entropy after splitting on bins."""
    parent_entropy = calculate_entropy(parent_labels)
    total = len(parent_labels)
    weighted = 0.0
    for bin_value in set(feature_bins):
        child = [lab for lab, b in zip(parent_labels, feature_bins)
                 if b == bin_value]
        weighted += len(child) / total * calculate_entropy(child)
    return float(parent_entropy - weighted)


def find_root_node(feature_matrix, feature_names, labels,
                   n_bins: int = 4, binning_type: str = "equal_width"):
    """Score every feature by information gain; return best + full ranking."""
    gains = {}
    columns = np.asarray(feature_matrix).T
    for name, column in zip(feature_names, columns):
        bins = bin_feature(column, n_bins, binning_type)
        gains[name] = information_gain(list(labels), bins)
    best_feature = max(gains, key=gains.get)
    return best_feature, gains


# ---------------------------------------------------------------------------
# A5. Own decision-tree module (multi-way splits on binned features)
# ---------------------------------------------------------------------------
def _majority_label(labels):
    """Most frequent label (ties -> first seen)."""
    return max(set(labels), key=list(labels).count)


def build_decision_tree(feature_matrix, labels, feature_names,
                        max_depth: int = 3):
    """Recursively split on max-gain binned feature; leaves hold labels."""
    label_list = list(labels)
    if len(set(label_list)) == 1 or max_depth == 0 or not feature_names:
        return {"label": _majority_label(label_list)}
    columns = np.asarray(feature_matrix).T
    best_gain, best_idx = -1.0, -1
    for idx in range(len(feature_names)):
        gain = information_gain(label_list, list(columns[idx]))
        if gain > best_gain:
            best_gain, best_idx = gain, idx
    if best_gain <= 0:
        return {"label": _majority_label(label_list)}
    node = {"feature": feature_names[best_idx],
            "majority": _majority_label(label_list), "children": {}}
    split_values = columns[best_idx]
    remaining = [n for i, n in enumerate(feature_names) if i != best_idx]
    for bin_value in set(split_values.tolist()):
        mask = [v == bin_value for v in split_values]
        child_matrix = [[row[i] for i in range(len(row)) if i != best_idx]
                        for row, keep in zip(feature_matrix, mask) if keep]
        child_labels = [lab for lab, keep in zip(label_list, mask) if keep]
        node["children"][bin_value] = build_decision_tree(
            child_matrix, child_labels, remaining, max_depth - 1)
    return node


def predict_decision_tree(tree, binned_row, feature_names):
    """Walk the tree; unseen bin values fall back to the node majority."""
    node = tree
    names = list(feature_names)
    row = list(binned_row)
    while "label" not in node:
        feature = node["feature"]
        idx = names.index(feature)
        child = node["children"].get(row[idx], None)
        if child is None:
            return node["majority"]
        node = child
    return node["label"]


def evaluate_accuracy(true_labels, pred_labels):
    """Fraction of correct predictions."""
    if not true_labels:
        return 0.0
    return float(sum(t == p for t, p in zip(true_labels, pred_labels))
                 / len(true_labels))


# ---------------------------------------------------------------------------
# A6/A7/A8. sklearn tree plot, 2-feature boundary, GridSearchCV tuning
# ---------------------------------------------------------------------------
def train_sklearn_tree(features, labels, **kwargs):
    """Fit a sklearn DecisionTreeClassifier with given hyper-parameters."""
    clf = DecisionTreeClassifier(random_state=42, **kwargs)
    clf.fit(np.asarray(features), np.asarray(labels))
    return clf


def save_tree_figure(classifier, feature_names, class_names, out_path: Path):
    """Render the fitted tree with plot_tree and save as PNG."""
    fig, axes = plt.subplots(figsize=(14, 8))
    plot_tree(classifier, feature_names=list(feature_names),
              class_names=list(class_names), filled=True, rounded=True,
              ax=axes, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def save_boundary_figure(features_2d, labels, classifier, out_path: Path):
    """Meshgrid contour of the 2-feature decision boundary + data scatter."""
    points = np.asarray(features_2d, dtype=float)
    targets = np.asarray(labels)
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    pad_x, pad_y = (x_max - x_min) * 0.1 + 1e-6, (y_max - y_min) * 0.1 + 1e-6
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min - pad_x, x_max + pad_x, 200),
        np.linspace(y_min - pad_y, y_max + pad_y, 200))
    grid_pred = classifier.predict(np.c_[grid_x.ravel(), grid_y.ravel()])
    label_to_id = {lab: i for i, lab in enumerate(sorted(set(targets.tolist())))}
    grid_id = np.vectorize(label_to_id.get)(grid_pred).reshape(grid_x.shape)
    fig, axes = plt.subplots(figsize=(7, 5))
    axes.contourf(grid_x, grid_y, grid_id, alpha=0.3)
    for lab in sorted(set(targets.tolist())):
        mask = targets == lab
        axes.scatter(points[mask, 0], points[mask, 1], label=lab, s=40)
    axes.set_xlabel(FEATURE_NAMES[0])
    axes.set_ylabel(FEATURE_NAMES[1])
    axes.set_title("Decision boundary (centroid vs ZCR)")
    axes.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def tune_hyperparameters(features, labels):
    """GridSearchCV over criterion/depth/split; returns best params + score."""
    grid = {"criterion": ["gini", "entropy"],
            "max_depth": [2, 3, 4, 5, None],
            "min_samples_split": [2, 4, 6]}
    search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                          grid, cv=3)
    search.fit(np.asarray(features), np.asarray(labels))
    return dict(search.best_params_), float(search.best_score_)


# ---------------------------------------------------------------------------
# Main program (all prints live here)
# ---------------------------------------------------------------------------
def main():
    work_dir = Path(tempfile.mkdtemp(prefix="lab07-gtzan-"))
    print(f"Downloading project GTZAN subset into {work_dir} ...")
    items = download_project_subset(work_dir, CLIPS_PER_GENRE)
    feature_rows, genre_labels = [], []
    for audio_path, genre in items:
        feature_rows.append(extract_feature_row(audio_path))
        genre_labels.append(genre)
    print(f"Loaded {len(genre_labels)} clips x {len(FEATURE_NAMES)} features.")

    entropy_all = calculate_entropy(genre_labels)
    gini_all = calculate_gini(genre_labels)
    print(f"A1 dataset entropy: {entropy_all:.4f} bits")
    print(f"A2 dataset Gini: {gini_all:.4f}")

    root_feature, gain_ranking = find_root_node(
        feature_rows, FEATURE_NAMES, genre_labels, n_bins=3)
    print(f"A3 root node by information gain: {root_feature}")
    for name, gain in sorted(gain_ranking.items(), key=lambda kv: -kv[1]):
        print(f"   gain({name}) = {gain:.4f}")

    binned = np.array([[int(v) for v in bin_feature(col, 3)]
                       for col in np.asarray(feature_rows).T]).T.tolist()
    train_rows, test_rows, train_labs, test_labs = train_test_split(
        binned, genre_labels, test_size=0.25, random_state=42,
        stratify=genre_labels)
    own_tree = build_decision_tree(train_rows, train_labs, FEATURE_NAMES,
                                   max_depth=3)
    own_preds = [predict_decision_tree(own_tree, r, FEATURE_NAMES)
                 for r in test_rows]
    own_acc = evaluate_accuracy(test_labs, own_preds)
    print(f"A5 own decision tree test accuracy: {own_acc:.3f}")

    scripts_dir = Path(__file__).resolve().parent
    train_full, _, labs_full, _ = train_test_split(
        feature_rows, genre_labels, test_size=0.25, random_state=42,
        stratify=genre_labels)
    sklearn_tree = train_sklearn_tree(train_full, labs_full, max_depth=3)
    class_names = sorted(set(genre_labels))
    tree_fig = save_tree_figure(sklearn_tree, FEATURE_NAMES, class_names,
                                scripts_dir / "lab07_fig_tree.png")
    print(f"A6 tree figure: {tree_fig}")

    two_features = [[r[0], r[1]] for r in feature_rows]
    boundary_tree = train_sklearn_tree(two_features, genre_labels,
                                       max_depth=3)
    boundary_fig = save_boundary_figure(two_features, genre_labels,
                                        boundary_tree,
                                        scripts_dir / "lab07_fig_boundary.png")
    print(f"A7 boundary figure: {boundary_fig}")

    best_params, best_score = tune_hyperparameters(feature_rows,
                                                   genre_labels)
    print(f"A8 best params: {best_params} (cv score {best_score:.3f})")

    metrics = {
        "clips": len(genre_labels), "genres": PROJECT_GENRES,
        "entropy": round(entropy_all, 4), "gini": round(gini_all, 4),
        "root_feature": root_feature,
        "gains": {k: round(v, 4) for k, v in gain_ranking.items()},
        "own_tree_test_accuracy": round(own_acc, 4),
        "best_params": best_params,
        "best_cv_score": round(best_score, 4),
        "tree_fig": Path(tree_fig).name,
        "boundary_fig": Path(boundary_fig).name,
    }
    with open(scripts_dir / "lab07_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    print("Metrics: scripts/lab07_metrics.json")
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
