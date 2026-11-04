"""TF-IDF + logistic-regression baseline with a small amount regressor.

The model is deliberately simple: it is the "quick baseline that takes minutes
on a laptop" from the project plan. A transformer can be compared against it
without changing the label schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import FeatureUnion

from .schema import (
    DEFAULT_FULL_MIX_OPS,
    OPERATION_KEYS,
    TARGET_KEYS,
    Label,
    UNITS_BY_OPERATION,
)

DEFAULT_ARTIFACT = Path(__file__).resolve().parents[2] / "models" / "baseline.joblib"


def _features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=2,
                    strip_accents="unicode",
                ),
            ),
        ]
    )


class _OneVsRest:
    """Tiny multi-label logistic regression that tolerates constant columns."""

    def __init__(self) -> None:
        self.models: dict[str, LogisticRegression] = {}
        self.constants: dict[str, int] = {}
        self.classes: list[str] = []

    def fit(self, X, Y: np.ndarray, classes: list[str]) -> "_OneVsRest":
        self.classes = list(classes)
        self.models = {}
        self.constants = {}
        for index, name in enumerate(self.classes):
            column = Y[:, index]
            if column.min() == column.max():
                self.constants[name] = int(column[0])
                continue
            clf = LogisticRegression(
                max_iter=2000,
                C=4.0,
                class_weight="balanced",
                solver="liblinear",
            )
            clf.fit(X, column)
            self.models[name] = clf
        return self

    def predict_proba(self, X) -> np.ndarray:
        out = np.zeros((X.shape[0], len(self.classes)))
        for index, name in enumerate(self.classes):
            if name in self.constants:
                out[:, index] = float(self.constants[name])
            else:
                out[:, index] = self.models[name].predict_proba(X)[:, 1]
        return out


@dataclass
class BaselineModel:
    ops_threshold: float = 0.45
    targets_threshold: float = 0.45
    ops_thresholds: dict[str, float] = field(default_factory=dict)
    target_thresholds: dict[str, float] = field(default_factory=dict)
    feature_union: FeatureUnion | None = None
    op_model: _OneVsRest | None = None
    target_model: _OneVsRest | None = None
    amount_models: dict[str, Ridge] = field(default_factory=dict)
    amount_units: dict[str, str] = field(default_factory=dict)
    amount_presence: LogisticRegression | None = None
    amount_threshold: float = 0.5
    train_size: int = 0

    # -- training ---------------------------------------------------------

    def fit(self, rows: list[Label]) -> "BaselineModel":
        texts = [row.text for row in rows]
        self.feature_union = _features()
        X = self.feature_union.fit_transform(texts)

        op_Y = np.array(
            [[1 if op in row.ops else 0 for op in OPERATION_KEYS] for row in rows]
        )
        target_Y = np.array(
            [[1 if target in row.targets else 0 for target in TARGET_KEYS] for row in rows]
        )
        self.op_model = _OneVsRest().fit(X, op_Y, list(OPERATION_KEYS))
        self.target_model = _OneVsRest().fit(X, target_Y, list(TARGET_KEYS))

        # Only emit an amount when the prompt actually states one. Without this
        # gate the ridge head happily invents "+4.9 dB" for "a bit louder".
        presence_y = np.array([1 if row.amount is not None else 0 for row in rows])
        if presence_y.min() != presence_y.max():
            self.amount_presence = LogisticRegression(
                max_iter=2000, C=4.0, class_weight="balanced", solver="liblinear"
            ).fit(X, presence_y)
        else:
            self.amount_presence = None

        self.amount_models = {}
        self.amount_units = {}
        indexed_rows = list(enumerate(rows))
        for op, units in UNITS_BY_OPERATION.items():
            if not units:
                continue
            for unit in units:
                unit_rows = [
                    (index, row)
                    for index, row in indexed_rows
                    if len(row.ops) == 1
                    and row.ops[0] == op
                    and row.amount is not None
                    and row.unit == unit
                ]
                if len(unit_rows) < 6:
                    continue
                model = Ridge(alpha=1.0)
                model.fit(
                    X[[index for index, _ in unit_rows]],
                    [row.amount for _, row in unit_rows],
                )
                self.amount_models[f"{op}|{unit}"] = model
                self.amount_units[op] = unit

        self.train_size = len(rows)
        return self

    def tune_thresholds(self, rows: list[Label], grid: tuple[float, ...] = (
        0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
    )) -> None:
        """Pick a per-class threshold on validation data.

        A single threshold is a poor fit for 12 operations with different
        base rates; this is cheap and uses only the validation split.
        """
        from sklearn.metrics import f1_score

        assert self.feature_union is not None
        assert self.op_model is not None
        assert self.target_model is not None
        X = self.feature_union.transform([row.text for row in rows])
        op_probs = self.op_model.predict_proba(X)
        target_probs = self.target_model.predict_proba(X)

        def tune(probs: np.ndarray, gold: np.ndarray, classes: list[str]) -> dict[str, float]:
            thresholds: dict[str, float] = {}
            for index, name in enumerate(classes):
                truth = gold[:, index]
                if truth.min() == truth.max():
                    continue
                best_f1 = -1.0
                best = 0.5
                for threshold in grid:
                    pred = (probs[:, index] >= threshold).astype(int)
                    score = f1_score(truth, pred, zero_division=0)
                    if score > best_f1:
                        best_f1 = score
                        best = threshold
                thresholds[name] = best
            return thresholds

        op_gold = np.array(
            [[1 if op in row.ops else 0 for op in OPERATION_KEYS] for row in rows]
        )
        target_gold = np.array(
            [[1 if target in row.targets else 0 for target in TARGET_KEYS] for row in rows]
        )
        self.ops_thresholds = tune(op_probs, op_gold, list(OPERATION_KEYS))
        self.target_thresholds = tune(target_probs, target_gold, list(TARGET_KEYS))

        if self.amount_presence is not None:
            truth = np.array([1 if row.amount is not None else 0 for row in rows])
            probs = self.amount_presence.predict_proba(X)[:, 1]
            best_f1, best = -1.0, 0.5
            for threshold in grid:
                score = f1_score(truth, (probs >= threshold).astype(int), zero_division=0)
                if score > best_f1:
                    best_f1, best = score, threshold
            self.amount_threshold = best

    # -- inference --------------------------------------------------------

    def _scores(self, text: str) -> tuple[dict[str, float], dict[str, float], np.ndarray]:
        assert self.feature_union is not None
        assert self.op_model is not None
        assert self.target_model is not None
        X = self.feature_union.transform([text])
        op_scores = self.op_model.predict_proba(X)[0]
        target_scores = self.target_model.predict_proba(X)[0]
        return (
            {op: float(score) for op, score in zip(OPERATION_KEYS, op_scores)},
            {target: float(score) for target, score in zip(TARGET_KEYS, target_scores)},
            X,
        )

    def predict(self, text: str) -> dict:
        op_scores, target_scores, X = self._scores(text)
        lowered = text.lower()
        speaker_cue = re.search(
            r"\b(speaker|host|interviewer|guest|voice|person)\b", lowered
        )
        invalid_ops = {"remove_speaker"} if not speaker_cue else set()

        ops = [
            op
            for op in OPERATION_KEYS
            if op_scores[op] >= self.ops_thresholds.get(op, self.ops_threshold)
            and op not in invalid_ops
        ]

        # Multi-label only when the sentence actually joins two instructions.
        if len(ops) > 1 and not re.search(r"\b(and|then|also|plus)\b|[,&;]", lowered):
            ops = [max(ops, key=lambda name: op_scores[name])]

        if not ops:
            for name in sorted(op_scores, key=op_scores.get, reverse=True):
                if name in invalid_ops:
                    continue
                if op_scores[name] > 0.15:
                    ops = [name]
                    break

        targets = [
            target
            for target in TARGET_KEYS
            if target_scores[target] >= self.target_thresholds.get(target, self.targets_threshold)
        ]

        # A speaker target only makes sense for a speaker operation.
        if "remove_speaker" not in ops:
            targets = [t for t in targets if t not in ("speaker_1", "speaker_2")]

        # A concrete stem beats a spurious full_mix prediction.
        concrete = [t for t in targets if t != "full_mix"]
        if concrete and "full_mix" in targets:
            targets.remove("full_mix")

        # One speaker per removal; keep the stronger prediction.
        if "remove_speaker" in ops and "speaker_1" in targets and "speaker_2" in targets:
            weaker = min(("speaker_1", "speaker_2"), key=lambda name: target_scores[name])
            targets.remove(weaker)

        if not targets and ops:
            if any(op in DEFAULT_FULL_MIX_OPS for op in ops):
                targets = ["full_mix"]
            else:
                targets = [max(target_scores, key=target_scores.get)]

        if ops and "remove_speaker" not in ops and set(targets).issubset(
            {"speaker_1", "speaker_2"}
        ):
            targets = ["full_mix"]

        amounts: dict[str, dict] = {}
        has_amount = True
        if self.amount_presence is not None:
            has_amount = bool(
                self.amount_presence.predict_proba(X)[0, 1] >= self.amount_threshold
            )
        if has_amount:
            for op in ops:
                unit = self.amount_units.get(op)
                model = self.amount_models.get(f"{op}|{unit}") if unit else None
                if model is not None:
                    value = float(model.predict(X)[0])
                    if unit == "factor":
                        value = max(0.25, min(4.0, value))
                    if unit == "dB":
                        value = max(-24.0, min(24.0, value))
                    if unit == "bpm":
                        value = max(30.0, min(300.0, value))
                    amounts[op] = {"amount": round(value, 2), "unit": unit}

        primary = next((op for op in ops if op in amounts), None)
        return {
            "ops": ops,
            "targets": targets,
            "amount": amounts[primary]["amount"] if primary else None,
            "unit": amounts[primary]["unit"] if primary else None,
            "amounts": amounts,
            "op_scores": op_scores,
            "target_scores": target_scores,
            "span": None,
        }

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path = DEFAULT_ARTIFACT) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path = DEFAULT_ARTIFACT) -> "BaselineModel":
        model = joblib.load(path)
        if not isinstance(model, BaselineModel):
            raise TypeError(f"{path} does not contain a BaselineModel")
        return model


def top_k(scores: dict[str, float], k: int = 3) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
