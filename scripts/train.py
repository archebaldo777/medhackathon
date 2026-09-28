#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import openpyxl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RIGHT_HIP, SPINE  # noqa: E402
from src.dicom_io import anatomical_region, normalize_pixels, read_dicom  # noqa: E402
from src.features import TARGET_BLOCKS, extract_features, feature_sizes  # noqa: E402
from src.model import DECISION_POINT, align_to_threshold  # noqa: E402


TARGETS = (
    "spine_quality",
    "spine_positioning",
    "spine_axis_deviation",
    "foreign_object_or_artifact",
    "hip_quality",
    "hip_positioning_or_rotation",
    "hip_roi_incorrect",
)

# Chosen by a sweep over estimator, regularisation and pooling on the compact
# per-target descriptors, then confirmed on five independent fold splits (seeds
# 1, 7, 13, 42, 2024) so the choice is not one lucky split.
#
# Caveat worth keeping in view: that sweep ran on scikit-learn 1.8 while this
# project pins 1.7.2, and the choice transferred only in part. Measured on the
# pinned version, v3 -> v4 moved the exported macro-F1 from 0.473 to 0.526 and
# the mean ROC-AUC of the five types from 0.683 to 0.771. Ranking improved on
# six heads of seven; spine_positioning and spine_axis_deviation lost F1 while
# gaining AUC, which is threshold placement on a rare class rather than the
# model. Re-running the sweep in the pinned environment is the obvious next
# step, and the selection stays exploratory until the closed test set.
MODEL_BY_TARGET = {
    "spine_quality": {
        "type": "extra_trees",
        "min_samples_leaf": 2,
        "max_features": 0.25,
        "pooling": "max",
    },
    "spine_positioning": {
        "type": "extra_trees",
        "min_samples_leaf": 1,
        "max_features": 0.25,
        "pooling": "mean",
    },
    "spine_axis_deviation": {
        "type": "extra_trees",
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "pooling": "mean",
    },
    "foreign_object_or_artifact": {"type": "linear", "c": 0.1, "pooling": "mean"},
    "hip_quality": {
        "type": "extra_trees",
        "min_samples_leaf": 4,
        "max_features": 0.25,
        "pooling": "max",
    },
    "hip_positioning_or_rotation": {"type": "linear", "c": 0.1, "pooling": "max"},
    "hip_roi_incorrect": {"type": "linear", "c": 0.01, "pooling": "max"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение воспроизводимой модели Эвектио")
    parser.add_argument(
        "--studies",
        type=Path,
        help="папка исследований: первый уровень — идентификатор study из таблицы разметки",
        default=PROJECT_ROOT / "dataset" / "for education" / "Исследования",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="таблица экспертной разметки .xlsx в формате организатора",
        default=PROJECT_ROOT / "dataset" / "for education" / "разметка.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="куда сохранить модель",
        default=PROJECT_ROOT / "models" / "dxa_quality_v4.joblib",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        help="куда сохранить отчёт валидации (JSON)",
        default=PROJECT_ROOT / "reports" / "validation_metrics_v4.json",
    )
    parser.add_argument("--folds", type=int, default=5, help="число фолдов кросс-валидации")
    parser.add_argument("--seed", type=int, default=42, help="seed генератора")
    return parser.parse_args()


def value_or_none(value: object) -> int | None:
    return int(value) if value in (0, 1) else None


def read_labels(path: Path) -> dict[str, dict[str, int | None]]:
    sheet = openpyxl.load_workbook(path, read_only=True, data_only=True).active
    result: dict[str, dict[str, int | None]] = {}
    for row in sheet.iter_rows(min_row=3, values_only=True):
        if not row[1]:
            continue
        result[str(row[1])] = {
            "spine_positioning": value_or_none(row[2]),
            "spine_axis_deviation": value_or_none(row[3]),
            "foreign_object_or_artifact": value_or_none(row[4]),
            "right_hip_positioning": value_or_none(row[5]),
            "right_hip_roi": value_or_none(row[6]),
            "left_hip_positioning": value_or_none(row[7]),
            "left_hip_roi": value_or_none(row[8]),
            "spine_quality": value_or_none(row[9]),
            "right_hip_quality": value_or_none(row[10]),
            "left_hip_quality": value_or_none(row[11]),
        }
    return result


def labels_for_region(labels: dict[str, int | None], region: str) -> dict[str, int | None]:
    if region == SPINE:
        return {
            "spine_quality": labels["spine_quality"],
            "spine_positioning": labels["spine_positioning"],
            "spine_axis_deviation": labels["spine_axis_deviation"],
            "foreign_object_or_artifact": labels["foreign_object_or_artifact"],
        }
    side = "right" if region == RIGHT_HIP else "left"
    return {
        "hip_quality": labels[f"{side}_hip_quality"],
        "hip_positioning_or_rotation": labels[f"{side}_hip_positioning"],
        "hip_roi_incorrect": labels[f"{side}_hip_roi"],
    }


def collect_records(studies: Path, label_map: dict[str, dict[str, int | None]]) -> list[dict]:
    records: list[dict] = []
    files = sorted(studies.rglob("*.dcm"))
    for index, path in enumerate(files, start=1):
        study = path.relative_to(studies).parts[0]
        if study not in label_map:
            continue
        dicom = read_dicom(path)
        normalized = normalize_pixels(dicom.pixels)
        # Scanner-native width is a reliable acquisition-field marker in the
        # supplied corpus; morphology determines left/right hip orientation.
        region = SPINE if dicom.columns >= 290 else anatomical_region(normalized)
        records.append(
            {
                # Every head reads its own descriptor, so features are per target.
                "features": {
                    target: extract_features(
                        normalized,
                        region,
                        target,
                        dicom.pixel_spacing_y_mm,
                        dicom.pixel_spacing_x_mm,
                    )
                    for target in TARGET_BLOCKS
                },
                "study": study,
                "region": region,
                "labels": labels_for_region(label_map[study], region),
            }
        )
        if index % 100 == 0:
            print(f"Прочитано {index}/{len(files)} DICOM")
    return records


def best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.0, 1.0, 101)
    ranked: list[tuple[float, float, float]] = []
    for value in candidates:
        predicted = probabilities >= value
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        specificity = tn / max(tn + fp, 1)
        sensitivity = tp / max(tp + fn, 1)
        if specificity >= 0.60:
            ranked.append((f1_score(y_true, predicted, zero_division=0), sensitivity, -value))
    if not ranked:
        return 0.5
    # Within a minimum-specificity operating range, maximise F1 and then
    # sensitivity. This avoids an unusable "flag everything" triage threshold.
    _, _, negative_threshold = max(ranked)
    return float(-negative_threshold)


# The refit estimator scores its own training set, so a memorising model pushes
# in-sample probabilities to 0 and 1. Rate matching on that distribution can
# return an edge value that no longer behaves like the validated operating point,
# so edge candidates are rejected and a rate this far from the target counts as a
# failed transfer.
RATE_MATCH_TOLERANCE = 0.15


def rate_matched_threshold(
    probabilities: np.ndarray, target_rate: float, fallback: float
) -> tuple[float, str]:
    """Choose a deployable threshold whose observed rate is closest to OOF.

    Unlike a raw quantile this handles tied probabilities without accidentally
    selecting zero and flagging every image. Candidates are restricted to the
    interior of the probability range: a transferred threshold of exactly 1.0
    fires only when every tree agrees, one of 0.0 flags everything, and neither
    reproduces the rate the head was validated at. When no interior candidate
    lands near the target rate the out-of-fold threshold is kept, because that is
    the value the reported metrics were measured with.
    """
    if target_rate <= 0.0:
        return float(np.nextafter(float(probabilities.max()), np.inf)), "no_positives"
    candidates = np.unique(probabilities)
    interior = candidates[(candidates > 0.0) & (candidates < 1.0)]
    if interior.size == 0:
        return float(fallback), "out_of_fold"
    rates = np.asarray([(probabilities >= value).mean() for value in interior])
    differences = np.abs(rates - target_rate)
    if float(differences.min()) > RATE_MATCH_TOLERANCE:
        return float(fallback), "out_of_fold"
    best = np.where(differences == differences.min())[0]
    return float(interior[best[-1]]), "rate_matched"


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    groups: np.ndarray,
    seed: int,
) -> dict[str, list[float] | None]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    values: dict[str, list[float]] = {
        "f1": [],
        "balanced_accuracy": [],
        "roc_auc": [],
        "pr_auc": [],
    }
    for _ in range(1000):
        sampled_groups = rng.choice(unique_groups, len(unique_groups), replace=True)
        sample = np.concatenate([np.where(groups == group)[0] for group in sampled_groups])
        labels = y_true[sample]
        scores = probabilities[sample]
        predicted = scores >= threshold
        values["f1"].append(float(f1_score(labels, predicted, zero_division=0)))
        values["balanced_accuracy"].append(float(balanced_accuracy_score(labels, predicted)))
        if len(np.unique(labels)) == 2:
            values["roc_auc"].append(float(roc_auc_score(labels, scores)))
            values["pr_auc"].append(float(average_precision_score(labels, scores)))
    return {
        key: (
            [round(float(np.percentile(items, 2.5)), 4), round(float(np.percentile(items, 97.5)), 4)]
            if items
            else None
        )
        for key, items in values.items()
    }


def sample_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(groups.tolist())
    return np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)


def make_estimator(target: str, seed: int):
    config = MODEL_BY_TARGET[target]
    if config["type"] == "linear":
        return LogisticRegression(
            C=config["c"],
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
            solver="liblinear",
        )
    if config["type"] == "rbf_svc":
        return SVC(
            C=config["c"],
            kernel="rbf",
            gamma=config.get("gamma", "scale"),
            class_weight="balanced",
            probability=True,
            random_state=seed,
        )
    return ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=config["min_samples_leaf"],
        class_weight="balanced",
        max_features=config.get("max_features", "sqrt"),
        n_jobs=-1,
        random_state=seed,
    )


def needs_scaling(target: str) -> bool:
    return MODEL_BY_TARGET[target]["type"] in {"linear", "rbf_svc"}


def training_samples(target: str, records: list[dict]) -> dict:
    """Build independent-image or study/side-level samples for one target."""
    usable = [
        (index, record)
        for index, record in enumerate(records)
        if record["labels"].get(target) is not None
    ]
    pooling = MODEL_BY_TARGET[target].get("pooling", "none")
    if pooling == "none":
        samples = [
            {
                "features": record["features"][target],
                "label": record["labels"][target],
                "study": record["study"],
                "record_indices": [index],
            }
            for index, record in usable
        ]
    else:
        buckets: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
        for index, record in usable:
            buckets[(record["study"], record["region"])].append((index, record))
        samples = []
        for (study, _), items in sorted(buckets.items()):
            features = np.stack([record["features"][target] for _, record in items])
            labels = {record["labels"][target] for _, record in items}
            if len(labels) != 1:
                raise ValueError(f"Противоречивые метки серии для {target}: {study}")
            samples.append(
                {
                    "features": getattr(features, pooling)(axis=0),
                    "label": labels.pop(),
                    "study": study,
                    "record_indices": [index for index, _ in items],
                }
            )
    return {
        "x": np.stack([sample["features"] for sample in samples]),
        "y": np.asarray([sample["label"] for sample in samples], dtype=np.int64),
        "groups": np.asarray([sample["study"] for sample in samples]),
        "record_indices": [sample["record_indices"] for sample in samples],
        "image_counts": np.asarray(
            [len(sample["record_indices"]) for sample in samples], dtype=np.int64
        ),
        "pooling": pooling,
    }


def fit_estimator(target: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int):
    scaler = StandardScaler().fit(x) if needs_scaling(target) else None
    transformed = scaler.transform(x) if scaler is not None else x
    estimator = make_estimator(target, seed)
    weights = None if MODEL_BY_TARGET[target]["type"] == "rbf_svc" else sample_weights(groups)
    estimator.fit(transformed, y, sample_weight=weights)
    return scaler, estimator


def train_target(
    target: str,
    records: list[dict],
    folds: int,
    seed: int,
) -> tuple[dict, dict]:
    data = training_samples(target, records)
    x = data["x"]
    y = data["y"]
    groups = data["groups"]
    unique_groups = np.unique(groups)
    n_splits = min(folds, len(unique_groups))
    out_of_fold = np.zeros(len(y), dtype=np.float64)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(x, y, groups):
        scaler, classifier = fit_estimator(
            target, x[train_idx], y[train_idx], groups[train_idx], seed
        )
        validation = scaler.transform(x[valid_idx]) if scaler is not None else x[valid_idx]
        out_of_fold[valid_idx] = classifier.predict_proba(validation)[:, 1]

    image_counts = data["image_counts"]
    image_y = np.repeat(y, image_counts)
    image_groups = np.repeat(groups, image_counts)
    image_probabilities = np.repeat(out_of_fold, image_counts)
    threshold = best_threshold(image_y, image_probabilities)
    predicted = (image_probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(image_y, predicted, labels=[0, 1]).ravel()
    confidence_intervals = cluster_bootstrap_ci(
        image_y, image_probabilities, threshold, image_groups, seed
    )
    oof_by_record = {
        record_index: float(probability)
        for probability, record_indices in zip(out_of_fold, data["record_indices"])
        for record_index in record_indices
    }
    metrics = {
        "images": int(len(image_y)),
        "series_samples": int(len(y)),
        "studies": int(len(unique_groups)),
        "positive_images": int(image_y.sum()),
        "pooling": data["pooling"],
        "oof_threshold": round(threshold, 4),
        "oof_positive_rate": round(float(predicted.mean()), 4),
        "f1": round(float(f1_score(image_y, predicted, zero_division=0)), 4),
        "f1_95_ci": confidence_intervals["f1"],
        "balanced_accuracy": round(
            float(balanced_accuracy_score(image_y, predicted)), 4
        ),
        "balanced_accuracy_95_ci": confidence_intervals["balanced_accuracy"],
        "sensitivity": round(float(tp / max(tp + fn, 1)), 4),
        "specificity": round(float(tn / max(tn + fp, 1)), 4),
        "roc_auc": (
            round(float(roc_auc_score(image_y, image_probabilities)), 4)
            if len(np.unique(image_y)) == 2
            else None
        ),
        "roc_auc_95_ci": confidence_intervals["roc_auc"],
        "pr_auc": round(
            float(average_precision_score(image_y, image_probabilities)), 4
        ),
        "pr_auc_95_ci": confidence_intervals["pr_auc"],
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "model": MODEL_BY_TARGET[target],
    }
    data.update({
        "oof_threshold": threshold,
        "oof_positive_rate": float(predicted.mean()),
        "oof_by_record": oof_by_record,
    })
    return data, metrics


def submission_policy_metrics(records: list[dict], target_data: dict[str, dict]) -> dict:
    """Evaluate exactly the promotion and closed-list policy used at inference."""
    violation_targets = (
        "spine_positioning",
        "spine_axis_deviation",
        "foreign_object_or_artifact",
        "hip_positioning_or_rotation",
        "hip_roi_incorrect",
    )
    quality_values: dict[str, list[tuple[int, float, int, int, float]]] = {
        "spine_quality": [],
        "hip_quality": [],
    }
    violation_values: dict[str, list[tuple[int, int]]] = {
        target: [] for target in violation_targets
    }
    ungated_values: dict[str, list[tuple[int, int]]] = {
        target: [] for target in violation_targets
    }
    for index, record in enumerate(records):
        if record["region"] == SPINE:
            quality_target = "spine_quality"
            type_targets = violation_targets[:3]
        else:
            quality_target = "hip_quality"
            type_targets = violation_targets[3:]
        quality_probability = target_data[quality_target]["oof_by_record"].get(index)
        if quality_probability is None:
            continue
        quality_label = record["labels"].get(quality_target)
        probabilities = {
            target: target_data[target]["oof_by_record"].get(index)
            for target in type_targets
        }
        if any(value is None for value in probabilities.values()):
            continue
        detected = {
            target
            for target, probability in probabilities.items()
            if probability >= target_data[target]["oof_threshold"]
        }
        aggregate_quality = int(
            quality_probability >= target_data[quality_target]["oof_threshold"]
        )
        exported_quality = int(bool(aggregate_quality) or bool(detected))
        # The exported quality_prob: every head aligned to its own threshold,
        # strongest wins, so it crosses 0.5 exactly when exported_quality is 1.
        exported_probability = max(
            align_to_threshold(value, target_data[target]["oof_threshold"])
            for target, value in ((quality_target, quality_probability), *probabilities.items())
        )
        if quality_label is not None:
            quality_values[quality_target].append(
                (
                    quality_label,
                    quality_probability,
                    aggregate_quality,
                    exported_quality,
                    exported_probability,
                )
            )
        for target in type_targets:
            label = record["labels"].get(target)
            if label is None:
                continue
            raw_prediction = int(
                probabilities[target] >= target_data[target]["oof_threshold"]
            )
            ungated_values[target].append((label, raw_prediction))
            violation_values[target].append((label, int(target in detected)))

    quality_report = {}
    for target, values in quality_values.items():
        labels = np.asarray([value[0] for value in values])
        probabilities = np.asarray([value[1] for value in values])
        aggregate = np.asarray([value[2] for value in values])
        exported = np.asarray([value[3] for value in values])
        exported_probabilities = np.asarray([value[4] for value in values])
        quality_report[target] = {
            "aggregate_f1": round(float(f1_score(labels, aggregate, zero_division=0)), 4),
            "exported_f1": round(float(f1_score(labels, exported, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(labels, probabilities)), 4),
            "exported_prob_roc_auc": round(
                float(roc_auc_score(labels, exported_probabilities)), 4
            ),
            "class_prob_disagreements": int(
                np.sum((exported_probabilities >= DECISION_POINT) != exported.astype(bool))
            ),
        }

    violation_report = {}
    raw_f1_values = []
    exported_f1_values = []
    for target in violation_targets:
        labels = np.asarray([value[0] for value in violation_values[target]])
        exported = np.asarray([value[1] for value in violation_values[target]])
        raw = np.asarray([value[1] for value in ungated_values[target]])
        raw_f1 = float(f1_score(labels, raw, zero_division=0))
        exported_f1 = float(f1_score(labels, exported, zero_division=0))
        raw_f1_values.append(raw_f1)
        exported_f1_values.append(exported_f1)
        violation_report[target] = {
            "positive_images": int(labels.sum()),
            "ungated_f1": round(raw_f1, 4),
            "exported_f1": round(exported_f1, 4),
        }
    return {
        "policy": (
            "independent thresholded types; any detected type promotes quality_class; "
            "quality_prob = max of threshold-aligned head scores"
        ),
        "quality": quality_report,
        "violations": violation_report,
        "ungated_violation_macro_f1": round(float(np.mean(raw_f1_values)), 4),
        "exported_violation_macro_f1": round(float(np.mean(exported_f1_values)), 4),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    started = time.perf_counter()
    label_map = read_labels(args.labels)
    records = collect_records(args.studies, label_map)
    if not records:
        raise SystemExit("Не найдено ни одной размеченной DICOM-записи")

    classifiers: dict[str, dict] = {}
    metrics: dict[str, dict] = {}
    target_data: dict[str, dict] = {}
    for target in TARGETS:
        data, target_metrics = train_target(target, records, args.folds, args.seed)
        scaler, classifier = fit_estimator(
            target, data["x"], data["y"], data["groups"], args.seed
        )
        transformed = scaler.transform(data["x"]) if scaler is not None else data["x"]
        final_probabilities = classifier.predict_proba(transformed)[:, 1]
        target_rate = data["oof_positive_rate"]
        image_probabilities = np.repeat(final_probabilities, data["image_counts"])
        deployment_threshold, threshold_source = rate_matched_threshold(
            image_probabilities, target_rate, data["oof_threshold"]
        )
        item = {
            "estimator_type": MODEL_BY_TARGET[target]["type"],
            "threshold": deployment_threshold,
            "oof_threshold": float(data["oof_threshold"]),
            "pooling": data["pooling"],
            "feature_blocks": list(TARGET_BLOCKS[target]),
            "feature_size": int(data["x"].shape[1]),
        }
        if MODEL_BY_TARGET[target]["type"] == "linear":
            item.update(
                {
                    "coef": classifier.coef_[0].astype(np.float32),
                    "intercept": float(classifier.intercept_[0]),
                    "feature_mean": scaler.mean_.astype(np.float32),
                    "feature_scale": scaler.scale_.astype(np.float32),
                }
            )
        else:
            item["estimator"] = classifier
            if scaler is not None:
                item["feature_mean"] = scaler.mean_.astype(np.float32)
                item["feature_scale"] = scaler.scale_.astype(np.float32)
        classifiers[target] = item
        target_data[target] = data
        target_metrics["deployment_threshold"] = round(deployment_threshold, 4)
        target_metrics["deployment_threshold_source"] = threshold_source
        target_metrics["deployment_positive_rate"] = round(
            float((image_probabilities >= deployment_threshold).mean()), 4
        )
        metrics[target] = target_metrics
        print(f"{target}: F1={target_metrics['f1']:.3f}, AUC={target_metrics['roc_auc']}")

    bundle = {
        "format_version": 4,
        "model_version": "dxa-quality-criteria-4.0",
        "seed": args.seed,
        "feature_sizes": feature_sizes(),
        "classifiers": classifiers,
        "training_summary": {
            "studies": len({record["study"] for record in records}),
            "images": len(records),
            "label_source": args.labels.name,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output, compress=3)

    report = {
        "evaluation": "5-fold StratifiedGroupKFold by study; duplicate images never cross folds",
        "seed": args.seed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "dataset": bundle["training_summary"],
        "metrics": metrics,
        "submission_policy": submission_policy_metrics(records, target_data),
        "notes": [
            "Labels are weak study-level labels propagated to images of the matching anatomical region.",
            "95% confidence intervals use cluster bootstrap resampling at study level.",
            "Target-specific estimators were fixed after an exploratory ablation; metrics remain internal and require external validation.",
            "The supplied data originate from one scanner model; external validation is required.",
        ],
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Модель: {args.output}")
    print(f"Метрики: {args.metrics}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
