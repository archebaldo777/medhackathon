#!/usr/bin/env python3
"""Error analysis by clinical subgroup and agreement of the markup checks.

ТЗ section 4 asks for an error analysis, and the organisers evaluate the
reasoning behind each check, not only the pooled F1/ROC-AUC. This script
answers two questions on out-of-fold predictions (same folds and seed as
``scripts/train.py``):

* how each head behaves on the studies the experts singled out in the comment
  column of the label sheet — scoliosis, endoprosthesis, transitional vertebra,
  fracture — where a quality check can be fooled by anatomy rather than by the
  acquisition;
* how well the geometric checks of the automatic markup (iliac crests in the
  field, 3 cm / 2 cm margins around the hip ROI) agree with the expert labels.

Usage: ``.venv/bin/python scripts/error_analysis.py`` (about a minute).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import openpyxl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dxa_train", PROJECT_ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(train)

from src.constants import SPINE  # noqa: E402
from src.dicom_io import anatomical_region, normalize_pixels, read_dicom  # noqa: E402
from src.markup import SCOLIOSIS_DEVIATION_MM, build_markup  # noqa: E402

SUBGROUPS = {
    "scoliosis": ("сколиоз",),
    "endoprosthesis": ("эндопротез",),
    "transitional_vertebra": ("l6", "люмбализ"),
    "fracture": ("перелом",),
}
RELEVANT = {
    "scoliosis": ("spine_axis_deviation", "spine_positioning", "spine_quality"),
    "transitional_vertebra": ("spine_positioning", "spine_quality"),
    "fracture": ("spine_axis_deviation", "foreign_object_or_artifact", "spine_quality"),
    "endoprosthesis": ("foreign_object_or_artifact", "hip_quality", "spine_quality"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--studies", type=Path, default=PROJECT_ROOT / "dataset" / "for education" / "Исследования")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "dataset" / "for education" / "разметка.xlsx")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "error_analysis.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_comments(path: Path) -> dict[str, str]:
    sheet = openpyxl.load_workbook(path, read_only=True, data_only=True).active
    return {
        str(row[1]): str(row[12] or "").lower()
        for row in sheet.iter_rows(min_row=3, values_only=True)
        if row[1]
    }


def subgroup_report(records: list[dict], target_data: dict[str, dict], comments: dict[str, str]) -> dict:
    report: dict[str, dict] = {}
    for name, keywords in SUBGROUPS.items():
        members = [
            index
            for index, record in enumerate(records)
            if any(word in comments.get(record["study"], "") for word in keywords)
        ]
        entry: dict[str, dict] = {"images": len(members), "studies": len({records[i]["study"] for i in members})}
        for target in RELEVANT[name]:
            data = target_data[target]
            scored = [
                (records[i]["labels"].get(target), data["oof_by_record"][i])
                for i in members
                if i in data["oof_by_record"] and records[i]["labels"].get(target) is not None
            ]
            if not scored:
                continue
            labels = np.asarray([label for label, _ in scored])
            predicted = np.asarray([p >= data["oof_threshold"] for _, p in scored])
            entry[target] = {
                "images": len(scored),
                "expert_positive": int(labels.sum()),
                "flagged": int(predicted.sum()),
                "false_positive": int(((labels == 0) & predicted).sum()),
                "false_negative": int(((labels == 1) & ~predicted).sum()),
            }
        report[name] = entry
    return report


def markup_agreement(studies: Path, label_map: dict, comments: dict[str, str]) -> dict:
    counts: dict[str, list] = defaultdict(list)
    for path in sorted(studies.rglob("*.dcm")):
        study = path.relative_to(studies).parts[0]
        if study not in label_map:
            continue
        dicom = read_dicom(path)
        normalized = normalize_pixels(dicom.pixels)
        region = SPINE if dicom.columns >= 290 else anatomical_region(normalized)
        labels = train.labels_for_region(label_map[study], region)
        markup = build_markup(normalized, region, dicom.pixel_spacing_y_mm, dicom.pixel_spacing_x_mm)
        checks = {check["criterion"]: check["ok"] for check in markup.checks}
        if region == SPINE:
            crest = checks.get("В поле видны верхние края подвздошных костей")
            if labels["spine_positioning"] is not None and crest is not None:
                counts["spine_positioning_vs_crest"].append((labels["spine_positioning"], not crest))
            curve = markup.metrics.get("curve_deviation_mm")
            if curve is not None:
                counts["scoliosis_vs_curve"].append(("сколиоз" in comments.get(study, ""), curve >= SCOLIOSIS_DEVIATION_MM))
        else:
            margins = [value for key, value in checks.items() if key.startswith("Поле")]
            if labels["hip_roi_incorrect"] is not None and margins:
                counts["hip_roi_vs_margins"].append((labels["hip_roi_incorrect"], not all(margins)))
    report = {}
    for name, pairs in counts.items():
        truth = np.asarray([bool(a) for a, _ in pairs])
        flag = np.asarray([bool(b) for _, b in pairs])
        report[name] = {
            "images": int(len(pairs)),
            "reference_positive": int(truth.sum()),
            "sensitivity": round(float(flag[truth].mean()), 3) if truth.any() else None,
            "specificity": round(float((~flag[~truth]).mean()), 3) if (~truth).any() else None,
        }
    return report


def main() -> None:
    args = parse_args()
    label_map = train.read_labels(args.labels)
    comments = read_comments(args.labels)
    records = train.collect_records(args.studies, label_map)
    target_data = {}
    for target in train.TARGETS:
        data, _ = train.train_target(target, records, args.folds, args.seed)
        target_data[target] = data
    result = {
        "note": "Out-of-fold predictions, same folds and seed as scripts/train.py; subgroups from the expert comment column.",
        "subgroups": subgroup_report(records, target_data, comments),
        "markup_checks": markup_agreement(args.studies, label_map, comments),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, entry in result["subgroups"].items():
        print(f"{name}: {entry['images']} изображений / {entry['studies']} исследований")
        for target, stats in entry.items():
            if isinstance(stats, dict):
                print(
                    f"  {target}: эксперт+ {stats['expert_positive']}, модель+ {stats['flagged']}, "
                    f"ЛП {stats['false_positive']}, ЛО {stats['false_negative']}"
                )
    for name, stats in result["markup_checks"].items():
        print(f"{name}: чувствительность {stats['sensitivity']}, специфичность {stats['specificity']} (n={stats['images']})")
    print(f"Отчёт: {args.output}")


if __name__ == "__main__":
    main()
