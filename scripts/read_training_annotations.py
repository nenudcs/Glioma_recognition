#!/usr/bin/env python3
"""Build a sequence-level index from the training annotation workbook.

The two inputs are deliberately kept separate:

* SeriesType.xlsx is the base table.  Its key is AccessionNumber + SeriesUid.
* 脑胶质瘤标注结果-训练集.xlsx contains check-, series- and ROI-level sheets.

The annotation workbook has a two-row header in the examples (a grouped title
followed by the real column names), so the reader detects the header row rather
than relying on a fixed row number.  IDs are read as objects and are never
converted to numbers; this matters for long DICOM UIDs and leading zeroes in
AccessionNumber.

This script indexes labels and metadata.  Loading DICOM pixels/masks is kept as
a separate step because the image directory layout is not encoded in either
workbook.  The output key is sufficient for a later image loader.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


EXPECTED = {
    "base": 6349,
    "check": 3252,
    "series": 6145,
    "roi": 9273,
}


def norm_name(value: Any) -> str:
    """Normalize a header for matching while retaining the original header."""

    text = "" if value is None else str(value)
    text = text.replace("\u200b", "").replace("\ufeff", "").strip().lower()
    return re.sub(r"[\s_\-()/\\:：、,，.]+", "", text)


def clean_id(value: Any) -> str:
    """Convert an identifier to text without destroying meaningful zeroes."""

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    # openpyxl returns a numeric Excel cell as int/float.  An integer float is
    # an Excel display artifact (e.g. 123.0), while a string ending in .0 may
    # be a real UID component and must be retained.
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value.is_integer():
            return str(int(value))
    return str(value).replace("\u200b", "").replace("\ufeff", "").strip()


def find_header_row(raw: pd.DataFrame, scan_rows: int = 15) -> int:
    """Find the row containing real headers in a grouped-header worksheet."""

    best_row, best_score = 0, -1
    markers = {
        "accessionnumber",
        "accession",
        "seriesuid",
        "researchseriesuid",
        "studyuid",
        "patientid",
        "roiname",
        "roilabel",
    }
    for row_idx in range(min(scan_rows, len(raw))):
        keys = {norm_name(v) for v in raw.iloc[row_idx].tolist()}
        score = len(keys & markers)
        if "accessionnumber" in keys:
            score += 3
        if score > best_score:
            best_row, best_score = row_idx, score
    # A normal one-row-header workbook has row zero.  The examples have the
    # actual header on row two (zero-based row 1), which this detects.
    return best_row if best_score > 0 else 0


def make_unique_columns(columns: Iterable[Any]) -> list[str]:
    counts: Counter[str] = Counter()
    result: list[str] = []
    for value in columns:
        name = clean_id(value) or "unnamed"
        counts[name] += 1
        result.append(name if counts[name] == 1 else f"{name}__{counts[name]}")
    return result


def read_sheet(path: Path, sheet_name: str) -> tuple[pd.DataFrame, int]:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=object)
    header_row = find_header_row(raw)
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = make_unique_columns(raw.iloc[header_row].tolist())
    # Completely blank rows/columns are formatting remnants from merged cells.
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df.reset_index(drop=True), header_row


def header_map(df: pd.DataFrame) -> dict[str, str]:
    """Map normalized headers to actual column names."""

    result: dict[str, str] = {}
    for col in df.columns:
        result.setdefault(norm_name(col), col)
    return result


def choose_column(
    df: pd.DataFrame,
    aliases: Iterable[str],
    *,
    required: bool = False,
    label: str = "column",
) -> str | None:
    mapping = header_map(df)
    for alias in aliases:
        col = mapping.get(norm_name(alias))
        if col is not None:
            return col
    if required:
        raise ValueError(
            f"{label} not found. Available columns: {', '.join(map(str, df.columns))}"
        )
    return None


ACCESSION_ALIASES = ["AccessionNumber", "Accession", "检查号"]
# Research SeriesUid is the name visible in the sequence and ROI sheets.
# SeriesUid is the name in SeriesType.xlsx.  StudyUid/PatientId are fallbacks
# for the exported annotation variants described by the user.
SERIES_ALIASES = [
    "SeriesUid",
    "Research SeriesUid",
    "ResearchSeriesUid",
    "StudyUid",
    "PatientId",
]


def choose_series_column(
    df: pd.DataFrame,
    reference_series_uids: set[str] | None = None,
    forced_column: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Choose the UID column and report evidence used for the choice.

    Some exports contain both PatientId/StudyUid and Research SeriesUid.  The
    explicit series column normally wins, but overlap with SeriesType.xlsx is
    stronger evidence and catches a mislabeled export.
    """

    mapping = header_map(df)
    candidates: list[tuple[str, int, str]] = []
    for priority, alias in enumerate(SERIES_ALIASES):
        actual = mapping.get(norm_name(alias))
        if actual is not None and actual not in {c[0] for c in candidates}:
            candidates.append((actual, priority, alias))
    if forced_column:
        forced = choose_column(df, [forced_column], required=True, label="forced SeriesUid")
        # Still return evidence for every candidate so QC can show why the
        # override was used and how well it overlaps the base UID set.
        selected_forced = forced
    elif not candidates:
        raise ValueError(
            "series UID column not found. Available columns: "
            + ", ".join(map(str, df.columns))
        )
    reference = reference_series_uids or set()
    evidence: list[dict[str, Any]] = []
    for actual, priority, alias in candidates:
        values = {v for v in df[actual].map(clean_id) if v}
        overlap = len(values & reference)
        evidence.append(
            {
                "column": actual,
                "alias": alias,
                "nonempty": len(values),
                "unique": len(values),
                "overlap_with_base": overlap,
                "overlap_ratio": round(overlap / len(values), 6) if values else 0.0,
                "priority": priority,
            }
        )
    # With a base reference, maximize overlap first.  The explicit SeriesUid /
    # Research SeriesUid names then break ties, followed by uniqueness.
    if forced_column:
        selected = next(x for x in evidence if x["column"] == selected_forced)
    elif reference:
        selected = max(
            evidence,
            key=lambda x: (
                x["overlap_with_base"],
                x["overlap_ratio"],
                -x["priority"],
                x["unique"],
            ),
        )
    else:
        selected = min(evidence, key=lambda x: x["priority"])
    return selected["column"], evidence


def add_keys(
    df: pd.DataFrame,
    kind: str,
    reference_series_uids: set[str] | None = None,
    forced_series_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    accession_col = choose_column(
        out, ACCESSION_ALIASES, required=True, label=f"{kind} AccessionNumber"
    )
    out["_accession"] = out[accession_col].map(clean_id)
    selected: dict[str, Any] = {"accession": accession_col, "series": None}
    if kind in {"base", "series", "roi"}:
        series_col, series_evidence = choose_series_column(
            out, reference_series_uids, forced_series_column
        )
        out["_series_uid"] = out[series_col].map(clean_id)
        out["series_uid_source"] = series_col
        complete = out["_accession"].ne("") & out["_series_uid"].ne("")
        out["_key"] = (out["_accession"] + "\x1f" + out["_series_uid"]).where(complete, "")
        selected["series"] = series_col
        selected["series_candidates"] = series_evidence
    return out, selected


def add_check_key(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    out = df.copy()
    accession_col = choose_column(
        out, ACCESSION_ALIASES, required=True, label="check AccessionNumber"
    )
    out["_accession"] = out[accession_col].map(clean_id)
    return out, {"accession": accession_col, "series": None}


def nonempty(series: pd.Series) -> pd.Series:
    return series.fillna("").map(clean_id).ne("")


def prefix_payload(
    df: pd.DataFrame,
    prefix: str,
    keep_keys: bool = True,
    preserve: Iterable[str] = (),
) -> pd.DataFrame:
    """Prefix source fields so check/series/ROI values cannot collide."""

    keep = {"_accession", "_series_uid", "_key", *preserve}
    rename = {
        col: (col if col in keep and keep_keys else f"{prefix}__{col}")
        for col in df.columns
        if col not in {"_accession", "_series_uid", "_key"}
    }
    return df.rename(columns=rename)


def field_value(df: pd.DataFrame, aliases: Iterable[str]) -> pd.Series | None:
    col = choose_column(df, aliases)
    return None if col is None else df[col]


SEMANTIC_FIELDS: dict[str, list[str]] = {
    "pathology_result": ["STUDY->CLINICAL->病理结果", "病理结果", "PathologyResult"],
    "clinical_note": ["STUDY->CLINICAL->备注", "备注", "ClinicalNote"],
    "glioma_with_label": ["glioma_with_label", "脑胶质瘤存在标签", "GliomaWithLabel"],
    "lesion_morphology": ["lesion_morphology", "病灶形态"],
    "location_of_lesion": ["location_of_lesion", "病灶位置"],
    "lesion_mor_feature_lobulation": ["lesion_mor_feature_lobulation", "边缘分叶"],
    "lesion_morph_feature_boundary": ["lesion_morph_feature_boundary", "边界"],
    "tumor_feature_necrosis": ["tumor_feature_necrosis", "坏死"],
    "tumor_feature_change": ["tumor_feature_change", "囊变"],
    "tumor_feature_hemorrhage": ["tumor_feature_hemorrhage", "出血"],
    "tumor_feature_calcification": ["tumor_feature_calcification", "钙化"],
    "tumor_t2wi_signal_intensity": ["tumor_t2wi_signal_intensity", "T2WI信号"],
    "tumor_t2_flair_sign_intensity": ["tumor_t2_flair_sign_intensity", "T2-FLAIR信号", "T2FLAIR信号"],
    "tumor_t1wi_c_enhan": ["tumor_t1wi_c_enhan", "T1WI+C强", "T1WI+C强化"],
    "tumor_t1wi_c_enhan_pattern": ["tumor_t1wi_c_enhan_pattern", "T1WI+C强形态", "T1WI+C强化形态"],
    "who_grade": ["WHO分级", "WHOGrade", "who_grade"],
    "detail_description": ["DetailDescription", "DetailDes", "序列详细描述"],
    "series_number": ["SeriesNumber", "SeriesNur", "序列号"],
    "series_date": ["SeriesDate", "序列日期"],
    "series_time": ["SeriesTime", "序列时间"],
    "modality": ["Modality", "模态"],
    "body_part": ["BodyPart", "检查部位"],
    "patient_position": ["PatientPosition", "PatientPc", "患者体位"],
    "spacing": ["Spacing", "层间距"],
    "image_size": ["Size", "图像尺寸"],
    "series_description": ["SeriesDescription", "序列描述"],
    "roi_name": ["RoiName", "ROIName", "ROI名称"],
    "roi_label": ["RoiLabel", "ROILabel"],
    "roi_volume": ["RoiVolume", "ROIVolume"],
    "roi_voxel_count": ["RoiVoxelCount", "ROIVoxelCount"],
    "roi_hu_max_value": ["RoiHUMaxValue"],
    "roi_hu_min_value": ["RoiHUMinValue"],
    "roi_hu_average_value": ["RoiHUAverageValue"],
    "roi_hu_standard_deviation": ["RoiHUStandardDeviation"],
    "cross_sectional_area_max_value": ["CrossSectionalAreaMaxValue"],
}


def add_semantic_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add stable English names while retaining every raw source column."""

    out = df.copy()
    for semantic, aliases in SEMANTIC_FIELDS.items():
        values = field_value(out, aliases)
        if values is not None:
            out[semantic] = values.map(clean_id)
    return out


BOOL_TRUE = {"true", "1", "yes", "y", "有", "是", "存在"}
BOOL_FALSE = {"false", "0", "no", "n", "无", "否", "不存在"}
VALUE_MAPS: dict[str, dict[str, str]] = {
    "lesion_morphology": {"regular": "规则", "irregular": "不规则"},
    "lesion_mor_feature_lobulation": {"true": "有/清", "false": "无/不清"},
    "lesion_morph_feature_boundary": {"clear": "有/清", "unclear": "无/不清"},
    "tumor_t2wi_signal_intensity": {"high": "高", "low": "低", "iso": "等"},
    "tumor_t2_flair_sign_intensity": {"high": "高", "low": "低", "iso": "等"},
    "tumor_t1wi_c_enhan_pattern": {
        "ring": "环形",
        "rimenhancement": "花环状",
        "nodular": "结节状",
        "multifocal": "多灶状",
    },
}


def add_standardized_values(df: pd.DataFrame) -> pd.DataFrame:
    """Create *_std columns for stable evaluation values; keep raw values too."""

    out = df.copy()
    for field in SEMANTIC_FIELDS:
        if field not in out.columns:
            continue
        values = out[field].map(clean_id)
        if field == "glioma_with_label" or field.startswith("tumor_feature_"):
            out[f"{field}__std"] = values.map(
                lambda x: 1 if x.lower() in BOOL_TRUE else 0 if x.lower() in BOOL_FALSE else pd.NA
            ).astype("Int64")
        elif field in VALUE_MAPS:
            mapping = VALUE_MAPS[field]
            out[f"{field}__std"] = values.map(
                lambda x: mapping.get(x.lower(), x if x else pd.NA)
            )
    return out


def select_roi_name(names: list[str], priorities: list[str]) -> tuple[str, str]:
    """Return the preferred ROI and a machine-readable fallback reason."""

    available = set(names)
    for index, candidate in enumerate(priorities):
        if candidate in available:
            if index == 0:
                return candidate, "preferred"
            return candidate, f"fallback_missing:{'|'.join(priorities[:index])}"
    return "", "no_preferred_roi"


def build_roi_selection(base: pd.DataFrame, roi: pd.DataFrame) -> pd.DataFrame:
    """Create per-sequence ROI choices for the two segmentation targets.

    This does not create a mask.  It records which annotation should be passed
    to the image/mask loader and why a fallback was used.
    """

    t2_priorities = ["水肿", "全肿瘤", "瘤体"]
    t1ce_priorities = ["肿瘤瘤体", "瘤体", "全肿瘤"]
    grouped: dict[str, list[str]] = {}
    if "roi_name" in roi.columns:
        for key, group in roi[roi["_key"].ne("")].groupby("_key", sort=False):
            grouped[key] = list(dict.fromkeys(v for v in group["roi_name"].map(clean_id) if v))
    rows: list[dict[str, str]] = []
    for row in base[["_accession", "_series_uid", "_key"]].to_dict(orient="records"):
        names = grouped.get(row["_key"], [])
        t2_name, t2_reason = select_roi_name(names, t2_priorities)
        t1_name, t1_reason = select_roi_name(names, t1ce_priorities)
        rows.append(
            {
                "_accession": row["_accession"],
                "_series_uid": row["_series_uid"],
                "_key": row["_key"],
                "t2_flair_roi_name": t2_name,
                "t2_flair_fallback_reason": t2_reason,
                "t1ce_core_roi_name": t1_name,
                "t1ce_core_fallback_reason": t1_reason,
                "roi_candidate_names": "|".join(names),
            }
        )
    return pd.DataFrame(rows)


def deduplicate_one_to_one(
    df: pd.DataFrame,
    key: str,
    table: str,
    conflicts: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep one row per key and record conflicting non-key values for review."""

    source = df.copy()
    source = source.drop_duplicates().reset_index(drop=True)
    missing_key = int(source[key].eq("").sum())
    source = source[source[key].ne("")].copy()
    duplicate_groups = source[source.duplicated(key, keep=False)]
    duplicate_keys = int(duplicate_groups[key].nunique())
    ignored = {key, "_accession", "_series_uid"}
    for group_key, group in duplicate_groups.groupby(key, sort=False):
        for col in group.columns:
            if col in ignored or col.startswith("_"):
                continue
            values = sorted({clean_id(v) for v in group[col].tolist() if clean_id(v)})
            if len(values) > 1:
                conflicts.append(
                    {"table": table, "key": group_key, "field": col, "values": " || ".join(values)}
                )
    source = source.drop_duplicates(key, keep="first")
    return source, {
        "raw_rows": int(len(df)),
        "exact_dedup_rows": int(len(df.drop_duplicates())),
        "missing_key_rows": missing_key,
        "duplicate_key_count": duplicate_keys,
        "unique_key_rows": int(len(source)),
    }


def classify_sheets(path: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    sheets = pd.ExcelFile(path).sheet_names
    kinds: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for name in sheets:
        df, header_row = read_sheet(path, name)
        keys = {norm_name(c) for c in df.columns}
        sheet_key = norm_name(name)
        if (
            {"roiname", "roilabel", "roivolume"} & keys
            or "roi" in sheet_key
            or "roi" in "".join(keys)
        ):
            kind = "roi"
        elif (
            {"researchseriesuid", "seriesuid", "seriesdescription", "modality"} & keys
            or "序列" in str(name)
            or "series" in sheet_key
        ):
            kind = "series"
        else:
            kind = "check"
        kinds[kind] = name
        metadata[name] = {
            "header_row_zero_based": header_row,
            "columns": [str(c) for c in df.columns],
            "row_count": int(len(df)),
            "detected_kind": kind,
        }
    return kinds, metadata


def read_base(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    sheets = pd.ExcelFile(path).sheet_names
    df, header_row = read_sheet(path, sheets[0])
    df, selected = add_keys(df, "base")
    series_type_col = choose_column(df, ["SeriesType", "序列类型"], required=True, label="base SeriesType")
    df["SeriesType"] = df[series_type_col].map(clean_id)
    return df, {
        "sheet": sheets[0],
        "header_row_zero_based": header_row,
        "selected_columns": selected,
        "columns": [str(c) for c in df.columns],
    }


def jsonable(value: Any) -> Any:
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def records_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(k): jsonable(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def public_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Expose stable field names in flat outputs while retaining JoinKey."""

    out = df.rename(
        columns={"_accession": "AccessionNumber", "_series_uid": "SeriesUid", "_key": "JoinKey"}
    )
    if {"AccessionNumber", "SeriesUid", "JoinKey"}.issubset(out.columns):
        valid = out["AccessionNumber"].map(clean_id).ne("") & out["SeriesUid"].map(clean_id).ne("")
        out["JoinKey"] = (
            out["AccessionNumber"].map(clean_id) + "|" + out["SeriesUid"].map(clean_id)
        ).where(valid, "")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, help="Directory containing the two workbooks")
    parser.add_argument("--series-type", type=Path, help="SeriesType.xlsx; defaults to annotation-dir/SeriesType.xlsx")
    parser.add_argument(
        "--annotation", "--annotations", dest="annotations", type=Path,
        help="Annotation workbook; defaults to annotation-dir/脑胶质瘤标注结果-训练集.xlsx",
    )
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", type=Path, default=Path("outputs/annotations"))
    parser.add_argument("--check-sheet", help="Override auto-detected check-level sheet name")
    parser.add_argument("--series-sheet", help="Override auto-detected series-level sheet name")
    parser.add_argument("--roi-sheet", help="Override auto-detected ROI-level sheet name")
    parser.add_argument(
        "--force-series-uid-column",
        help="Force the sequence/ROI UID source, e.g. 'StudyUid' or 'PatientId'; QC still reports overlap evidence",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.annotation_dir is None and (args.series_type is None or args.annotations is None):
        raise ValueError("Provide --annotation-dir or both --series-type and --annotation")
    series_type_path = args.series_type or args.annotation_dir / "SeriesType.xlsx"
    annotations_path = args.annotations or args.annotation_dir / "脑胶质瘤标注结果-训练集.xlsx"
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not series_type_path.exists():
        raise FileNotFoundError(f"SeriesType workbook not found: {series_type_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotation workbook not found: {annotations_path}")

    conflicts: list[dict[str, Any]] = []
    base, base_meta = read_base(series_type_path)
    base = add_semantic_fields(base)
    base, base_stats = deduplicate_one_to_one(base, "_key", "base", conflicts)

    detected, sheet_meta = classify_sheets(annotations_path)
    selected_sheets = {
        "check": args.check_sheet or detected.get("check"),
        "series": args.series_sheet or detected.get("series"),
        "roi": args.roi_sheet or detected.get("roi"),
    }
    if any(v is None for v in selected_sheets.values()):
        raise ValueError(f"Could not detect all three annotation sheets. Detected: {detected}")

    check, check_header = read_sheet(annotations_path, selected_sheets["check"])
    check, check_selected = add_check_key(check)
    check = add_semantic_fields(check)
    check = add_standardized_values(check)
    check, check_stats = deduplicate_one_to_one(check, "_accession", "check", conflicts)

    base_series_uids = set(base.loc[base["_series_uid"].ne(""), "_series_uid"])

    seq, seq_header = read_sheet(annotations_path, selected_sheets["series"])
    seq, seq_selected = add_keys(
        seq, "series", base_series_uids, args.force_series_uid_column
    )
    seq = add_semantic_fields(seq)
    seq = add_standardized_values(seq)
    seq, seq_stats = deduplicate_one_to_one(seq, "_key", "series", conflicts)

    roi, roi_header = read_sheet(annotations_path, selected_sheets["roi"])
    roi, roi_selected = add_keys(
        roi, "roi", base_series_uids, args.force_series_uid_column
    )
    roi = add_semantic_fields(roi)
    roi = add_standardized_values(roi)
    roi_before = len(roi)
    roi = roi.drop_duplicates().reset_index(drop=True)
    roi_stats = {
        "raw_rows": int(roi_before),
        "exact_dedup_rows": int(len(roi)),
        "missing_accession_rows": int(roi["_accession"].eq("").sum()),
        "missing_series_uid_rows": int(roi["_series_uid"].eq("").sum()),
        "unique_key_rows": int(roi.loc[roi["_key"].ne(""), "_key"].nunique()),
    }

    # Keep only one source copy of the IDs in output; raw fields remain under
    # check__/series__/roi__ prefixes for auditability.
    base_payload = prefix_payload(base, "base", preserve={"SeriesType", "series_uid_source"})
    check_payload = prefix_payload(check, "check")
    seq_payload = prefix_payload(seq, "series")
    merged = base_payload.merge(check_payload, on="_accession", how="left", suffixes=("", "__check"))
    merged = merged.merge(seq_payload, on=["_accession", "_series_uid", "_key"], how="left", suffixes=("", "__series"))
    roi_selection = build_roi_selection(base, roi)
    merged = merged.merge(roi_selection, on=["_accession", "_series_uid", "_key"], how="left")
    merged_public = public_keys(merged)
    merged_public.to_csv(out_dir / "series_merged.csv", index=False, encoding="utf-8-sig")
    merged_json = records_json(merged_public)
    write_json(out_dir / "series_merged.json", merged_json)

    roi_payload = prefix_payload(roi, "roi")
    roi_merged = roi_payload.merge(
        base_payload[["_accession", "_series_uid", "_key", "SeriesType"]],
        on=["_accession", "_series_uid", "_key"],
        how="left",
        suffixes=("", "__base"),
    )
    roi_merged_public = public_keys(roi_merged)
    roi_merged_public.to_csv(out_dir / "roi_merged.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "roi_merged.json", records_json(roi_merged_public))
    roi_selection_public = public_keys(roi_selection)
    roi_selection_public.to_csv(out_dir / "roi_selection.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "roi_selection.json", records_json(roi_selection_public))

    # Nested output is convenient for a later model/dataset loader.
    check_by_accession = check.set_index("_accession", drop=False).to_dict(orient="index")
    seq_by_key = seq.set_index("_key", drop=False).to_dict(orient="index")
    roi_groups = {key: records_json(group) for key, group in roi.groupby("_key", sort=False) if key}
    selection_by_key = roi_selection.set_index("_key", drop=False).to_dict(orient="index")
    nested: list[dict[str, Any]] = []
    for row in base.to_dict(orient="records"):
        key = row["_key"]
        nested.append(
            {
                "AccessionNumber": row["_accession"],
                "SeriesUid": row["_series_uid"],
                "SeriesType": row.get("SeriesType", ""),
                "check": {str(k): jsonable(v) for k, v in check_by_accession.get(row["_accession"], {}).items() if not str(k).startswith("_")},
                "series": {str(k): jsonable(v) for k, v in seq_by_key.get(key, {}).items() if not str(k).startswith("_")},
                "rois": roi_groups.get(key, []),
                "roi_selection": {
                    str(k): jsonable(v)
                    for k, v in selection_by_key.get(key, {}).items()
                    if not str(k).startswith("_")
                },
            }
        )
    write_json(out_dir / "series_nested.json", nested)

    base_keys = set(base["_key"]) - {""}
    seq_keys = set(seq["_key"]) - {""}
    roi_keys = set(roi.loc[roi["_key"].ne(""), "_key"])
    unmatched = []
    for kind, keys in [("series", seq_keys - base_keys), ("roi", roi_keys - base_keys), ("base_without_series", base_keys - seq_keys)]:
        unmatched.extend({"kind": kind, "key": key} for key in sorted(keys))
    pd.DataFrame(unmatched, columns=["kind", "key"]).to_csv(out_dir / "unmatched_keys.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(conflicts, columns=["table", "key", "field", "values"]).to_csv(
        out_dir / "duplicate_conflicts.csv", index=False, encoding="utf-8-sig"
    )

    qc = {
        "inputs": {"series_type": str(series_type_path), "annotations": str(annotations_path)},
        "sheets": sheet_meta,
        "selected_sheets": selected_sheets,
        "selected_columns": {"base": base_meta["selected_columns"], "check": check_selected, "series": seq_selected, "roi": roi_selected},
        "counts": {"base": base_stats, "check": check_stats, "series": seq_stats, "roi": roi_stats, "series_output": int(len(merged)), "roi_output": int(len(roi_merged))},
        "coverage": {
            "base_keys": len(base_keys),
            "series_keys": len(seq_keys),
            "roi_keys": len(roi_keys),
            "base_without_series": len(base_keys - seq_keys),
            "series_not_in_base": len(seq_keys - base_keys),
            "roi_not_in_base": len(roi_keys - base_keys),
            "check_accessions": int(check["_accession"].ne("").sum()),
        },
        "duplicate_conflict_count": len(conflicts),
        "expected_counts": EXPECTED,
        "count_warnings": {
            name: {"observed": observed, "expected": EXPECTED[name]}
            for name, observed in {
                "base": base_stats["unique_key_rows"],
                "check": check_stats["unique_key_rows"],
                "series": seq_stats["unique_key_rows"],
                "roi": len(roi),
            }.items()
            if observed != EXPECTED[name]
        },
    }
    write_json(out_dir / "qc_report.json", qc)
    print(json.dumps({"output_dir": str(out_dir), "selected_sheets": selected_sheets, "counts": qc["counts"], "coverage": qc["coverage"], "count_warnings": qc["count_warnings"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
