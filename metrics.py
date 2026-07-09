"""
Compares teacher (annotation-time) outputs vs. student (trained model)
outputs on a held-out sample. Supports both whole-JSON similarity (legacy)
and field-level accuracy (recommended) for a specific target key like "claim".
"""

import json
import difflib
import pandas as pd


def _safe_json(s):
    try:
        return json.loads(s) if isinstance(s, str) else s
    except Exception:
        return {}


def json_similarity(a: dict, b: dict) -> float:
    a_str = json.dumps(a, sort_keys=True)
    b_str = json.dumps(b, sort_keys=True)
    return difflib.SequenceMatcher(None, a_str, b_str).ratio()


def exact_match(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def valid_json_rate(outputs: list) -> float:
    valid = sum(1 for o in outputs if _safe_json(o))
    return valid / len(outputs) if outputs else 0.0


def _normalize_field(v):
    if isinstance(v, str):
        return v.strip().lower()
    return v


def field_match(t: dict, s: dict, field: str) -> bool:
    if field not in t or field not in s:
        return False
    return _normalize_field(t[field]) == _normalize_field(s[field])


def compare_teacher_student(teacher_outputs: list, student_outputs: list, target_field: str = None) -> dict:
    """
    teacher_outputs / student_outputs: lists of JSON strings (same order, same inputs)
    target_field: if provided (e.g. "claim"), also computes field-level accuracy
                  on that specific key -- this is usually what actually matters,
                  since free-text fields like "justification" will rarely match exactly.
    Returns aggregate metrics dict + per-row dataframe.
    """
    assert len(teacher_outputs) == len(student_outputs), "Mismatched lengths"

    rows = []
    similarities = []
    exact_matches = 0
    field_matches = 0
    field_comparable = 0

    for t_raw, s_raw in zip(teacher_outputs, student_outputs):
        t = _safe_json(t_raw)
        s = _safe_json(s_raw)
        sim = json_similarity(t, s)
        em = exact_match(t, s)
        similarities.append(sim)
        exact_matches += int(em)

        row = {
            "teacher_output": t_raw,
            "student_output": s_raw,
            "similarity": round(sim, 3),
            "exact_match": em
        }

        if target_field:
            has_both = target_field in t and target_field in s
            fm = field_match(t, s, target_field) if has_both else False
            row[f"teacher_{target_field}"] = t.get(target_field, None)
            row[f"student_{target_field}"] = s.get(target_field, None)
            row[f"{target_field}_match"] = fm
            if has_both:
                field_comparable += 1
                field_matches += int(fm)

        rows.append(row)

    n = len(teacher_outputs)
    summary = {
        "n_samples": n,
        "avg_similarity": round(sum(similarities) / n, 4) if n else 0,
        "exact_match_rate": round(exact_matches / n, 4) if n else 0,
        "teacher_valid_json_rate": round(valid_json_rate(teacher_outputs), 4),
        "student_valid_json_rate": round(valid_json_rate(student_outputs), 4),
    }

    if target_field:
        summary[f"{target_field}_accuracy"] = (
            round(field_matches / field_comparable, 4) if field_comparable else 0.0
        )
        summary[f"{target_field}_comparable_rows"] = field_comparable

    return summary, pd.DataFrame(rows)


def run_student_inference(connector, system_prompt, texts: list, max_tokens=800):
    """
    Runs the trained/selected student connector over a list of raw texts
    and returns JSON-string outputs, mirroring the teacher annotation format.
    """
    from silver_generator import annotate_text
    outputs = []
    for text in texts:
        parsed, _ = annotate_text(connector, system_prompt, text, max_tokens=max_tokens)
        outputs.append(json.dumps(parsed))
    return outputs
