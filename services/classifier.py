from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from services.nlp_pipeline import DEFAULT_BASELINE_STATS, FEATURE_NAMES


ARCHETYPE_COLORS = {
    "Builder": "#2ea44f",
    "Fixer": "#f78166",
    "Documenter": "#58a6ff",
    "Experimenter": "#d2a8ff",
}


INTERPRETATIONS = {
    "Builder": "This profile suggests someone who spends most of their energy shipping, expanding, and iterating on working software. Their GitHub history leans toward forward progress, implementation momentum, and practical delivery.",
    "Fixer": "This profile points to an engineer who gets pulled toward reliability work and production-quality cleanup. Their history shows a strong bias for resolving issues, tightening rough edges, and stabilizing code that others depend on.",
    "Documenter": "This profile looks like someone who invests in clarity as much as code. Their repositories and commit history suggest a habit of explaining decisions, maintaining readable project surfaces, and making collaboration easier.",
    "Experimenter": "This profile suggests a curious engineer who explores broadly and tests ideas in motion. Their activity patterns point toward prototyping, trying new tools, and learning through hands-on iteration across different stacks.",
}


SIGNAL_MAP = {
    "Fixer": ["bug_fix_ratio", "technical_depth_score", "commit_msg_specificity"],
    "Documenter": ["doc_ratio", "avg_readme_length", "readme_consistency"],
    "Experimenter": ["experiment_ratio", "repo_diversity", "feature_ratio"],
    "Builder": ["feature_ratio", "technical_depth_score", "commit_msg_length"],
}


def load_model_bundle(model_path: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        return bundle
    except Exception:
        return None


def _format_signal(feature_name: str, raw_value: float) -> str:
    label = feature_name.replace("_", " ")
    if raw_value <= 1:
        return f"{label} at {raw_value:.2f}"
    return f"{label} at {raw_value:.1f}"


def _confidence_from_rule(archetype: str, raw_features: Dict[str, float]) -> float:
    if archetype == "Fixer":
        score = min(1.0, 0.55 + raw_features["bug_fix_ratio"])
    elif archetype == "Documenter":
        consistency_bonus = max(0.0, 0.3 - raw_features["readme_consistency"])
        score = min(1.0, 0.55 + raw_features["doc_ratio"] + consistency_bonus)
    elif archetype == "Experimenter":
        diversity_bonus = min(raw_features["repo_diversity"] / 10.0, 0.25)
        score = min(1.0, 0.5 + raw_features["experiment_ratio"] + diversity_bonus)
    else:
        score = min(1.0, 0.5 + raw_features["feature_ratio"] + (raw_features["technical_depth_score"] / 100.0))
    return round(max(0.51, score), 2)


def _rule_based_classification(feature_data: Dict[str, Any]) -> Dict[str, Any]:
    raw_features = feature_data["raw_features"]

    if raw_features["bug_fix_ratio"] > 0.35:
        archetype = "Fixer"
    elif raw_features["doc_ratio"] > 0.25 or raw_features["readme_consistency"] < 0.3:
        archetype = "Documenter"
    elif raw_features["experiment_ratio"] > 0.25 or raw_features["repo_diversity"] > 5:
        archetype = "Experimenter"
    else:
        archetype = "Builder"

    top_signals = [
        _format_signal(feature_name, raw_features[feature_name])
        for feature_name in SIGNAL_MAP[archetype]
    ][:3]

    return {
        "archetype": archetype,
        "confidence": _confidence_from_rule(archetype, raw_features),
        "top_signals": top_signals,
        "model_used": "rules",
    }


def _model_based_classification(feature_data: Dict[str, Any], model_bundle: Dict[str, Any]) -> Dict[str, Any]:
    model = model_bundle["model"]
    scaler = model_bundle.get("scaler")
    feature_names = model_bundle.get("feature_names", FEATURE_NAMES)
    raw_features = feature_data["raw_features"]
    vector = [[raw_features[name] for name in feature_names]]
    transformed = scaler.transform(vector) if scaler is not None else vector

    probabilities = model.predict_proba(transformed)[0]
    classes = list(model.classes_)
    best_index = max(range(len(probabilities)), key=lambda idx: probabilities[idx])
    archetype = classes[best_index]
    confidence = round(float(probabilities[best_index]), 2)

    candidate_features = SIGNAL_MAP.get(archetype, feature_names[:3])
    top_signals = [
        _format_signal(feature_name, raw_features[feature_name])
        for feature_name in candidate_features[:3]
    ]

    return {
        "archetype": archetype,
        "confidence": confidence,
        "top_signals": top_signals,
        "model_used": "ml",
    }


def build_radar_scores(feature_data: Dict[str, Any]) -> Dict[str, float]:
    percentiles = feature_data["percentiles"]
    return {
        "Technical Depth": round(
            (percentiles["technical_depth_score"] + percentiles["commit_msg_specificity"] + percentiles["bug_fix_ratio"]) / 3,
            1,
        ),
        "Communication": round(
            (percentiles["communication_directness"] + percentiles["avg_sentiment"] + percentiles["vocabulary_richness"]) / 3,
            1,
        ),
        "Collaboration": round(
            (percentiles["issue_tone"] + percentiles["pr_tone"] + percentiles["avg_topics_per_repo"]) / 3,
            1,
        ),
        "Breadth": round(
            (percentiles["repo_diversity"] + percentiles["experiment_ratio"] + percentiles["feature_ratio"]) / 3,
            1,
        ),
        "Consistency": round(
            ((100.0 - percentiles["readme_consistency"]) + percentiles["commit_msg_length"] + (100.0 - percentiles["repo_churn_rate"])) / 3,
            1,
        ),
        "Documentation": round(
            (percentiles["avg_readme_length"] + percentiles["doc_ratio"] + percentiles["code_blocks_ratio"]) / 3,
            1,
        ),
    }


def classify_profile(feature_data: Dict[str, Any], model_bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    baseline_stats = DEFAULT_BASELINE_STATS
    if model_bundle and model_bundle.get("baseline_stats"):
        baseline_stats = model_bundle["baseline_stats"]

    result = _model_based_classification(feature_data, model_bundle) if model_bundle else _rule_based_classification(feature_data)
    archetype = result["archetype"]
    radar_scores = build_radar_scores(feature_data)

    result["color"] = ARCHETYPE_COLORS[archetype]
    result["interpretation"] = INTERPRETATIONS[archetype]
    result["radar_scores"] = radar_scores
    result["technical_depth_percentile"] = round(float(feature_data["technical_depth_percentile"]), 1)
    result["baseline_stats"] = baseline_stats
    return result
