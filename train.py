from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

from github import Auth, Github
from github.GithubException import GithubException
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from services.classifier import classify_profile
from services.github_scraper import fetch_github_profile
from services.nlp_pipeline import FEATURE_NAMES, extract_features


MODEL_OUTPUT = Path("models/archetype_model.pkl")


def build_search_queries() -> List[str]:
    return [
        "followers:>50 repos:>10 language:Python",
        "followers:>50 repos:>10 language:JavaScript",
        "followers:>50 repos:>10 language:TypeScript",
        "followers:>50 repos:>10 language:Go",
        "followers:>50 repos:>10 language:Rust",
        "followers:>50 repos:>10 language:Java",
        "followers:>50 repos:>10 language:C++",
    ]


def fetch_training_usernames(target_count: int = 500) -> List[str]:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN must be set to train the model.")

    client = Github(auth=Auth.Token(token), per_page=100)
    usernames: set[str] = set()

    for query in build_search_queries():
        try:
            users = client.search_users(query=query, sort="followers", order="desc")
            for user in users:
                usernames.add(user.login)
                if len(usernames) >= target_count:
                    break
        except GithubException as exc:
            print(f"Skipping search query {query!r}: {exc}")
        if len(usernames) >= target_count:
            break

    username_list = list(usernames)
    random.shuffle(username_list)
    return username_list[:target_count]


def build_dataset(usernames: List[str]) -> tuple[List[List[float]], List[str], Dict[str, Dict[str, float]]]:
    rows: List[List[float]] = []
    labels: List[str] = []
    feature_history: Dict[str, List[float]] = {name: [] for name in FEATURE_NAMES}

    for index, username in enumerate(usernames, start=1):
        try:
            raw_data = fetch_github_profile(username)
            feature_data = extract_features(raw_data, extract_topics=False)
            classification = classify_profile(feature_data, model_bundle=None)
        except Exception as exc:
            print(f"[{index}/{len(usernames)}] Skipping {username}: {exc}")
            continue

        vector = [feature_data["raw_features"][name] for name in FEATURE_NAMES]
        rows.append(vector)
        labels.append(classification["archetype"])
        for feature_name, value in zip(FEATURE_NAMES, vector):
            feature_history[feature_name].append(float(value))
        print(f"[{index}/{len(usernames)}] Collected {username} -> {classification['archetype']}")

    baseline_stats = {
        feature_name: {
            "mean": float(mean(values)) if values else 0.0,
            "std": float(pstdev(values)) if len(values) > 1 else 1.0,
        }
        for feature_name, values in feature_history.items()
    }
    return rows, labels, baseline_stats


def main() -> None:
    usernames = fetch_training_usernames()
    rows, labels, baseline_stats = build_dataset(usernames)
    if len(rows) < 20:
        raise RuntimeError("Not enough training data was collected to build a model.")

    x_train, x_test, y_train, y_test = train_test_split(
        rows,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    print(classification_report(y_test, predictions))

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    bundle: Dict[str, Any] = {
        "model": pipeline.named_steps["model"],
        "scaler": pipeline.named_steps["scaler"],
        "feature_names": FEATURE_NAMES,
        "baseline_stats": baseline_stats,
    }
    with MODEL_OUTPUT.open("wb") as handle:
        pickle.dump(bundle, handle)
    print(f"Saved model bundle to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
