from __future__ import annotations

import math
import re
from collections import Counter
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


FEATURE_NAMES = [
    "avg_readme_length",
    "readme_consistency",
    "commit_msg_length",
    "commit_msg_specificity",
    "vocabulary_richness",
    "readability_score",
    "technical_depth_score",
    "code_blocks_ratio",
    "communication_directness",
    "avg_sentiment",
    "issue_tone",
    "pr_tone",
    "repo_diversity",
    "repo_churn_rate",
    "avg_topics_per_repo",
    "bug_fix_ratio",
    "feature_ratio",
    "doc_ratio",
    "experiment_ratio",
]


DEFAULT_BASELINE_STATS: Dict[str, Dict[str, float]] = {
    "avg_readme_length": {"mean": 32.0, "std": 18.0},
    "readme_consistency": {"mean": 15.0, "std": 8.0},
    "commit_msg_length": {"mean": 6.5, "std": 2.5},
    "commit_msg_specificity": {"mean": 0.42, "std": 0.15},
    "vocabulary_richness": {"mean": 0.48, "std": 0.12},
    "readability_score": {"mean": 10.5, "std": 3.5},
    "technical_depth_score": {"mean": 18.0, "std": 9.0},
    "code_blocks_ratio": {"mean": 0.28, "std": 0.17},
    "communication_directness": {"mean": 0.88, "std": 0.08},
    "avg_sentiment": {"mean": 0.14, "std": 0.18},
    "issue_tone": {"mean": 0.16, "std": 0.2},
    "pr_tone": {"mean": 0.18, "std": 0.2},
    "repo_diversity": {"mean": 3.0, "std": 1.8},
    "repo_churn_rate": {"mean": 0.18, "std": 0.13},
    "avg_topics_per_repo": {"mean": 2.0, "std": 1.2},
    "bug_fix_ratio": {"mean": 0.18, "std": 0.11},
    "feature_ratio": {"mean": 0.24, "std": 0.12},
    "doc_ratio": {"mean": 0.11, "std": 0.08},
    "experiment_ratio": {"mean": 0.09, "std": 0.08},
}


FILLER_WORDS = {
    "stuff",
    "things",
    "misc",
    "update",
    "changes",
    "some",
    "various",
    "temp",
    "cleanup",
    "fixes",
    "work",
    "minor",
    "small",
    "quick",
    "notes",
    "todo",
    "wip",
}


FORMAL_MARKERS = {"please", "kindly", "regarding", "appreciate", "therefore", "thanks"}
CASUAL_MARKERS = {"hey", "cool", "awesome", "super", "btw", "lol", "thanks!", "nice"}


TECHNICAL_TERMS = [
    "api",
    "sdk",
    "cli",
    "orm",
    "rest",
    "graphql",
    "grpc",
    "http",
    "https",
    "tcp",
    "udp",
    "dns",
    "tls",
    "ssl",
    "oauth",
    "jwt",
    "saml",
    "openid",
    "proxy",
    "reverse proxy",
    "load balancer",
    "cache",
    "caching",
    "redis",
    "memcached",
    "database",
    "postgres",
    "postgresql",
    "mysql",
    "sqlite",
    "mongodb",
    "cassandra",
    "dynamodb",
    "elasticsearch",
    "opensearch",
    "indexing",
    "query planner",
    "migration",
    "schema",
    "sharding",
    "replication",
    "consistency",
    "availability",
    "partition tolerance",
    "acid",
    "cap theorem",
    "transaction",
    "rollback",
    "cursor",
    "join",
    "normalization",
    "denormalization",
    "vector database",
    "embedding",
    "transformer",
    "tokenizer",
    "llm",
    "machine learning",
    "deep learning",
    "neural network",
    "gradient descent",
    "classifier",
    "regression",
    "random forest",
    "logistic regression",
    "precision",
    "recall",
    "f1 score",
    "inference",
    "training",
    "fine tuning",
    "dataset",
    "feature engineering",
    "pipeline",
    "batching",
    "streaming",
    "event driven",
    "message queue",
    "kafka",
    "rabbitmq",
    "pubsub",
    "webhook",
    "cron",
    "scheduler",
    "worker",
    "queue",
    "async",
    "await",
    "threadpool",
    "concurrency",
    "parallelism",
    "multithreading",
    "multiprocessing",
    "mutex",
    "semaphore",
    "deadlock",
    "race condition",
    "latency",
    "throughput",
    "benchmark",
    "profiling",
    "memory leak",
    "garbage collection",
    "allocator",
    "compiler",
    "interpreter",
    "bytecode",
    "jit",
    "runtime",
    "vm",
    "virtual machine",
    "container",
    "docker",
    "kubernetes",
    "helm",
    "terraform",
    "ansible",
    "iac",
    "infrastructure",
    "serverless",
    "lambda",
    "cloudformation",
    "aws",
    "gcp",
    "azure",
    "cdn",
    "edge",
    "observability",
    "tracing",
    "telemetry",
    "metrics",
    "logging",
    "alerting",
    "prometheus",
    "grafana",
    "sentry",
    "new relic",
    "datadog",
    "opentelemetry",
    "unit test",
    "integration test",
    "e2e",
    "snapshot",
    "mock",
    "stub",
    "fixture",
    "test harness",
    "lint",
    "formatter",
    "ci",
    "cd",
    "deployment",
    "rollback strategy",
    "feature flag",
    "canary",
    "blue green",
    "release",
    "semantic versioning",
    "monorepo",
    "microservice",
    "service mesh",
    "domain driven",
    "hexagonal",
    "clean architecture",
    "mvc",
    "mvvm",
    "component",
    "frontend",
    "backend",
    "full stack",
    "react",
    "vue",
    "svelte",
    "angular",
    "fastapi",
    "django",
    "flask",
    "rails",
    "spring",
    "express",
    "node",
    "typescript",
    "javascript",
    "python",
    "rust",
    "go",
    "java",
    "kotlin",
    "swift",
    "objective c",
    "cpp",
    "c++",
    "csharp",
    "c#",
    "wasm",
    "webassembly",
    "binary",
    "serialization",
    "protobuf",
    "avro",
    "json",
    "yaml",
    "toml",
    "xml",
    "parsing",
    "lexer",
    "parser",
    "ast",
    "regex",
    "algorithm",
    "data structure",
    "hash map",
    "hash table",
    "binary tree",
    "trie",
    "heap",
    "graph",
    "dfs",
    "bfs",
    "dynamic programming",
    "greedy",
    "backtracking",
    "sort",
    "search",
    "binary search",
    "big o",
    "complexity",
    "optimization",
    "vectorization",
    "simd",
    "gpu",
    "cuda",
    "opencl",
    "security",
    "encryption",
    "decryption",
    "hashing",
    "salting",
    "signature",
    "xss",
    "csrf",
    "sql injection",
    "sandbox",
    "permission",
    "auth",
    "authorization",
    "authentication",
    "access control",
    "zero trust",
    "firewall",
    "scanner",
    "static analysis",
    "dynamic analysis",
    "fuzzing",
    "crash",
    "segfault",
    "panic",
    "exception",
    "retry",
    "backoff",
    "idempotent",
    "deduplication",
    "eventual consistency",
    "websocket",
    "polling",
    "sse",
    "throttling",
    "debounce",
    "pagination",
    "cursor pagination",
    "rate limiting",
]


BUG_WORDS = re.compile(r"\b(fix|bug|patch|resolve|hotfix)\b", re.IGNORECASE)
FEATURE_WORDS = re.compile(r"\b(add|implement|create|build|new)\b", re.IGNORECASE)
DOC_WORDS = re.compile(r"\b(docs?|readme|comment|document(?:ation)?)\b", re.IGNORECASE)
EXPERIMENT_WORDS = re.compile(r"\b(test|try|prototype|experiment|poc)\b", re.IGNORECASE)
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_+#./-]*")


_SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()
_TECH_PATTERN = re.compile(
    "|".join(sorted((re.escape(term) for term in TECHNICAL_TERMS), key=len, reverse=True)),
    re.IGNORECASE,
)


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in WORD_PATTERN.findall(text)]


def _safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _safe_readability(texts: Sequence[str]) -> float:
    combined = " ".join(text for text in texts if text and text.strip())
    if not combined:
        return 0.0
    try:
        return float(textstat.flesch_kincaid_grade(combined))
    except Exception:
        normalized = " ".join(re.findall(r"[A-Za-z]+", combined))
        if not normalized:
            return 0.0
        try:
            return float(textstat.flesch_kincaid_grade(normalized))
        except Exception:
            return 0.0


def percentile_from_stats(value: float, stats: Dict[str, float]) -> float:
    std = max(stats.get("std", 1.0), 1e-6)
    z_score = (value - stats.get("mean", 0.0)) / std
    percentile = 50.0 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
    return max(0.0, min(100.0, percentile))


def _sentiment_scores(texts: Iterable[str]) -> List[float]:
    scores: List[float] = []
    for text in texts:
        cleaned = text.strip()
        if not cleaned:
            continue
        scores.append(_SENTIMENT_ANALYZER.polarity_scores(cleaned)["compound"])
    return scores


def _extract_topics_with_bertopic(documents: List[str]) -> List[str]:
    try:
        from bertopic import BERTopic
    except Exception:
        return []

    if len(documents) < 4:
        return []

    try:
        topic_model = BERTopic(verbose=False, min_topic_size=max(2, min(5, len(documents) // 2)))
        topic_model.fit_transform(documents)
        topic_info = topic_model.get_topic_info()
        topics: List[str] = []
        for _, row in topic_info.iterrows():
            topic_id = int(row["Topic"])
            if topic_id == -1:
                continue
            words = topic_model.get_topic(topic_id) or []
            keywords = [word for word, _score in words[:3] if word and word != "-1"]
            if keywords:
                topics.append(" / ".join(keywords))
            if len(topics) >= 5:
                break
        return topics
    except Exception:
        return []


def _fallback_topics(raw_data: Dict[str, Any]) -> List[str]:
    counter: Counter[str] = Counter()
    for repo in raw_data.get("repos", []):
        for topic in repo.get("topics", []):
            if topic:
                counter[topic.lower()] += 3
        for text in [repo.get("language", ""), repo.get("description", ""), repo.get("readme", "")[:3000]]:
            for token in _tokenize(text):
                if len(token) < 4 or token in FILLER_WORDS:
                    continue
                counter[token] += 1
    return [topic for topic, _ in counter.most_common(5)]


def _extract_topics(raw_data: Dict[str, Any]) -> List[str]:
    documents = []
    for repo in raw_data.get("repos", []):
        doc = " ".join(
            part
            for part in [
                repo.get("description", ""),
                " ".join(repo.get("topics", [])),
                repo.get("readme", "")[:4000],
            ]
            if part
        ).strip()
        if doc:
            documents.append(doc)

    return _fallback_topics(raw_data)


def _communication_style(texts: Sequence[str], directness_percentile: float) -> Dict[str, Any]:
    joined = " ".join(texts)
    tokens = _tokenize(joined)
    formal_count = sum(token in FORMAL_MARKERS for token in tokens)
    casual_count = sum(token in CASUAL_MARKERS for token in tokens) + joined.count("!")

    direct_label = "Direct" if directness_percentile >= 55 else "Indirect"
    formality_label = "Formal" if formal_count >= casual_count else "Casual"

    indicators: List[str] = []
    if direct_label == "Direct":
        indicators.append("Low sentiment variance across comments and PRs")
    else:
        indicators.append("Tone shifts noticeably across discussions")
    if formality_label == "Formal":
        indicators.append("Language leans structured and professional")
    else:
        indicators.append("Language includes relaxed or conversational markers")

    return {
        "directness": direct_label,
        "formality": formality_label,
        "indicators": indicators,
    }


def extract_features(
    raw_data: Dict[str, Any],
    baseline_stats: Dict[str, Dict[str, float]] | None = None,
    extract_topics: bool = True,
    use_bertopic: bool = False,
) -> Dict[str, Any]:
    repos = raw_data.get("repos", [])
    commit_entries = raw_data.get("commit_messages", [])
    issue_entries = raw_data.get("issue_comments", [])
    pr_entries = raw_data.get("pull_requests", [])

    readme_densities: List[float] = []
    readme_code_block_hits = 0
    combined_texts: List[str] = []

    for repo in repos:
        readme = repo.get("readme", "") or ""
        description = repo.get("description", "") or ""
        combined_texts.extend([readme, description])
        if readme.strip():
            words = _tokenize(readme)
            size_kb = max(float(repo.get("size_kb", 0) or 0), 1.0)
            readme_densities.append(len(words) / size_kb)
            if "```" in readme:
                readme_code_block_hits += 1

    commit_texts = [entry.get("message", "") for entry in commit_entries if entry.get("message")]
    issue_texts = [entry.get("body", "") for entry in issue_entries if entry.get("body")]
    pr_texts = [
        " ".join(part for part in [entry.get("title", ""), entry.get("body", "")] if part).strip()
        for entry in pr_entries
        if entry.get("title") or entry.get("body")
    ]
    combined_texts.extend(commit_texts)
    combined_texts.extend(issue_texts)
    combined_texts.extend(pr_texts)

    commit_lengths = [len(_tokenize(message)) for message in commit_texts if message.strip()]
    commit_tokens = [token for message in commit_texts for token in _tokenize(message)]
    all_tokens = [token for text in combined_texts for token in _tokenize(text)]

    technical_matches = _TECH_PATTERN.findall(" ".join(combined_texts))
    technical_depth_score = float(len(technical_matches))
    tech_token_hits = sum(1 for token in commit_tokens if _TECH_PATTERN.search(token))
    filler_hits = sum(1 for token in commit_tokens if token in FILLER_WORDS)
    specificity_denominator = tech_token_hits + filler_hits
    commit_specificity = (
        tech_token_hits / specificity_denominator if specificity_denominator else (tech_token_hits / max(len(commit_tokens), 1))
    )

    sentiment_scores = _sentiment_scores(combined_texts)
    issue_sentiments = _sentiment_scores(issue_texts)
    pr_sentiments = _sentiment_scores(pr_texts)
    sentiment_variance = _safe_std(sentiment_scores)

    repo_languages = {
        repo.get("language")
        for repo in repos
        if repo.get("language") and repo.get("language") != "Unknown"
    }
    stale_like = raw_data.get("stats", {}).get("churn_like_repo_count", 0)

    raw_features = {
        "avg_readme_length": _safe_mean(readme_densities),
        "readme_consistency": _safe_std(readme_densities),
        "commit_msg_length": _safe_mean(commit_lengths),
        "commit_msg_specificity": float(commit_specificity),
        "vocabulary_richness": float(len(set(all_tokens)) / max(len(all_tokens), 1)),
        "readability_score": _safe_readability(combined_texts),
        "technical_depth_score": technical_depth_score,
        "code_blocks_ratio": float(readme_code_block_hits / max(len([repo for repo in repos if repo.get("readme")]), 1)),
        "communication_directness": float(max(0.0, 1.0 - sentiment_variance)),
        "avg_sentiment": _safe_mean(sentiment_scores),
        "issue_tone": _safe_mean(issue_sentiments),
        "pr_tone": _safe_mean(pr_sentiments),
        "repo_diversity": float(len(repo_languages)),
        "repo_churn_rate": float(stale_like / max(len(repos), 1)),
        "avg_topics_per_repo": float(
            _safe_mean([len(repo.get("topics", [])) for repo in repos]) if repos else 0.0
        ),
        "bug_fix_ratio": float(sum(bool(BUG_WORDS.search(message)) for message in commit_texts) / max(len(commit_texts), 1)),
        "feature_ratio": float(
            sum(bool(FEATURE_WORDS.search(message)) for message in commit_texts) / max(len(commit_texts), 1)
        ),
        "doc_ratio": float(sum(bool(DOC_WORDS.search(message)) for message in commit_texts) / max(len(commit_texts), 1)),
        "experiment_ratio": float(
            sum(bool(EXPERIMENT_WORDS.search(message)) for message in commit_texts) / max(len(commit_texts), 1)
        ),
    }

    resolved_baseline = baseline_stats or DEFAULT_BASELINE_STATS
    percentiles = {
        feature_name: percentile_from_stats(raw_features[feature_name], resolved_baseline.get(feature_name, {"mean": 0.0, "std": 1.0}))
        for feature_name in FEATURE_NAMES
    }

    technical_depth_percentile = percentiles["technical_depth_score"]
    topics = []
    if extract_topics:
        topics = _extract_topics_with_bertopic([repo.get("readme", "") for repo in repos if repo.get("readme")]) if use_bertopic else []
        if not topics:
            topics = _extract_topics(raw_data)

    communication_style = _communication_style(combined_texts, percentiles["communication_directness"])
    communication_style["avg_sentiment"] = raw_features["avg_sentiment"]

    return {
        "raw_features": raw_features,
        "percentiles": percentiles,
        "technical_depth_percentile": technical_depth_percentile,
        "topics": topics,
        "communication_style": communication_style,
        "metadata": {
            "text_sample_count": len([text for text in combined_texts if text.strip()]),
            "technical_terms_matched": len(technical_matches),
            "total_tokens": len(all_tokens),
        },
    }
