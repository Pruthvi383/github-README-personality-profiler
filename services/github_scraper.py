from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar

from github import Auth, Github
from github.GithubException import GithubException, RateLimitExceededException, UnknownObjectException


F = TypeVar("F", bound=Callable[..., Any])


class GitHubScraperError(Exception):
    """Raised when GitHub data cannot be fetched safely."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_error: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RateLimitExceededException as exc:
                    last_error = exc
                except GithubException as exc:
                    if exc.status not in {403, 429, 500, 502, 503, 504}:
                        raise
                    last_error = exc

                if attempt == max_retries - 1:
                    break
                time.sleep(delay)
                delay *= 2

            if last_error is not None:
                raise last_error
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def _github_client() -> Github:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise GitHubScraperError(
            "Missing GitHub token. Add GITHUB_TOKEN to your .env file before analyzing profiles.",
            status_code=503,
        )
    return Github(auth=Auth.Token(token), per_page=50, timeout=15)


def _to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _is_bot_login(login: Optional[str]) -> bool:
    lowered = (login or "").lower()
    return lowered.endswith("[bot]") or "bot" in lowered


def _clean_commit_message(message: str) -> str:
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    filtered: List[str] = []
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("co-authored-by:"):
            continue
        if lowered.startswith("signed-off-by:"):
            continue
        if lowered.startswith("merge pull request"):
            continue
        if lowered.startswith("merge branch"):
            continue
        if lowered.startswith("merge remote-tracking branch"):
            continue
        filtered.append(line)
    if not filtered:
        return ""
    cleaned = " ".join(filtered).strip()
    if cleaned.lower().startswith(("bump ", "chore(deps", "build(deps", "dependabot")):
        return ""
    return cleaned


@retry_with_backoff()
def _get_user(client: Github, username: str) -> Any:
    return client.get_user(username)


@retry_with_backoff()
def _iter_repos(user: Any) -> Iterable[Any]:
    return user.get_repos(type="owner", sort="updated", direction="desc")


@retry_with_backoff()
def _get_readme(repo: Any) -> Optional[str]:
    try:
        readme = repo.get_readme()
    except UnknownObjectException:
        return None
    return readme.decoded_content.decode("utf-8", errors="ignore")


@retry_with_backoff()
def _get_topics(repo: Any) -> List[str]:
    try:
        topics = repo.get_topics()
    except GithubException:
        return []
    return list(topics or [])


@retry_with_backoff()
def _get_commits(repo: Any, author_login: str) -> Iterable[Any]:
    return repo.get_commits(author=author_login)


@retry_with_backoff()
def _search_issues(client: Github, query: str) -> Iterable[Any]:
    return client.search_issues(query=query, sort="updated", order="desc")


@retry_with_backoff()
def _get_issue_comments(issue: Any) -> Iterable[Any]:
    return issue.get_comments()


def _sample_recent_comments(issue: Any, sample_size: int = 12) -> List[Any]:
    comments = _get_issue_comments(issue)
    try:
        total_count = int(getattr(comments, "totalCount", 0))
        if total_count > 0:
            start_index = max(total_count - sample_size, 0)
            return list(comments[start_index:total_count])
    except Exception:
        pass

    sampled: List[Any] = []
    for comment in comments:
        sampled.append(comment)
        if len(sampled) >= sample_size:
            break
    return sampled


def _build_repo_snapshot(repo: Any) -> tuple[Dict[str, Any], Optional[str]]:
    readme_error: Optional[str] = None
    try:
        readme_content = _get_readme(repo)
    except GithubException as exc:
        readme_content = None
        readme_error = f"README unavailable for {repo.full_name}: {exc.data if hasattr(exc, 'data') else exc}"

    try:
        topics = _get_topics(repo)
    except GithubException:
        topics = []

    return (
        {
            "name": repo.name,
            "full_name": repo.full_name,
            "html_url": repo.html_url,
            "description": repo.description or "",
            "topics": topics,
            "language": repo.language or "Unknown",
            "stars": int(repo.stargazers_count or 0),
            "size_kb": int(repo.size or 0),
            "updated_at": _to_iso(repo.updated_at),
            "created_at": _to_iso(repo.created_at),
            "archived": bool(repo.archived),
            "fork": bool(repo.fork),
            "readme": readme_content or "",
        },
        readme_error,
    )


def fetch_github_profile(username: str) -> Dict[str, Any]:
    client = _github_client()
    normalized_username = username.strip()
    if not normalized_username:
        raise GitHubScraperError("Please provide a GitHub username.", status_code=400)

    try:
        user = _get_user(client, normalized_username)
    except UnknownObjectException as exc:
        raise GitHubScraperError(f"GitHub user '{normalized_username}' was not found.", status_code=404) from exc
    except RateLimitExceededException as exc:
        raise GitHubScraperError("GitHub API rate limit exceeded. Please try again in a few minutes.", 503) from exc
    except GithubException as exc:
        raise GitHubScraperError("GitHub API is unavailable right now. Please try again shortly.", 503) from exc

    repos: List[Dict[str, Any]] = []
    commit_messages: List[Dict[str, Any]] = []
    issue_comments: List[Dict[str, Any]] = []
    pull_requests: List[Dict[str, Any]] = []
    errors: List[str] = []

    try:
        repo_candidates = [repo for repo in _iter_repos(user) if not repo.private][:30]
    except GithubException as exc:
        raise GitHubScraperError("Unable to fetch repositories from GitHub right now.", 503) from exc

    if not repo_candidates:
        raise GitHubScraperError(
            f"GitHub user '{normalized_username}' has no public repositories to analyze.",
            status_code=404,
        )

    repo_by_name = {repo.full_name: repo for repo in repo_candidates}

    with ThreadPoolExecutor(max_workers=min(8, len(repo_candidates))) as pool:
        futures = {pool.submit(_build_repo_snapshot, repo): repo.full_name for repo in repo_candidates}
        collected_by_name: Dict[str, Dict[str, Any]] = {}
        for future in as_completed(futures):
            repo_name = futures[future]
            try:
                repo_snapshot, repo_error = future.result()
            except Exception as exc:
                repo_snapshot = {
                    "name": repo_by_name[repo_name].name,
                    "full_name": repo_name,
                    "html_url": repo_by_name[repo_name].html_url,
                    "description": repo_by_name[repo_name].description or "",
                    "topics": [],
                    "language": repo_by_name[repo_name].language or "Unknown",
                    "stars": int(repo_by_name[repo_name].stargazers_count or 0),
                    "size_kb": int(repo_by_name[repo_name].size or 0),
                    "updated_at": _to_iso(repo_by_name[repo_name].updated_at),
                    "created_at": _to_iso(repo_by_name[repo_name].created_at),
                    "archived": bool(repo_by_name[repo_name].archived),
                    "fork": bool(repo_by_name[repo_name].fork),
                    "readme": "",
                }
                repo_error = f"Repository metadata partially unavailable for {repo_name}: {exc}"

            collected_by_name[repo_name] = repo_snapshot
            if repo_error:
                errors.append(repo_error)

    repos = [collected_by_name[repo.full_name] for repo in repo_candidates if repo.full_name in collected_by_name]

    commit_deadline = time.monotonic() + 12
    for repo in repo_candidates:
        if time.monotonic() > commit_deadline:
            errors.append("Commit collection timed out; continuing with partial history.")
            break

        if len(commit_messages) >= 100:
            continue

        try:
            for commit in _get_commits(repo, user.login):
                author_login = None
                if commit.author is not None:
                    author_login = getattr(commit.author, "login", None)
                if _is_bot_login(author_login):
                    continue

                cleaned_message = _clean_commit_message(commit.commit.message or "")
                if not cleaned_message:
                    continue

                commit_messages.append(
                    {
                        "repo": repo.full_name,
                        "message": cleaned_message,
                        "committed_at": _to_iso(commit.commit.author.date if commit.commit.author else None),
                        "sha": commit.sha,
                    }
                )
                if len(commit_messages) >= 100:
                    break
        except GithubException as exc:
            errors.append(f"Commit history unavailable for {repo.full_name}: {exc}")

    issue_search_query = f"commenter:{user.login} type:issue"
    issue_deadline = time.monotonic() + 8
    inspected_issue_threads = 0
    try:
        for issue in _search_issues(client, issue_search_query):
            if time.monotonic() > issue_deadline:
                errors.append("Issue comment collection timed out; continuing with partial data.")
                break
            if len(issue_comments) >= 50:
                break
            if inspected_issue_threads >= 12:
                errors.append("Issue comment collection capped after scanning 12 recent threads.")
                break
            inspected_issue_threads += 1
            try:
                comments = _sample_recent_comments(issue)
            except GithubException:
                continue

            for comment in reversed(comments):
                issue_user = getattr(comment.user, "login", "") if getattr(comment, "user", None) else ""
                if issue_user.lower() != user.login.lower():
                    continue
                issue_comments.append(
                    {
                        "repo": getattr(issue.repository, "full_name", ""),
                        "issue_number": getattr(issue, "number", None),
                        "body": comment.body or "",
                        "created_at": _to_iso(comment.created_at),
                        "html_url": comment.html_url,
                    }
                )
                if len(issue_comments) >= 50:
                    break
    except GithubException as exc:
        errors.append(f"Issue comments unavailable: {exc}")

    pr_search_query = f"author:{user.login} type:pr"
    pr_deadline = time.monotonic() + 6
    try:
        for pull_request in _search_issues(client, pr_search_query):
            if time.monotonic() > pr_deadline:
                errors.append("Pull request collection timed out; continuing with partial data.")
                break
            if len(pull_requests) >= 30:
                break
            pull_requests.append(
                {
                    "repo": getattr(pull_request.repository, "full_name", ""),
                    "number": getattr(pull_request, "number", None),
                    "title": pull_request.title or "",
                    "body": pull_request.body or "",
                    "created_at": _to_iso(pull_request.created_at),
                    "html_url": pull_request.html_url,
                }
            )
    except GithubException as exc:
        errors.append(f"Pull requests unavailable: {exc}")

    stale_cutoff = datetime.now(timezone.utc) - timedelta(days=365)
    churn_like_repos = 0
    for repo in repos:
        updated_at = repo.get("updated_at")
        updated_dt = datetime.fromisoformat(updated_at) if updated_at else None
        if repo["archived"] or (
            repo["stars"] == 0 and updated_dt is not None and updated_dt < stale_cutoff
        ):
            churn_like_repos += 1

    return {
        "user": {
            "login": user.login,
            "name": user.name or user.login,
            "avatar_url": user.avatar_url,
            "bio": user.bio or "",
            "followers": int(user.followers or 0),
            "public_repos": int(user.public_repos or 0),
            "html_url": user.html_url,
        },
        "repos": repos,
        "commit_messages": commit_messages[:100],
        "issue_comments": issue_comments[:50],
        "pull_requests": pull_requests[:30],
        "stats": {
            "repo_count": len(repos),
            "commit_count": len(commit_messages[:100]),
            "issue_comment_count": len(issue_comments[:50]),
            "pull_request_count": len(pull_requests[:30]),
            "churn_like_repo_count": churn_like_repos,
        },
        "raw_text": {
            "readmes": [repo["readme"] for repo in repos if repo["readme"]],
            "descriptions": [repo["description"] for repo in repos if repo["description"]],
            "commits": [item["message"] for item in commit_messages[:100]],
            "issue_comments": [item["body"] for item in issue_comments[:50] if item["body"]],
            "pull_requests": [
                " ".join(part for part in [item["title"], item["body"]] if part).strip()
                for item in pull_requests[:30]
                if item["title"] or item["body"]
            ],
        },
        "errors": errors,
    }
