"""Collect and verify the Assignment 4 holdout using public GitHub release metadata."""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


CANDIDATES = """
astral-sh/ruff astral-sh/uv python-poetry/poetry pypa/pip pypa/build
pypa/twine pypa/hatch pypa/virtualenv tox-dev/tox pre-commit/pre-commit
dask/dask dask/distributed pydata/xarray scikit-image/scikit-image bokeh/bokeh
plotly/plotly.py vega/altair mwaskom/seaborn joblib/joblib statsmodels/statsmodels
aio-libs/aiohttp encode/httpx encode/httpcore django/django encode/django-rest-framework
scrapy/scrapy pallets/jinja pallets/click celery/celery redis/redis-py
psf/black sphinx-doc/sphinx mkdocs/mkdocs mkdocstrings/mkdocstrings dbt-labs/dbt-core
prefecthq/prefect streamlit/streamlit gradio-app/gradio jupyterlab/jupyterlab jupyter/notebook
""".split()

PILOT = set("""
pallets/flask psf/requests numpy/numpy scipy/scipy pandas-dev/pandas
scikit-learn/scikit-learn python-pillow/pillow matplotlib/matplotlib
pytest-dev/pytest nltk/nltk encode/uvicorn encode/starlette kludex/uvicorn kludex/starlette
pydantic/pydantic fastapi/fastapi fastapi/typer huggingface/transformers
huggingface/tokenizers ukplab/sentence-transformers
pallets/werkzeug textualize/rich
""".split())

CUTOFFS = {"frozen": "2025-01-01T00:00:00Z", "updated": "2025-06-01T00:00:00Z"}
VERSION = "github-releases-holdout-v1"
FIELDS = ("id", "tag_name", "published_at", "html_url", "draft", "prerelease")
NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')
TAG = re.compile(r"(?<![\w.])v?\d+\.\d+\.\d+(?:[-+][0-9a-zA-Z.-]+)?(?!\w|\.\d)")


class RateLimit(Exception):
    pass


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(json_bytes(value))
    tmp.replace(path)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def first_url(repository):
    return f"https://api.github.com/repos/{repository}/releases?per_page=100"


def utc(timestamp):
    value = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if value.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError(f"Expected UTC timestamp: {timestamp}")
    return value


def github_host(url, expected):
    info = urlsplit(url)
    return info.scheme == "https" and info.netloc.lower() == expected


def fetch_page(url):
    if not github_host(url, "api.github.com"):
        raise ValueError(f"Unexpected pagination URL: {url}")
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Jarvis-Team-Assignment4-Release-Study",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    for attempt in range(2):
        try:
            with urlopen(request, timeout=18) as response:
                body = response.read()
                final_url = response.geturl()
                headers = response.headers
            records = json.loads(body)
            if not isinstance(records, list) or not github_host(final_url, "api.github.com"):
                raise ValueError("Unexpected GitHub API response")
            match = NEXT.search(headers.get("Link", ""))
            return {
                "url": url,
                "response_url": final_url,
                "status": 200,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "response_sha256": digest(body),
                "next": match.group(1) if match else None,
                "records": [{field: record[field] for field in FIELDS} for record in records],
            }
        except HTTPError as error:
            if error.code == 429 or (
                error.code == 403 and (
                    error.headers.get("X-RateLimit-Remaining") == "0"
                    or error.headers.get("Retry-After")
                )
            ):
                reset = error.headers.get("X-RateLimit-Reset")
                reset_at = datetime.fromtimestamp(int(reset), timezone.utc) if reset else None
                raise RateLimit(f"HTTP {error.code}; rate-limit reset UTC: {reset_at}") from None
            if error.code == 404 and "json" in error.headers.get("Content-Type", ""):
                body = error.read()
                if json.loads(body).get("message") == "Not Found":
                    return {
                        "url": url,
                        "response_url": error.geturl(),
                        "status": 404,
                        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                        "response_sha256": digest(body),
                        "next": None,
                        "records": [],
                    }
            if error.code not in (500, 502, 503, 504) or attempt == 1:
                raise
        except (URLError, TimeoutError):
            if attempt == 1:
                raise
        time.sleep(2)
    raise RuntimeError("GitHub request did not complete")


def archive_path(root, repository):
    return root / "release_metadata" / f"{repository.replace('/', '__')}.json"


def collect_repository(root, repository):
    path = archive_path(root, repository)
    archive = read_json(path) if path.exists() else {"candidate": repository, "pages": []}
    if archive.get("provider") == "github_connector":
        if complete_archive(root, repository) is None:
            raise ValueError(f"Incomplete connector archive for {repository}")
        return
    pages = archive["pages"]
    url = pages[-1]["next"] if pages else first_url(repository)
    seen = {page["url"] for page in pages}
    while url:
        if url in seen:
            raise ValueError(f"Pagination loop for {repository}")
        page = fetch_page(url)
        if page["status"] == 404 and pages:
            raise ValueError(f"GitHub returned 404 partway through {repository}")
        pages.append(page)
        write_json(path, archive)
        seen.add(url)
        url = page["next"]


def collect(root):
    failures = {}
    consecutive_errors = 0
    for repository in CANDIDATES:
        try:
            collect_repository(root, repository)
        except RateLimit as error:
            failures[repository] = str(error)
            print(f"Paused at {repository}: {error}", flush=True)
            break
        except (OSError, ValueError, KeyError) as error:
            failures[repository] = str(error)
            consecutive_errors += 1
            print(f"Unresolved {repository}: {error}", flush=True)
            if consecutive_errors >= 3:
                break
        else:
            consecutive_errors = 0
            print(f"Archived {repository}", flush=True)
    if failures:
        write_json(root / "unresolved_requests.json", failures)


def complete_archive(root, repository):
    path = archive_path(root, repository)
    if not path.exists():
        return None
    archive = read_json(path)
    pages = archive["pages"]
    connector = archive.get("provider") == "github_connector"
    if not pages or (not pages[-1]["end"] if connector else pages[-1]["next"] is not None):
        return None
    endpoint_repo = repository
    redirect = archive.get("redirect")
    if connector and redirect:
        endpoint_repo = redirect["canonical"]
        if (
            not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", endpoint_repo)
            or redirect["source_url"] != f"https://api.github.com/repos/{repository}"
            or type(redirect["repository_id"]) is not int
            or redirect["redirect_url"] != f"https://api.github.com/repositories/{redirect['repository_id']}"
        ):
            raise ValueError(f"Invalid repository redirect evidence for {repository}")
    expected = first_url(repository)
    ids = set()
    for index, page in enumerate(pages):
        records = page["records"]
        if connector:
            size = page["size"]
            if type(size) is not int or not 1 <= size <= 100 or len(ids) % size:
                raise ValueError(f"Invalid page size or offset for {repository}")
            page_number = len(ids) // size + 1
            expected = f"https://api.github.com/repos/{endpoint_repo}/releases?per_page={size}&page={page_number}"
            if (
                page["page"] != page_number
                or len(records) > size
                or page["end"] != (len(records) < size)
                or page["end"] != (index == len(pages) - 1)
                or page["metadata_sha256"] != digest(json_bytes(records))
            ):
                raise ValueError(f"Invalid connector page for {repository}")
        if page["url"] != expected or not github_host(page["response_url"], "api.github.com"):
            raise ValueError(f"Broken page chain for {repository}")
        if not connector:
            expected = page["next"]
            if not re.fullmatch(r"[0-9a-f]{64}", page["response_sha256"]):
                raise ValueError(f"Missing response hash for {repository}")
        utc(page["retrieved_at_utc"])
        if page["status"] == 404 and len(pages) == 1 and not page["records"]:
            continue
        if page["status"] != 200:
            raise ValueError(f"Non-success response archived for {repository}")
        for release in records:
            if type(release["id"]) is not int or release["id"] in ids:
                raise ValueError(f"Invalid or repeated release ID for {repository}")
            if type(release["draft"]) is not bool or type(release["prerelease"]) is not bool:
                raise ValueError(f"Missing release flags for {repository}")
            if not github_host(release["html_url"], "github.com"):
                raise ValueError(f"Invalid release URL for {repository}")
            if release["published_at"]:
                utc(release["published_at"])
            ids.add(release["id"])
    if archive["candidate"] != repository:
        raise ValueError(f"Repository mismatch in {path.name}")
    return pages


def normalize(tag):
    return tag.strip().lower().removeprefix("v")


def supported_tag(tag):
    # Matches extract_tag in Assignment 3's analyze_results.py.
    return {normalize(match) for match in TAG.findall(tag)} == {normalize(tag)}


def canonical_repo(release):
    owner, repository, section, *_ = urlsplit(release["html_url"]).path.strip("/").split("/")
    if section != "releases":
        raise ValueError(f"Expected release URL, received {release['html_url']}")
    return f"{owner}/{repository}"


def latest_release(releases, cutoff):
    boundary = utc(cutoff)
    eligible = (
        release for release in releases
        if not release["draft"] and not release["prerelease"] and release["published_at"]
        and utc(release["published_at"]) <= boundary
    )
    return max(eligible, key=lambda release: (utc(release["published_at"]), release["id"]), default=None)


def snapshot_record(repository, release, cutoff):
    tag = release["tag_name"]
    published = release["published_at"]
    source = release["html_url"]
    return {
        "document_id": repository.replace("/", "__"),
        "repository": repository,
        "release_id": release["id"],
        "tag": tag,
        "published_at": published,
        "source_url": source,
        "snapshot_cutoff": cutoff,
        "text": (
            f"Repository: {repository}\nRelease tag: {tag}\nPublished at: {published}\n"
            f"Status: non-prerelease, non-draft\nSource: {source}"
        ),
    }


def build(root):
    archives = {repo: complete_archive(root, repo) for repo in CANDIDATES}
    unresolved = [repo for repo, pages in archives.items() if pages is None]
    if unresolved:
        raise ValueError(
            f"Complete source histories: {len(CANDIDATES) - len(unresolved)}/{len(CANDIDATES)}. "
            "Resume the collection before freezing the holdout."
        )
    questions, answers, ledger = [], [], []
    snapshots = {state: [] for state in CUTOFFS}
    canonical_seen = set()
    for candidate, pages in archives.items():
        releases = [record for page in pages for record in page["records"]]
        row = {"candidate": candidate, "pages": len(pages), "release_count": len(releases)}
        ledger.append(row)
        if pages[0]["status"] == 404:
            row["reason"] = "GitHub REST API returned 404 Not Found for the public release list."
            continue
        selected = {state: latest_release(releases, cutoff) for state, cutoff in CUTOFFS.items()}
        if any(release is None for release in selected.values()):
            row["reason"] = "No eligible published release at one or both cutoffs."
            continue
        old, new = selected.values()
        repository = canonical_repo(new)
        aliases = {candidate.lower(), *(canonical_repo(release).lower() for release in releases)}
        row["canonical_repository"] = repository
        row["selected_release_ids"] = {state: release["id"] for state, release in selected.items()}
        if aliases & PILOT:
            row["reason"] = "Repository or alias appeared in Assignment 3."
            continue
        if aliases & canonical_seen:
            row["reason"] = "Duplicate repository or alias in this candidate list."
            continue
        if not all(supported_tag(release["tag_name"]) for release in selected.values()):
            row["reason"] = "One or both selected release tags are incompatible with the Assignment 3 scorer."
            continue
        canonical_seen.update(aliases)
        question_id = f"A4Q{len(questions) + 1:03}"
        row["question_id"] = question_id
        questions.append({
            "id": question_id,
            "repository": repository,
            "split": "evaluation",
            "answer_date": CUTOFFS["updated"],
            "question": (
                f"For {repository}, what is the tag of the most recently published non-prerelease, "
                "non-draft release in the archived GitHub release records as of 2025-06-01 00:00 UTC? "
                "Return only its release tag."
            ),
        })
        answers.append({
            "id": question_id,
            "answer_tag": new["tag_name"],
            "old_tag": old["tag_name"],
            "changed": normalize(old["tag_name"]) != normalize(new["tag_name"]),
            "source_url": new["html_url"],
            "release_published_at": new["published_at"],
            "release_id": new["id"],
        })
        for state, cutoff in CUTOFFS.items():
            snapshots[state].append(snapshot_record(repository, selected[state], cutoff))

    if not questions:
        raise ValueError("No eligible questions. A final dataset cannot be created.")
    payloads = {
        "questions.json": questions,
        "answer_key.json": answers,
        "kb_frozen.json": snapshots["frozen"],
        "kb_updated.json": snapshots["updated"],
        "collection_ledger.json": ledger,
    }
    changed = sum(answer["changed"] for answer in answers)
    payloads["dataset_manifest.json"] = {
        "version": VERSION,
        "source": "https://docs.github.com/en/rest/releases/releases#list-releases",
        "cutoffs_utc": CUTOFFS,
        "candidates": len(CANDIDATES),
        "included_questions": len(questions),
        "excluded_candidates": len(CANDIDATES) - len(questions),
        "changed_tags": changed,
        "unchanged_tags": len(questions) - changed,
        "sha256": {name: digest(json_bytes(value)) for name, value in payloads.items()},
        "source_archive_sha256": {
            archive_path(root, repo).name: digest(archive_path(root, repo).read_bytes())
            for repo in CANDIDATES
        },
    }
    return payloads


def verify(root, payloads):
    for name, value in payloads.items():
        if (root / name).read_bytes() != json_bytes(value):
            raise ValueError(f"{name} does not match reconstruction from the source archives")
    manifest = payloads["dataset_manifest.json"]
    print(f"Verified {manifest['included_questions']} questions from {manifest['candidates']} candidates.")
    print(f"Changed: {manifest['changed_tags']}; unchanged: {manifest['unchanged_tags']}.")
    for name in payloads:
        print(f"{digest((root / name).read_bytes())}  {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for archived pages and test files")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="Rebuild in memory and verify offline")
    mode.add_argument("--build-from-archive", action="store_true", help="Create dataset files from archived release pages")
    args = parser.parse_args()
    try:
        if not args.verify and not args.build_from_archive:
            collect(args.output)
        payloads = build(args.output)
        if not args.verify:
            for name, value in payloads.items():
                write_json(args.output / name, value)
        verify(args.output, payloads)
    except (OSError, ValueError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
