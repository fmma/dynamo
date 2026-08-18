# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remove a trusted pull-request branch after its workflow runs materialize."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# This is the complete set of workflows triggered by a numeric pull-request branch.
# A missing run blocks deletion so a newly added workflow cannot be overlooked
# silently; the static test keeps this list and the workflow wiring aligned.
EXPECTED_WORKFLOWS = frozenset(
    {
        ".github/workflows/fern-docs.yml",
        ".github/workflows/pr-xpu.yaml",
        ".github/workflows/pr.yaml",
        ".github/workflows/request-nvskills-ci.yml",
    }
)
SUPPORTED_ACTIONS = frozenset({"requested", "in_progress"})
BRANCH_PATTERN = re.compile(r"^pull-request/[0-9]+$")
SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
PAGE_SIZE = 100


class GitHubApiError(RuntimeError):
    """An unsuccessful GitHub API request."""


@dataclass(frozen=True)
class MaterializationResult:
    """Whether every expected workflow run exists for one branch and SHA."""

    ready: bool
    missing_workflows: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowRunEvent:
    """Validated identity of the workflow run that woke the controller."""

    repository: str
    branch: str
    sha: str
    workflow_path: str
    action: str


@dataclass(frozen=True)
class RefOutcome:
    """Result of attempting one exact-SHA ref deletion."""

    branch: str
    expected_sha: str
    outcome: str


class GitHubClient:
    """GitHub REST client backed by the preinstalled ``gh`` command."""

    def __init__(self, repository: str, token: str) -> None:
        """Create a client for ``owner/repository`` using an in-memory token."""

        validate_repository(repository)
        if not token:
            raise ValueError("GH_TOKEN must be set")
        self.repository = repository
        self._token = token

    def _get_all(
        self,
        endpoint: str,
        item_key: str | None,
        parameters: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch and flatten every API page, optionally extracting ``item_key``."""

        # Without --slurp, gh writes one JSON document per page. Slurping makes
        # pagination a single JSON value that can be parsed without a temp file.
        command = [
            "gh",
            "api",
            "--method",
            "GET",
            "--paginate",
            "--slurp",
            endpoint,
            "--raw-field",
            f"per_page={PAGE_SIZE}",
        ]
        for key, value in sorted((parameters or {}).items()):
            command.extend(["--raw-field", f"{key}={value}"])

        # Keep the token out of argv and logs.
        environment = os.environ.copy()
        environment.pop("BRANCH_DELETE_TOKEN", None)
        environment["GH_TOKEN"] = self._token
        completed: subprocess.CompletedProcess[str] | None = None
        for attempt in range(3):
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    env=environment,
                    text=True,
                    timeout=60,
                )
            except subprocess.TimeoutExpired as error:
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise GitHubApiError(
                    f"GitHub API request timed out: {endpoint}"
                ) from error
            if completed.returncode == 0:
                break
            if attempt < 2:
                time.sleep(2**attempt)
        if completed is None or completed.returncode != 0:
            detail = completed.stderr.strip() if completed is not None else "no result"
            raise GitHubApiError(f"GitHub API request failed for {endpoint}: {detail}")

        pages = json.loads(completed.stdout)
        if not isinstance(pages, list):
            raise GitHubApiError(f"Unexpected response from {endpoint}")
        items: list[dict[str, Any]] = []
        for payload in pages:
            page_items = payload if item_key is None else payload[item_key]
            if not isinstance(page_items, list):
                raise GitHubApiError(f"Unexpected response from {endpoint}")
            items.extend(page_items)
        return items

    def get_ref_sha(self, branch: str) -> str | None:
        """Return the current SHA for an exact branch, or ``None`` if absent."""

        validate_branch(branch)
        full_ref = f"refs/heads/{branch}"
        # The matching-refs endpoint also returns longer refs sharing this prefix,
        # so accept only the exact fully qualified branch name.
        matches = self._get_all(
            f"/repos/{self.repository}/git/matching-refs/heads/{branch}", None
        )
        for ref in matches:
            if ref["ref"] == full_ref:
                return str(ref["object"]["sha"])
        return None

    def workflow_runs(self, branch: str, sha: str) -> list[dict[str, Any]]:
        """Return push workflow runs whose branch and head SHA match exactly."""

        validate_branch(branch)
        validate_sha(sha)
        return self._get_all(
            f"/repos/{self.repository}/actions/runs",
            "workflow_runs",
            {"branch": branch, "event": "push", "head_sha": sha},
        )


def validate_repository(repository: str) -> None:
    """Require GitHub's ``owner/repository`` name form."""

    parts = repository.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Invalid GitHub repository: {repository!r}")


def validate_branch(branch: str) -> None:
    """Require an unqualified numeric ``pull-request/N`` branch name."""

    if not BRANCH_PATTERN.fullmatch(branch):
        raise ValueError(f"Invalid pull request branch: {branch!r}")


def validate_sha(sha: str) -> None:
    """Require a full 40-character hexadecimal commit SHA."""

    if not SHA_PATTERN.fullmatch(sha):
        raise ValueError(f"Invalid commit SHA: {sha!r}")


def validate_workflow_run_event(
    payload: Mapping[str, Any], repository: str
) -> WorkflowRunEvent:
    """Validate a ``workflow_run`` payload before trusting its ref identity."""

    validate_repository(repository)
    payload_repository = payload.get("repository")
    if not isinstance(payload_repository, dict):
        raise ValueError("Workflow event is missing its repository")
    if payload_repository.get("full_name") != repository:
        raise ValueError("Workflow event repository does not match GITHUB_REPOSITORY")

    action = str(payload.get("action", ""))
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"Unsupported workflow_run action: {action!r}")

    workflow_run = payload.get("workflow_run")
    if not isinstance(workflow_run, dict):
        raise ValueError("Workflow event is missing workflow_run")
    if workflow_run.get("event") != "push":
        raise ValueError("Only push workflow runs may request branch cleanup")

    head_repository = workflow_run.get("head_repository")
    if not isinstance(head_repository, dict):
        raise ValueError("Workflow run is missing its head repository")
    if head_repository.get("full_name") != repository:
        raise ValueError("Workflow run head repository is not this repository")

    branch = str(workflow_run.get("head_branch", ""))
    sha = str(workflow_run.get("head_sha", ""))
    workflow_path = str(workflow_run.get("path", ""))
    validate_branch(branch)
    validate_sha(sha)
    if workflow_path not in EXPECTED_WORKFLOWS:
        raise ValueError(f"Unexpected triggering workflow: {workflow_path!r}")

    return WorkflowRunEvent(repository, branch, sha, workflow_path, action)


def load_workflow_run_event(event_path: Path, repository: str) -> WorkflowRunEvent:
    """Load and validate the GitHub event JSON from ``GITHUB_EVENT_PATH``."""

    with event_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Workflow event must be a JSON object")
    return validate_workflow_run_event(payload, repository)


def materialization_status(
    client: GitHubClient, branch: str, sha: str
) -> MaterializationResult:
    """Return whether all expected push workflow runs have materialized.

    A queued run is sufficient. GitHub records and reruns a workflow against its
    immutable head SHA, so the controller does not need to wait for jobs or
    checkouts to finish before removing the branch pointer.
    """

    matching_paths: set[str] = set()
    for run in client.workflow_runs(branch, sha):
        path = run.get("path")
        if (
            path not in EXPECTED_WORKFLOWS
            or run.get("event") != "push"
            or run.get("head_branch") != branch
            or str(run.get("head_sha", "")).lower() != sha.lower()
        ):
            continue
        matching_paths.add(str(path))

    missing_workflows = tuple(sorted(EXPECTED_WORKFLOWS - matching_paths))
    return MaterializationResult(
        ready=not missing_workflows,
        missing_workflows=missing_workflows,
    )


def build_push_command(
    repository: str, git_directory: Path, expected_refs: Mapping[str, str]
) -> list[str]:
    """Build a deletion push guarded by each ref's exact expected SHA."""

    validate_repository(repository)
    if not expected_refs:
        raise ValueError("At least one ref is required")
    command = [
        "git",
        "-C",
        str(git_directory),
        "-c",
        "credential.helper=",
        "-c",
        "core.hooksPath=/dev/null",
        "push",
        "--porcelain",
    ]
    for branch, sha in sorted(expected_refs.items()):
        validate_branch(branch)
        validate_sha(sha)
        command.append(f"--force-with-lease=refs/heads/{branch}:{sha}")
    command.append(f"https://github.com/{repository}.git")
    command.extend(f":refs/heads/{branch}" for branch in sorted(expected_refs))
    return command


def push_ref_deletions(repository: str, expected_refs: Mapping[str, str]) -> int:
    """Push lease-guarded deletions without placing the token in argv.

    The return code is intentionally returned rather than raised so callers can
    reread every ref and distinguish a successful peer deletion or moved branch
    from an authorization or transport failure.
    """

    token = os.environ.get("BRANCH_DELETE_TOKEN", "")
    if not token:
        raise ValueError("BRANCH_DELETE_TOKEN must be set")

    with tempfile.TemporaryDirectory(prefix="dynamo-pr-branch-cleanup-") as temp_dir:
        temp_path = Path(temp_dir)
        git_directory = temp_path / "repository.git"
        init_environment = os.environ.copy()
        init_environment.pop("BRANCH_DELETE_TOKEN", None)
        init_environment.pop("GH_TOKEN", None)
        subprocess.run(
            ["git", "init", "--bare", str(git_directory)],
            check=True,
            capture_output=True,
            env=init_environment,
            text=True,
        )

        # An askpass helper reads the short-lived App token only at authentication
        # time. The token never enters the remote URL, argv, or git configuration.
        askpass_path = temp_path / "git-askpass.sh"
        with askpass_path.open("w", encoding="utf-8") as stream:
            stream.write(
                "#!/bin/sh\n"
                'case "$1" in\n'
                "  *sername*) printf '%s\\n' x-access-token ;;\n"
                "  *assword*) printf '%s\\n' \"$BRANCH_DELETE_TOKEN\" ;;\n"
                "  *) exit 1 ;;\n"
                "esac\n"
            )
        askpass_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

        environment = os.environ.copy()
        environment.pop("GH_TOKEN", None)
        environment.update(
            {
                "GIT_ASKPASS": str(askpass_path),
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
        completed = subprocess.run(
            build_push_command(repository, git_directory, expected_refs),
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )
        return completed.returncode


def delete_refs(
    client: GitHubClient, repository: str, expected_refs: Mapping[str, str]
) -> list[RefOutcome]:
    """Delete refs that still match their expected SHA.

    Refs that are already absent or have moved are successful no-ops. A failed
    push is followed by a fresh API read so partial batch success and concurrent
    controller runs are reported accurately.
    """

    outcomes: list[RefOutcome] = []
    eligible: dict[str, str] = {}
    for branch, expected_sha in sorted(expected_refs.items()):
        validate_branch(branch)
        validate_sha(expected_sha)
        current_sha = client.get_ref_sha(branch)
        if current_sha is None:
            outcomes.append(RefOutcome(branch, expected_sha, "already-missing"))
        elif current_sha.lower() != expected_sha.lower():
            outcomes.append(RefOutcome(branch, expected_sha, "moved"))
        else:
            eligible[branch] = expected_sha

    if not eligible:
        return outcomes

    return_code = push_ref_deletions(repository, eligible)
    if return_code == 0:
        outcomes.extend(
            RefOutcome(branch, expected_sha, "deleted")
            for branch, expected_sha in sorted(eligible.items())
        )
        return outcomes

    for branch, expected_sha in sorted(eligible.items()):
        current_sha = client.get_ref_sha(branch)
        if current_sha is None:
            outcome = "deleted"
        elif current_sha.lower() != expected_sha.lower():
            outcome = "moved"
        else:
            outcome = "deletion-failed"
        outcomes.append(RefOutcome(branch, expected_sha, outcome))
    return outcomes


def append_summary(lines: Sequence[str]) -> None:
    """Append lines to the Actions step summary when running in Actions."""

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with Path(summary_path).open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        stream.write("\n")


def emit_error(message: str) -> None:
    """Emit an escaped GitHub Actions error annotation."""

    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error title=Pull request branch cleanup::{escaped}")


def cleanup_branch(client: GitHubClient, repository: str, branch: str, sha: str) -> int:
    """Remove one unchanged branch after all expected runs materialize.

    Deferral, an absent ref, and a ref moved to another SHA all exit successfully.
    Authorization or transport failures fail the controller without changing the
    conclusions of the independently running workload workflows.
    """

    validate_branch(branch)
    validate_sha(sha)

    current_sha = client.get_ref_sha(branch)
    if current_sha is None:
        append_summary(
            ["## Pull request branch cleanup", "", "The branch is already absent."]
        )
        return 0
    if current_sha.lower() != sha.lower():
        # A newer push owns a moved branch; an older workflow must never remove it.
        append_summary(
            [
                "## Pull request branch cleanup",
                "",
                "The branch moved to another commit; this run left it unchanged.",
            ]
        )
        return 0

    materialization = materialization_status(client, branch, sha)
    if not materialization.ready:
        # Every expected workflow emits requested/in-progress events. Early
        # controller runs defer; a later event observes the complete run set.
        details = [
            *(
                f"Missing workflow: `{path}`"
                for path in materialization.missing_workflows
            ),
        ]
        append_summary(
            [
                "## Pull request branch cleanup",
                "",
                "Cleanup was deferred until the other trusted workflows materialize.",
                "",
                *details,
            ]
        )
        return 0

    outcomes = delete_refs(client, repository, {branch: sha})
    outcome = outcomes[0]
    if outcome.outcome == "deletion-failed":
        message = (
            f"Could not delete {branch} at {sha}; the ref still points to the "
            "expected commit. Check token and ruleset permissions."
        )
        emit_error(message)
        append_summary(["## Pull request branch cleanup", "", f"Error: {message}"])
        return 1

    append_summary(
        [
            "## Pull request branch cleanup",
            "",
            f"Result for `{branch}`: `{outcome.outcome}`.",
        ]
    )
    return 0


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse controller command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY"),
        required=os.environ.get("GITHUB_REPOSITORY") is None,
    )
    parser.add_argument(
        "--event-path",
        type=Path,
        default=os.environ.get("GITHUB_EVENT_PATH"),
        required=os.environ.get("GITHUB_EVENT_PATH") is None,
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Validate one workflow event and clean up its pull-request branch."""

    args = parse_args(arguments)
    event = load_workflow_run_event(args.event_path, args.repository)
    client = GitHubClient(args.repository, os.environ.get("GH_TOKEN", ""))
    return cleanup_branch(client, event.repository, event.branch, event.sha)


if __name__ == "__main__":
    raise SystemExit(main())
