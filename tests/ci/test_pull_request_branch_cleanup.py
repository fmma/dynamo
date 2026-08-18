# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit and static-contract tests for pull-request branch cleanup."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".github/scripts/cleanup_pull_request_branches.py"
WORKFLOW_DIR = REPO_ROOT / ".github/workflows"
CONTROLLER_PATH = WORKFLOW_DIR / "cleanup-pull-request-branches.yml"

# Pytest requires a list when applying multiple module-level marks.
pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def load_cleanup_module():
    """Load the standalone Actions helper as an importable test module."""

    spec = importlib.util.spec_from_file_location(
        "cleanup_pull_request_branches", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cleanup_module():
    """Return a fresh cleanup helper module for each test."""

    return load_cleanup_module()


def workflow_run(path, sha, branch="pull-request/123"):
    """Build one minimal Actions API workflow-run record."""

    return {
        "event": "push",
        "head_branch": branch,
        "head_sha": sha,
        "path": path,
        "status": "queued",
    }


def workflow_event(module, action="requested"):
    """Build one valid ``workflow_run`` webhook payload."""

    return {
        "action": action,
        "repository": {"full_name": "ai-dynamo/dynamo"},
        "workflow_run": {
            "event": "push",
            "head_branch": "pull-request/123",
            "head_repository": {"full_name": "ai-dynamo/dynamo"},
            "head_sha": "a" * 40,
            "path": sorted(module.EXPECTED_WORKFLOWS)[0],
        },
    }


class MaterializationClient:
    """Minimal fake exposing workflow runs to the controller."""

    def __init__(self, runs):
        self.runs = runs

    def workflow_runs(self, branch, sha):
        return self.runs


class RefClient:
    """Minimal fake returning a prescribed sequence of ref SHAs."""

    def __init__(self, shas):
        self.shas = iter(shas)

    def get_ref_sha(self, branch):
        return next(self.shas)


class CleanupClient(MaterializationClient):
    """Fake exposing both run materialization and ref reads."""

    def __init__(self, runs, shas):
        super().__init__(runs)
        self.shas = iter(shas)

    def get_ref_sha(self, branch):
        return next(self.shas)


def all_expected_runs(module, sha):
    """Build one queued run for every expected workflow path."""

    return [workflow_run(path, sha) for path in sorted(module.EXPECTED_WORKFLOWS)]


def test_materialization_accepts_queued_runs(cleanup_module):
    sha = "a" * 40

    result = cleanup_module.materialization_status(
        MaterializationClient(all_expected_runs(cleanup_module, sha)),
        "pull-request/123",
        sha,
    )

    assert result.ready
    assert result.missing_workflows == ()


def test_materialization_defers_for_missing_workflow(cleanup_module):
    sha = "b" * 40
    runs = all_expected_runs(cleanup_module, sha)[:-1]

    result = cleanup_module.materialization_status(
        MaterializationClient(runs), "pull-request/123", sha
    )

    assert not result.ready
    assert len(result.missing_workflows) == 1


def test_materialization_ignores_runs_for_other_ref_identities(cleanup_module):
    sha = "c" * 40
    runs = all_expected_runs(cleanup_module, sha)
    runs.extend(
        (
            workflow_run(sorted(cleanup_module.EXPECTED_WORKFLOWS)[0], "d" * 40),
            workflow_run(
                sorted(cleanup_module.EXPECTED_WORKFLOWS)[0],
                sha,
                branch="pull-request/999",
            ),
            {
                **workflow_run(sorted(cleanup_module.EXPECTED_WORKFLOWS)[0], sha),
                "event": "workflow_dispatch",
            },
        )
    )

    result = cleanup_module.materialization_status(
        MaterializationClient(runs), "pull-request/123", sha
    )

    assert result.ready


@pytest.mark.parametrize("action", ("requested", "in_progress"))
def test_workflow_run_event_accepts_controller_actions(cleanup_module, action):
    event = cleanup_module.validate_workflow_run_event(
        workflow_event(cleanup_module, action), "ai-dynamo/dynamo"
    )

    assert event.action == action
    assert event.branch == "pull-request/123"
    assert event.sha == "a" * 40


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    (
        (("action",), "completed", "Unsupported workflow_run action"),
        (("repository", "full_name"), "other/repo", "repository does not match"),
        (("workflow_run", "event"), "pull_request", "Only push workflow runs"),
        (
            ("workflow_run", "head_repository", "full_name"),
            "other/repo",
            "head repository is not this repository",
        ),
        (("workflow_run", "head_branch"), "main", "Invalid pull request branch"),
        (("workflow_run", "head_sha"), "abc", "Invalid commit SHA"),
        (("workflow_run", "path"), ".github/workflows/other.yml", "Unexpected"),
    ),
)
def test_workflow_run_event_rejects_untrusted_identity(
    cleanup_module, field_path, value, message
):
    payload = workflow_event(cleanup_module)
    target = payload
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = value

    with pytest.raises(ValueError, match=message):
        cleanup_module.validate_workflow_run_event(payload, "ai-dynamo/dynamo")


def test_load_workflow_run_event_reads_json(cleanup_module, tmp_path):
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(workflow_event(cleanup_module)), encoding="utf-8")

    event = cleanup_module.load_workflow_run_event(event_path, "ai-dynamo/dynamo")

    assert event.workflow_path in cleanup_module.EXPECTED_WORKFLOWS


@pytest.mark.parametrize("branch", ("pull-request/foo", "pull-request/1/extra", "main"))
def test_branch_validation_rejects_non_numeric_refs(cleanup_module, branch):
    with pytest.raises(ValueError, match="Invalid pull request branch"):
        cleanup_module.validate_branch(branch)


@pytest.mark.parametrize("sha", ("abc", "g" * 40, "a" * 41))
def test_sha_validation_requires_exact_commit_sha(cleanup_module, sha):
    with pytest.raises(ValueError, match="Invalid commit SHA"):
        cleanup_module.validate_sha(sha)


def test_delete_refs_does_not_push_missing_or_moved_refs(cleanup_module, monkeypatch):
    sha = "1" * 40
    client = RefClient((None, "2" * 40))

    def unexpected_push(repository, expected_refs):
        raise AssertionError("push should not run")

    monkeypatch.setattr(cleanup_module, "push_ref_deletions", unexpected_push)
    outcomes = cleanup_module.delete_refs(
        client,
        "ai-dynamo/dynamo",
        {"pull-request/1": sha, "pull-request/2": sha},
    )

    assert {outcome.outcome for outcome in outcomes} == {"already-missing", "moved"}


def test_push_command_uses_exact_force_with_lease(cleanup_module, tmp_path):
    sha = "3" * 40

    command = cleanup_module.build_push_command(
        "ai-dynamo/dynamo", tmp_path, {"pull-request/123": sha}
    )

    assert f"--force-with-lease=refs/heads/pull-request/123:{sha}" in command
    assert command[-1] == ":refs/heads/pull-request/123"
    assert "https://github.com/ai-dynamo/dynamo.git" in command
    assert "core.hooksPath=/dev/null" in command
    assert all("token" not in argument.lower() for argument in command)


def test_cleanup_defers_until_all_runs_materialize(cleanup_module, monkeypatch):
    sha = "4" * 40
    runs = all_expected_runs(cleanup_module, sha)[:-1]

    def unexpected_push(repository, expected_refs):
        raise AssertionError("push should not run")

    monkeypatch.setattr(cleanup_module, "push_ref_deletions", unexpected_push)

    result = cleanup_module.cleanup_branch(
        CleanupClient(runs, (sha,)),
        "ai-dynamo/dynamo",
        "pull-request/123",
        sha,
    )

    assert result == 0


def test_cleanup_fails_controller_when_deletion_is_rejected(
    cleanup_module, monkeypatch, capsys
):
    sha = "5" * 40
    runs = all_expected_runs(cleanup_module, sha)
    client = CleanupClient(runs, (sha, sha, sha))

    monkeypatch.setattr(cleanup_module, "push_ref_deletions", lambda _repo, _refs: 1)
    result = cleanup_module.cleanup_branch(
        client, "ai-dynamo/dynamo", "pull-request/123", sha
    )

    assert result == 1
    assert "::error title=Pull request branch cleanup::" in capsys.readouterr().out


def test_cleanup_deletes_after_all_runs_materialize(cleanup_module, monkeypatch):
    sha = "6" * 40
    runs = all_expected_runs(cleanup_module, sha)
    client = CleanupClient(runs, (sha, sha))
    pushed = []

    def successful_push(repository, expected_refs):
        pushed.append((repository, expected_refs))
        return 0

    monkeypatch.setattr(cleanup_module, "push_ref_deletions", successful_push)
    result = cleanup_module.cleanup_branch(
        client, "ai-dynamo/dynamo", "pull-request/123", sha
    )

    assert result == 0
    assert pushed == [("ai-dynamo/dynamo", {"pull-request/123": sha})]


def test_controller_uses_default_branch_code_and_scoped_tokens(cleanup_module):
    workflow = yaml.safe_load(CONTROLLER_PATH.read_text(encoding="utf-8"))
    trigger = workflow[True]["workflow_run"]

    assert set(trigger["workflows"]) == {
        "PR",
        "PR-XPU",
        "Fern Docs",
        "Request NVSkills CI",
    }
    assert set(trigger["types"]) == {"requested", "in_progress"}
    assert trigger["branches"] == ["pull-request/[0-9]+"]
    assert workflow["permissions"] == {"actions": "read", "contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is False

    job = workflow["jobs"]["cleanup"]
    assert job["environment"] == "pull-request-branch-cleanup"
    assert "workflow_run.event == 'push'" in job["if"]

    checkout_step, token_step, cleanup_step = job["steps"]
    assert (
        checkout_step["with"]["ref"] == "${{ github.event.repository.default_branch }}"
    )
    assert checkout_step["with"]["persist-credentials"] is False
    assert "github.event.workflow_run.head_sha" not in str(checkout_step)

    action_sha = token_step["uses"].rsplit("@", 1)[1]
    assert re.fullmatch(r"[0-9a-f]{40}", action_sha)
    assert token_step["with"] == {
        "client-id": "${{ vars.PR_BRANCH_CLEANER_CLIENT_ID }}",
        "private-key": "${{ secrets.PR_BRANCH_CLEANER_PRIVATE_KEY }}",
        "permission-contents": "write",
    }
    assert cleanup_step["env"] == {
        "BRANCH_DELETE_TOKEN": "${{ steps.maintenance-token.outputs.token }}",
        "GH_TOKEN": "${{ github.token }}",
    }


def test_controller_names_every_expected_workflow(cleanup_module):
    controller = yaml.safe_load(CONTROLLER_PATH.read_text(encoding="utf-8"))
    trigger_names = set(controller[True]["workflow_run"]["workflows"])
    expected_names = {
        yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))["name"]
        for path in cleanup_module.EXPECTED_WORKFLOWS
    }

    assert trigger_names == expected_names


def test_expected_workflows_are_the_complete_push_trigger_set(cleanup_module):
    matching_workflows = set()
    branch_pattern = "pull-request/[0-9]+"
    workflow_sources = (*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml"))

    for path in workflow_sources:
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        triggers = workflow.get(True, {})
        if not isinstance(triggers, dict) or "push" not in triggers:
            continue
        push = triggers["push"]
        branches = push.get("branches", []) if isinstance(push, dict) else []
        if push is None or branch_pattern in branches:
            matching_workflows.add(str(path.relative_to(REPO_ROOT)))

    assert matching_workflows == cleanup_module.EXPECTED_WORKFLOWS


def test_workload_workflows_have_no_cleanup_jobs(cleanup_module):
    for path in cleanup_module.EXPECTED_WORKFLOWS:
        workflow = yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))
        assert "finalize-pull-request-branch" not in workflow["jobs"]
        assert all(
            job.get("uses") != "./.github/workflows/cleanup-pull-request-branches.yml"
            for job in workflow["jobs"].values()
        )


def test_workflows_do_not_fetch_a_pull_request_branch_by_name(cleanup_module):
    fetch_by_branch = re.compile(
        r"git\s+fetch[^\n]*(pull-request|GITHUB_REF|github\.ref|HEAD_BRANCH)",
        re.IGNORECASE,
    )
    workflow_sources = (*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml"))
    action_sources = tuple((REPO_ROOT / ".github/actions").glob("**/*.yml"))
    violations = []
    for path in (*workflow_sources, *action_sources):
        contents = path.read_text(encoding="utf-8")
        if fetch_by_branch.search(contents):
            violations.append(str(path.relative_to(REPO_ROOT)))

    assert violations == []
