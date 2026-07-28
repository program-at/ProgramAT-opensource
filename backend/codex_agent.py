"""Minimal local Codex orchestration for ProgramAT-generated GitHub issues."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

LogCallback = Callable[[dict], Awaitable[None]]


@dataclass
class CodexRun:
    run_id: str
    issue_number: int
    branch: str
    worktree: Path
    process: Optional[asyncio.subprocess.Process] = None
    task: Optional[asyncio.Task] = None
    cancelled: bool = False
    events: list[dict] = field(default_factory=list)


active_runs: dict[int, CodexRun] = {}


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value[:40] or "tool"


async def _command(
    *args: str,
    cwd: Path,
    env: Optional[dict[str, str]] = None,
    input_text: Optional[str] = None,
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd),
        env=env,
        stdin=asyncio.subprocess.PIPE if input_text is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(
        input_text.encode("utf-8") if input_text is not None else None
    )
    return (
        proc.returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


async def _checked(*args: str, cwd: Path, env: Optional[dict[str, str]] = None) -> str:
    code, stdout, stderr = await _command(*args, cwd=cwd, env=env)
    if code:
        raise RuntimeError(f"{' '.join(args)} failed ({code}): {stderr.strip()}")
    return stdout.strip()


def build_prompt(
    issue_number: int,
    title: str,
    body: str,
    update_text: str = "",
) -> str:
    update = f"\nLatest user update:\n{update_text.strip()}\n" if update_text.strip() else ""
    return f"""Implement ProgramAT GitHub issue #{issue_number}.

Issue title:
{title}

Issue body:
{body}
{update}
Read the issue title and body.
Read and follow .github/copilot-instructions.md.
Modify only the requested Python tool in tools/ unless the public runtime is genuinely insufficient.
Preserve the requested behavior, examples, and output format.
Validate the generated tool before finishing.

Work only in the current isolated checkout. Do not create branches, commits, pushes,
pull requests, or GitHub comments; the backend will inspect, validate, and publish
the result. Do not modify unrelated files.
"""


async def cancel_run(issue_number: Optional[int] = None) -> bool:
    candidates = (
        [active_runs.get(issue_number)]
        if issue_number is not None
        else list(active_runs.values())[-1:]
    )
    run = next((item for item in candidates if item is not None), None)
    if run is None:
        return False
    run.cancelled = True
    if run.process and run.process.returncode is None:
        run.process.terminate()
        try:
            await asyncio.wait_for(run.process.wait(), timeout=5)
        except asyncio.TimeoutError:
            run.process.kill()
            await run.process.wait()
    if run.task and run.task is not asyncio.current_task() and not run.task.done():
        await run.task
    return True


async def run_issue(
    *,
    repo_root: Path,
    issue_number: int,
    title: str,
    body: str,
    github_repo: str,
    github_token: str,
    codex_binary: str,
    codex_home: Optional[Path],
    model: str,
    base_branch: str,
    worktree_root: Path,
    update_text: str = "",
    existing_branch: str = "",
    existing_pr_number: Optional[int] = None,
    log_callback: Optional[LogCallback] = None,
) -> dict:
    """Run Codex in an isolated worktree, validate, commit, push, and create/update a PR."""

    await cancel_run(issue_number)
    branch = existing_branch or f"codex/issue-{issue_number}-{_slug(title)}"
    run_id = f"codex-issue-{issue_number}-{uuid.uuid4().hex[:8]}"
    worktree = worktree_root / run_id
    run = CodexRun(run_id, issue_number, branch, worktree)
    active_runs[issue_number] = run

    async def emit(event: str, **fields) -> None:
        payload = {
            "provider": "codex",
            "run_id": run_id,
            "issue_number": issue_number,
            "event": event,
            **fields,
        }
        logger.info("[Codex] %s", payload)
        if log_callback:
            await log_callback(payload)

    async def execute() -> dict:
        worktree_root.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["GH_TOKEN"] = github_token
        if codex_home is not None:
            env["CODEX_HOME"] = str(codex_home)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"programat-codex-{issue_number}-"))
        askpass = temp_dir / "git-askpass.py"
        askpass.write_text(
            "#!/usr/bin/env python3\n"
            "import base64, os, sys\n"
            "prompt = sys.argv[1] if len(sys.argv) > 1 else ''\n"
            "value = 'x-access-token' if 'Username' in prompt else "
            "base64.b64decode(os.environ['PROGRAMAT_GIT_TOKEN_B64']).decode()\n"
            "print(value)\n",
            encoding="utf-8",
        )
        askpass.chmod(0o700)
        env["GIT_ASKPASS"] = str(askpass)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["PROGRAMAT_GIT_TOKEN_B64"] = base64.b64encode(
            github_token.encode("utf-8")
        ).decode("ascii")
        issue_file: Optional[Path] = None
        last_message: Optional[Path] = None
        try:
            await emit("starting", branch=branch)
            ref = f"origin/{existing_branch or base_branch}"
            await _checked(
                "git", "fetch", "origin", existing_branch or base_branch,
                cwd=repo_root, env=env,
            )
            await _checked(
                "git", "worktree", "add", "--force", "-B", branch, str(worktree), ref,
                cwd=repo_root,
            )
            await emit("worktree_ready", worktree=str(worktree))

            issue_file = temp_dir / "issue.md"
            issue_file.write_text(body, encoding="utf-8")
            last_message = temp_dir / "last-message.txt"

            prompt = build_prompt(issue_number, title, body, update_text)
            args = [
                codex_binary, "exec", "--cd", str(worktree),
                "--sandbox", "workspace-write", "--json", "--ephemeral",
                "--output-last-message", str(last_message),
            ]
            if model:
                args.extend(["--model", model])
            args.append("-")
            run.process = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(worktree),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert run.process.stdin and run.process.stdout and run.process.stderr
            run.process.stdin.write(prompt.encode("utf-8"))
            await run.process.stdin.drain()
            run.process.stdin.close()

            async def read_stdout() -> None:
                async for raw in run.process.stdout:
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        event = {"raw": line}
                    run.events.append(event)
                    await emit("jsonl", data=event)

            async def read_stderr() -> str:
                chunks = []
                async for raw in run.process.stderr:
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if line:
                        chunks.append(line)
                        await emit("stderr", line=line)
                return "\n".join(chunks)

            _, stderr = await asyncio.gather(read_stdout(), read_stderr())
            exit_code = await run.process.wait()
            final_message = (
                last_message.read_text(encoding="utf-8").strip()
                if last_message.exists() else ""
            )
            if run.cancelled:
                await emit("cancelled", exit_code=exit_code)
                return {"status": "cancelled", "run_id": run_id}
            if exit_code:
                raise RuntimeError(f"Codex exited with {exit_code}: {stderr[-2000:]}")

            status = await _checked("git", "status", "--porcelain", cwd=worktree)
            changed_files = []
            for line in status.splitlines():
                path = line[3:].split(" -> ")[-1].strip()
                if path:
                    changed_files.append(path)
            tool_files = [
                path for path in changed_files
                if path.startswith("tools/") and path.endswith(".py")
            ]
            unrelated = [path for path in changed_files if path not in tool_files]
            if not tool_files:
                raise RuntimeError("Codex did not create or modify a Python file in tools/.")
            if unrelated:
                raise RuntimeError(
                    "Codex modified files outside the requested Python tool: "
                    + ", ".join(unrelated)
                )

            diff = await _checked("git", "diff", "--", *tool_files, cwd=worktree)
            for path in tool_files:
                if f"?? {path}" in status:
                    diff += f"\n--- /dev/null\n+++ b/{path}\n"
                    diff += (worktree / path).read_text(encoding="utf-8")
            await emit("changes_inspected", changed_files=changed_files, diff=diff)
            validation_args = [
                sys.executable,
                str(worktree / "backend" / "validate_generated_tools.py"),
                *tool_files,
                "--issue-file",
                str(issue_file),
            ]
            validation_code, validation_out, validation_err = await _command(
                *validation_args, cwd=worktree
            )
            await emit(
                "validation",
                passed=validation_code == 0,
                stdout=validation_out,
                stderr=validation_err,
            )
            if validation_code:
                raise RuntimeError(
                    f"Generated-tool validation failed: {validation_out}\n{validation_err}"
                )

            await _checked("git", "add", "--", *tool_files, cwd=worktree)
            await _checked(
                "git", "commit", "-m", f"Implement tool for issue #{issue_number}",
                cwd=worktree,
            )
            await _checked(
                "git", "push", "-u", "origin", f"HEAD:{branch}",
                cwd=worktree, env=env,
            )

            pr_number = existing_pr_number
            pr_url = ""
            if pr_number is None:
                pr_url = await _checked(
                    "gh", "pr", "create",
                    "--repo", github_repo,
                    "--base", base_branch,
                    "--head", branch,
                    "--title", f"{title} (Fixes #{issue_number})",
                    "--body", (
                        f"Implements the requested ProgramAT tool.\n\n"
                        f"Fixes #{issue_number}"
                    ),
                    cwd=worktree,
                    env=env,
                )
                match = re.search(r"/pull/(\d+)", pr_url)
                pr_number = int(match.group(1)) if match else None
            else:
                pr_url = await _checked(
                    "gh", "pr", "view", str(pr_number), "--repo", github_repo,
                    "--json", "url", "--jq", ".url", cwd=worktree, env=env,
                )
            await emit(
                "completed",
                branch=branch,
                pr_number=pr_number,
                pr_url=pr_url,
                final_message=final_message,
                changed_files=changed_files,
            )
            return {
                "status": "completed",
                "run_id": run_id,
                "branch": branch,
                "pr_number": pr_number,
                "pr_url": pr_url,
                "changed_files": changed_files,
                "validation": validation_out,
                "final_message": final_message,
            }
        except asyncio.CancelledError:
            await cancel_run(issue_number)
            await emit("cancelled")
            raise
        except Exception as exc:
            await emit("failed", error=str(exc))
            return {"status": "failed", "run_id": run_id, "error": str(exc)}
        finally:
            run.process = None
            if worktree.exists():
                await _command(
                    "git", "worktree", "remove", "--force", str(worktree), cwd=repo_root
                )
            await _command("git", "worktree", "prune", cwd=repo_root)
            shutil.rmtree(temp_dir, ignore_errors=True)
            if active_runs.get(issue_number) is run:
                active_runs.pop(issue_number, None)

    run.task = asyncio.create_task(execute())
    return await run.task
