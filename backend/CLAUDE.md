# backend/ — ProgramAT server

Async Python WebSocket server. It receives camera frames and text from the app,
`exec()`s `tools/*.py` against each frame, calls the GitHub API for issues/PRs, and
streams GitHub Copilot session logs. Entry point: `stream_server.py`.

Run all commands below from this `backend/` directory.

## Run it

The backend ships a `uv.lock`, so `uv` gives a fast, reproducible install:

```bash
uv sync                          # creates .venv from the locked dependencies
cp .env.example .env             # then fill in the keys (see README.md)
uv run python stream_server.py   # serves ws://0.0.0.0:8080
```

`README.md` documents the equivalent pip workflow, which also works:
`python3 -m venv .venv`, `source .venv/bin/activate`,
`pip install -r requirements.txt`, then `python stream_server.py`.

`stream_server.py` exits immediately if `GITHUB_TOKEN`, `GITHUB_REPO`, or
`GEMINI_API_KEY` are missing from `.env`. Stop the server with Ctrl-C.

## Layout

- `stream_server.py` — the server: frame/text routing, tool execution, GitHub API.
- `gemini_live.py` — Gemini Live sessions for streaming "Custom GPT" tools.
- `gemini_summarizer.py` — summarizes Copilot session logs for screen readers.
- `copilot_db.py` — SQLite store for Copilot session data.
- `module_manager.py` — installs missing pip packages for tools at runtime.
- `litellm_utils.py` — shared LLM helpers (model-name and API-key resolution).

`ARCHITECTURE.md` (repo root) explains how these fit together.

## How tools run

The server `exec()`s a tool's source and calls `main(image, input_data)` on it.
The relevant code is `handle_client` (connection loop), `run_tool` (one-shot) (found inside the handle_client handler), and
`run_streaming_tools` (per-frame loop) in `stream_server.py`. The tool-authoring
contract lives in `tools/CLAUDE.md`.

## Gotchas

- **`module_manager.py` runs `pip install` at runtime** and appends new packages to
  `requirements.txt` (not `pyproject.toml` / `uv.lock`). An unexpected
  `requirements.txt` diff is usually this.
- **A tool's `print()` output is spoken to the user** — the server captures a tool's
  stdout and prepends it to the TTS text.
- **Blocking calls stall the event loop.** `pip install` and `cv2.imwrite` run
  synchronously; DB and LLM work is offloaded to executors. Keep new I/O off the
  async path.
- **`HOST`/`PORT` are hardcoded** to `0.0.0.0:8080`; the matching `.env` keys are
  ignored.
- `pyproject.toml` requires Python `>=3.12`; the shipped `.venv` is 3.14. If you
  recreate the environment, prefer 3.12/3.13 — some deps (torch, opencv) may lack
  3.14 wheels. `restart_server.sh` points at a stale `venv/` path; the real one
  is `.venv/`.

## Tests

There is no test runner configured. The `test_*.py` files are standalone scripts —
run one directly, e.g. `uv run python test_appearance_check.py`. After changing
tool execution or a specific tool, run its matching `test_*.py`.
