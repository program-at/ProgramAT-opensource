# ProgramAT

AI-powered assistive-technology platform. Blind and low-vision (BLV) users build
custom camera-based tools by describing them in natural language, then run those
tools live against their phone camera and hear the result spoken back.

**The user is blind.** Audio — speech, beeps, haptics — is the primary output
channel, not an accessibility add-on. Judge every change by whether it works
without sight.

## Repository map

A monorepo with three components and one data flow:

```
phone camera → ProgramATApp → WebSocket → backend → tools/*.py → spoken result
```

- `ProgramATApp/` — React Native mobile app (iOS-first), TypeScript.
- `backend/` — async WebSocket server that runs the tools. Python.
- `tools/` — pluggable camera vision tools, one per file. Python.

Each directory has its own `CLAUDE.md` with details for working there; it loads
automatically when you open files in that directory.

## Documentation

Read these instead of guessing — don't copy their content back into code:

- `README.md` — setup, API keys, running the app and server, usage.
- `ARCHITECTURE.md` — system design and how the components interact.
- `DEV_PRODUCTION_MODES.md` — the development / production / review modes.
- `CONTRIBUTING.md` — contribution categories, reviews, PR requirements.

## Conventions

- Most contributions are a single new tool in `tools/` — see `tools/CLAUDE.md`.
- Branches: `feat/...`, `fix/...`, `docs/...` (kebab-case). `copilot/*` is reserved
  for Copilot-generated tool work — don't push there by hand.
- Never commit secrets: `backend/.env`, `backend/credentials.json`, service-account
  JSON. Check they are git-ignored first.
- This repo is a fork of `program-at/ProgramAT-opensource` (the `upstream` remote).
