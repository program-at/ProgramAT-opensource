# ProgramAT System Architecture

## System Overview

ProgramAT is a **mobile-first assistive technology platform** for blind and visually impaired users. It combines a React Native app with a Python WebSocket backend to run AI-powered vision tools in real-time. Camera frames stream from the device to the server, where Python tools process them using ML models (YOLO, Gemini, Google Vision) and return audio-friendly results that are spoken aloud via TTS. The platform also supports GitHub issue/PR management and live Copilot session monitoring.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    Mobile App (React Native / iOS & Android)      │
│                                                                    │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────┐  ┌────────┐ │
│  │  PRs & Text   │  │ Tools/Runner  │  │   Chat   │  │Settings│ │
│  │  (Dev mode)   │  │ (both modes)  │  │          │  │        │ │
│  └───────┬───────┘  └───────┬───────┘  └────┬─────┘  └────────┘ │
│          │                  │               │                     │
│          └──────────────────┴───────────────┘                    │
│                             │                                     │
│               ┌─────────────▼──────────────┐                     │
│               │       WebSocketService      │                     │
│               │  (native WebSocket, not     │                     │
│               │   Socket.IO)                │                     │
│               └─────────────┬──────────────┘                     │
│                             │                                     │
│       ┌─────────────────────┤                                     │
│       │                     │                                     │
│  ┌────▼──────┐  ┌───────────▼────────┐  ┌────────────────────┐  │
│  │CameraView │  │  AudioOutputService │  │TextToSpeechService │  │
│  │(frame     │  │  (TTS, beeps,       │  │ (react-native-tts) │  │
│  │ capture)  │  │   earcons, haptics) │  │                    │  │
│  └───────────┘  └────────────────────┘  └────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               │ WebSocket (ws://server:8080)
                               │ Binary / JSON messages
                               │
┌──────────────────────────────▼───────────────────────────────────┐
│                    Python Backend (GCP VM)                         │
│                      stream_server.py                             │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                 Message Router (async)                    │    │
│  │  frame | text | run_tool | start_streaming | stop |       │    │
│  │  request_issue_list | request_pr_list | ping | ...        │    │
│  └───┬──────────────┬───────────────────┬────────────────────┘   │
│      │              │                   │                          │
│  ┌───▼────┐  ┌──────▼──────┐  ┌────────▼─────────────────────┐  │
│  │ Frame  │  │ Text / Issue │  │       Tool Executor          │  │
│  │ Store  │  │ Processor    │  │                              │  │
│  │(disk + │  │ (Gemini AI   │  │  • exec() Python tool code   │  │
│  │ last   │  │  parsing,    │  │  • YOLO model cache          │  │
│  │ frame) │  │  GitHub ops) │  │  • Module auto-install       │  │
│  └────────┘  └─────────────┘  │  • Streaming (per-frame loop)│  │
│                                │  • Gemini Live (custom GPT)  │  │
│                                └──────────────────────────────┘  │
│                                                                    │
│  ┌────────────────────┐  ┌────────────────┐  ┌─────────────────┐ │
│  │  GeminiLiveManager │  │  copilot_db.py │  │GeminiSummarizer │ │
│  │  (native audio,    │  │  (SQLite,      │  │(summarize logs  │ │
│  │   real-time conv.) │  │   sessions,    │  │ for TTS)        │ │
│  │                    │  │   logs,        │  │                 │ │
│  └────────────────────┘  │   summaries)   │  └─────────────────┘ │
│                           └────────────────┘                      │
└───────────────────────────────────────────────────────────────────┘

External Services:
  • GitHub API (PyGithub)        — issue/PR/comment management
  • Google Gemini API            — AI parsing, tool execution, live audio
  • Google Cloud Vision API      — OCR / text extraction (in tools)
  • GCP Secret Manager           — secret storage (GitHub token, etc.)
```

## Frontend Architecture

### Technology Stack
- **Framework**: React Native (iOS & Android)
- **Language**: TypeScript
- **Camera**: react-native-vision-camera
- **Speech-to-Text**: @react-native-voice/voice (used in ToolRunner for follow-up queries)
- **Text-to-Speech**: react-native-tts + `AudioOutputService.ts`
- **WebSocket**: Native WebSocket (not Socket.IO)
- **Storage**: AsyncStorage (persisted tool selections, chat sessions)
- **UI**: React Native core components + react-native-safe-area-context

### App Modes

The app runs in one of two modes configured in `config.ts`:

| Feature | Development | Production |
|---------|-------------|------------|
| PRs & Text tab | ✅ | ❌ |
| Issues tab | ✅ | ❌ |
| PR / branch selection | ✅ | ❌ |
| Tools tab | ✅ | ✅ |
| Chat tab | ✅ | ✅ |
| Tools loaded from | Any PR/branch | `main` only |
| Default tab on launch | PRs & Text | Tools |

### Key Components

#### 1. **TabNavigator.tsx** – App Shell
- Bottom tab bar with: **PRs & Text** (dev only), **Tools**, **Chat**, **Settings**
- Passes `issueTools`, `prList`, `copilotSessions/Logs/Summaries` down to children
- Handles mode switching (Dev ↔ Production)
- Coordinates cross-tab navigation (e.g. ToolRunner → Chat)

#### 2. **PRsAndText.tsx** – Dev Workflow Tab *(development mode only)*
- Three sub-views: PR list (`IssueSelector`), text input (`TextInput`), log viewer (`ViewLogsScreen`)
- Browse open PRs, select one, compose issue text, watch Copilot logs

#### 3. **ToolsAndRunner.tsx** – Tool Tab
- Composes `ToolSelector` + `ToolRunner`
- Persists selected tool via AsyncStorage across app restarts
- Refreshes tool definition when code or `gpt_query` changes on server

#### 4. **ToolSelector.tsx** – Tool Browser
- Fetches available tools from backend via `request_tool_list` WebSocket message
- Displays tools from the `tools/` directory (filtered by mode)
- Plays loading beep if fetch takes > 3 seconds

#### 5. **ToolRunner.tsx** – Tool Execution UI
- Sends `run_tool` / `start_streaming_tool` / `stop_streaming_tool` messages
- Displays text output from tool results
- **Audio output**: passes result through `AudioOutputService` (TTS, beeps, earcons, haptics)
- **Streaming mode**: continuous per-frame execution with throttling and similarity filtering
- **Custom GPT mode**: activates `GeminiLiveSession` on the server; supports voice follow-up questions via `@react-native-voice/voice`; auto-re-queries every `query_interval` seconds
- Camera frames captured via `CameraView` ref and attached to each run

#### 6. **CameraView.tsx** – Camera
- Live preview via react-native-vision-camera
- Captures single frames (base64 JPEG) on demand or on a timer
- Exposes `CameraViewHandle` ref so parent components can trigger captures

#### 7. **Chat.tsx** – Conversation History
- Stores chat sessions in AsyncStorage
- Shows image thumbnail, tool name, and full message history per session
- Supports follow-up questions that are sent to the server as `chat_followup` messages
- Auto-navigated to from ToolRunner when a Gemini Live conversation produces a conversation ID

#### 8. **WebSocketService.ts** – Network Layer
- Native WebSocket connection to `ws://server:8080`
- 25-second heartbeat ping with 8-second pong timeout and auto-reconnect
- 15-second connection timeout
- Single `onMessage` callback dispatched to all registered listeners
- Sends: `StreamFrame` (base64 image), `StreamText` (text), and arbitrary JSON commands

#### 9. **AudioOutputService.ts** – Audio Output
- Unified interface for TTS, beeps (200 / 440 / 880 Hz), earcons (success/error/warning), and haptic patterns
- `interrupt` flag stops current speech before starting new audio
- Tools return a dict with `audio.type` to control which output mode is used

#### 10. **IssueSelector.tsx** – PR/Issue Browser *(dev mode)*
- Browse open GitHub issues and PRs
- Select to enter update mode, view Copilot logs, or navigate to Tools

#### 11. **ViewLogsScreen.tsx** – Copilot Log Viewer *(dev mode)*
- Shows live Copilot agent logs for a selected PR
- Displays rolling summaries generated by `GeminiSummarizer`

### Tool Execution Flow

```
User selects tool in ToolSelector
          │
          ▼
ToolRunner sends start_streaming_tool (or run_tool for one-shot)
          │
          ▼
CameraView captures frame (base64 JPEG)
          │
          ▼
WebSocket sends frame + run command to backend
          │
          ▼
Backend exec()s tool code with (image, input_data)
          │
          ▼
Tool returns string or dict with 'audio'/'text' keys
          │
          ▼
Backend sends tool_result WebSocket message
          │
          ▼
ToolRunner receives result
          │
     ┌────┴────────────────────────────┐
     │                                 │
     ▼                                 ▼
AudioOutputService.speak()     setToolOutput() → UI text
(TTS / beep / earcon)
```

## Backend Architecture

### Technology Stack
- **Language**: Python 3.11
- **WebSocket Server**: `websockets` library (asyncio)
- **AI – General**: Google Gemini API (`gemini-2.5-flash-lite` default; configurable)
- **AI – Live Audio**: `gemini-2.5-flash-native-audio-preview` via `google-genai` SDK
- **Object Detection**: YOLO11 (`yolo11n.pt`) and YOLOWorld (`yolov8s-world.pt`) via Ultralytics
- **OCR**: Google Cloud Vision API
- **GitHub**: PyGithub
- **Image Processing**: OpenCV, NumPy, Pillow
- **Secrets**: GCP Secret Manager (with `.env` fallback)
- **Database**: SQLite via `copilot_db.py`
- **Deployment**: GCP Compute Engine VM

### Core Modules

#### 1. **stream_server.py** – WebSocket Server & Message Router
- Single asyncio event loop; handles all connected clients concurrently
- 20 MB max message size; 20s ping/pong keepalive
- Per-client `SessionLogger` writes timestamped `.log` files to `user_logs/`
- Global state: `last_frame`, `active_streaming_tools`, `yolo_model_cache`, `conversation_images`, `active_copilot_streams`, `pending_copilot_issues`

#### 2. **Tool Executor**
- Tools from the `tools/` directory are fetched from GitHub and `exec()`-ed directly in Python
- Each tool must expose a `main(image, input_data)` function
- Returns a `str` (spoken via TTS) or `dict` with `audio` / `text` keys
- **YOLO model cache** (`yolo_model_cache`) is shared across all tool runs to avoid reloading weights on every frame
- **Module auto-install**: `ModuleManager` installs missing pip packages at runtime and updates `requirements.txt`
- **Streaming tools**: `run_streaming_tools()` loops on every incoming frame; per-client throttle (default 500 ms)
- **Gemini Live tools** (`custom_gpt=True`): handled by `GeminiLiveManager`, bypass the `exec()` loop

#### 3. **module_manager.py** – Dependency Manager
- `install_module(name)` runs `pip install` in a subprocess
- `load_module(name)` imports and caches; installs if missing
- Tracks installed modules per session to avoid duplicate installs
- Maps common import aliases (`cv2` → `opencv-python`, `PIL` → `pillow`, etc.)

#### 4. **gemini_live.py** – Gemini Live Session Manager
- `GeminiLiveSession`: connects to Gemini Live API with native audio response modality
- Background `_receive_loop` collects audio transcription and text parts
- `send_image_query(base64_image, query)` sends a frame + prompt; awaits `turn_complete`
- Images are downscaled to ≤ 1024 px on longest edge before transmission
- `GeminiLiveManager` (in `stream_server.py`) manages per-client sessions and the periodic re-query loop

#### 5. **copilot_db.py** – SQLite Database
- Tables: `copilot_sessions`, `copilot_logs`, `copilot_summaries`
- Tracks GitHub Copilot agent subprocess sessions by PR number
- Bulk log inserts run in a `ThreadPoolExecutor` to avoid blocking the asyncio loop

#### 6. **gemini_summarizer.py** – Log Summarizer
- Summarizes batches of Copilot log entries into 1–3 sentence TTS-friendly strings
- Uses `gemini-2.5-flash-lite` (configurable via `GEMINI_MODEL` env var)
- Called periodically while a Copilot session is streaming

#### 7. **GitHub Integration** (in `stream_server.py`)
- Creates issues from `.github/ISSUE_TEMPLATE/` templates
- Fills templates with Gemini-parsed field data
- Adds comments to existing issues (update mode)
- Mentions `@copilot` for code-change requests
- Polls GitHub for new PRs after Copilot issue creation (`pending_copilot_issues`)

### Message Types

#### Client → Server

| Message type | Key fields | Purpose |
|---|---|---|
| `frame` | `data.base64Image` | Camera frame for active streaming tool |
| `text` | `data.text` | Issue/PR text input (delta-tracked) |
| `run_tool` | `tool`, `image` | One-shot tool execution |
| `start_streaming_tool` | `tool` | Begin continuous per-frame tool |
| `stop_streaming_tool` | — | Stop streaming tool |
| `request_tool_list` | `productionMode`, `issueNumber` | Fetch available tools |
| `request_issue_list` | — | Fetch open GitHub issues |
| `request_pr_list` | — | Fetch open PRs |
| `start_copilot_stream` | `pr_number` | Stream Copilot agent logs for a PR |
| `stop_copilot_stream` | — | Stop Copilot log stream |
| `chat_followup` | `conversationId`, `message` | Follow-up question on a tool result |
| `resume_live_query` | — | Resume Gemini Live query loop after voice input |
| `ping` | — | Keepalive |

#### Server → Client

| Message type | Key fields | Purpose |
|---|---|---|
| `tool_result` | `result`, `audio` | Tool execution output |
| `tool_list` | `tools[]` | Available tools |
| `streaming_started` | `tool_name` | Streaming tool activated |
| `streaming_stopped` | — | Streaming tool deactivated |
| `issue_created` | `issue_number`, `issue_url` | New GitHub issue created |
| `issue_updated` | `issue_number`, `message` | Comment added to issue |
| `issue_list` | `issues[]` | Open GitHub issues |
| `pr_list` | `prs[]` | Open PRs |
| `copilot_log` | `log_line`, `session_id` | Live Copilot agent log line |
| `copilot_summary` | `summary`, `session_id` | Gemini summary of log batch |
| `feedback` | `message`, `missing_fields` | Request more info from user |
| `ack` | `frame_number` | Frame acknowledged |
| `pong` | — | Keepalive response |

### Tools Directory (`tools/`)

Each tool is a standalone Python file with:
- `main(image, input_data)` — entry point called by the server
- Returns `str` (auto-TTS) or `dict` with `audio`/`text` keys
- Tools share building-block patterns but are fully self-contained (no cross-imports at runtime)

| Tool | Description | Model |
|---|---|---|
| `object_recognition.py` | Detect objects in scene | YOLO11 + COCO |
| `scene_description.py` | Describe full scene | Gemini Vision |
| `live_ocr.py` | Real-time text extraction | Google Cloud Vision |
| `door_detection.py` | Detect doors and openings | YOLO11 |
| `empty_seat_detection.py` | Find empty seats | YOLO11 |
| `clothing_recognition.py` | Identify clothing items | YOLOWorld |
| `camera_aiming.py` | Guide camera alignment (clock-face directions) | YOLO11 |

### Processing Flows

#### One-Shot Tool Execution
```
run_tool message received (with base64 image)
    │
    ▼
Decode image → PIL Image
    │
    ▼
exec() tool code in isolated namespace
    │
    ▼
Call main(image, input_data)
    │
    ▼
Serialize result (str or dict)
    │
    ▼
Send tool_result to client → AudioOutputService → TTS
```

#### Streaming Tool Execution
```
start_streaming_tool received
    │
    ▼
Register in active_streaming_tools[client_id]
    │
    ▼
Each incoming frame triggers run_streaming_tools()
    │  (throttled per tool config, default 500 ms)
    ▼
exec() tool → result → tool_result message
```

#### Gemini Live (Custom GPT) Mode
```
start_streaming_tool with custom_gpt=True
    │
    ▼
GeminiLiveManager.start_session(client_id, system_instruction)
    │
    ▼
Background query loop every query_interval seconds:
    send_image_query(last_frame, gpt_query)
    │
    ▼
Receive audio transcription + turn_complete
    │
    ▼
Send tool_result with Gemini's spoken response
    │
    ▼
Client receives → AudioOutputService plays native audio or TTS
    │
    ▼
User can ask follow-up via voice (ToolRunner speech-to-text)
→ resume_live_query restarts the query loop
```

#### GitHub Issue Creation (Dev Mode)
```
Text received → delta extraction (new words only)
    │
    ▼
5-second pause detection
    │
    ▼
Gemini parses transcript → structured issue fields
    │
    ▼
Missing fields? → send feedback → wait for more input
    │
    ▼
All fields present → fill .github/ISSUE_TEMPLATE → create GitHub issue
    │
    ▼
Poll for Copilot PR creation → stream logs to client
```

## Deployment

### Frontend
- React Native app built for iOS/Android
- Distributed via App Store / TestFlight / APK

### Backend
- GCP Compute Engine VM
- WebSocket server: `ws://<SERVER_IP>:8080`
- Python 3.11 virtual environment (`venv`)
- `restart_server.sh` / `clear_frames.sh` utility scripts
- Logs: per-session files in `backend/user_logs/`; SQLite DB at `backend/copilot_logs.db`

### Environment Variables

Backend `.env` (or GCP Secret Manager):
```bash
GITHUB_TOKEN=<GitHub PAT>
GITHUB_REPO=owner/repo
GEMINI_API_KEY=<Google AI API key>
GEMINI_MODEL=gemini-2.5-flash-lite    # or gemini-2.5-flash, etc.
GCP_PROJECT=<project-id>              # for Secret Manager fallback
PAUSE_DURATION=5.0                    # seconds before issue creation
```

Frontend `config.ts`:
```typescript
export const WEBSOCKET_SERVER_URL = 'ws://<SERVER_IP>:8080';
export const APP_MODE: AppMode = 'development'; // or 'production'
```

## Security Considerations

- GitHub PAT stored in GCP Secret Manager (`.env` fallback for local dev)
- WebSocket runs over plain WS — use WSS/TLS behind a reverse proxy for production
- Input validation on backend; tool code is fetched from a trusted GitHub repo
- Rate limiting should be applied at the reverse-proxy level
- Frame storage in `received_frames/` should be cleaned up periodically (`clear_frames.sh`)

---

**Last Updated**: March 24, 2026  
**Architecture Version**: 3.0 (Vision Tools + Gemini Live)
