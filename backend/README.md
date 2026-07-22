# ProgramAT Backend Server

WebSocket server for receiving camera frames and audio transcriptions from the ProgramAT mobile app.

## Features

- **Frame Receiving**: Accepts base64-encoded JPEG images from mobile clients
- **Text Processing**: Receives and logs transcribed text messages
- **GitHub Integration**: Automatically creates GitHub issues from voice transcriptions after a 5-second pause
- **Statistics Broadcasting**: Sends real-time statistics to connected clients
- **Connection Management**: Handles multiple WebSocket connections with automatic cleanup

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Optional: install the Tesseract system binary if you configure
   `tesseract_local` as an OCR candidate:

   Ubuntu/WSL:
   ```bash
   sudo apt-get install -y tesseract-ocr
   ```

   macOS:
   ```bash
   brew install tesseract
   ```

## Configuration

The server uses environment variables for configuration:

### Required for GitHub Integration

- `GITHUB_TOKEN`: Your GitHub personal access token with `repo` scope
  - Create one at: https://github.com/settings/tokens
  - Required permissions: `repo` (to create issues)

- `GITHUB_REPO`: Repository in format `owner/repo` (default: `idea11y/ProgramAT`)

### Required for AI-Powered Template Filling

- `SYSTEM_MODEL` in `backend/model_registry.py`: fixed model used for internal LLM work.

- Provider credentials depend on the models selected by a tool policy; the fixed system model uses `GROQ_API_KEY`.
  - System calls use the fixed system model.
  - The default take-photo policy makes one Gemini 3.1 Flash Lite call.

### LLM Interfaces

ProgramAT separates fixed infrastructure LLM calls from take-photo capability execution.

For take-photo tools, the system LLM extracts issue fields and the coding agent
authors one fused `TOOL_PROMPT`. The generated tool makes exactly one call to
the shared VLM helper and does not execute task stages, routers, or specialist
calls. The helper executes an optional literal `TOOL_POLICY`; without one it
uses the default single-model Gemini policy.

Infrastructure/system work should call `system_llm_call(...)` from `tool_policy_runtime.py`.

Generated tools call `call_take_photo_vlm(...)`. A minimal policy is:

```json
{"strategy": "single", "models": ["gemini-3.1-flash-lite"]}
```

Models and strategy schemas live in `model_registry.py` and
`strategy_registry.py`. The runtime does not infer or generate policies.

### Optional

- `ENABLE_MOONDREAM_CLOUD`: Enable Moondream when an explicit policy selects it (`true` by default).
- `MOONDREAM_MODEL`: Cloud model (`moondream/moondream3-preview` by default).
- `MOONDREAM_TIMEOUT_SECONDS`: Per-request timeout (`2.0` by default).
- `MOONDREAM_MAX_IMAGE_DIMENSION`: Moondream-only resize bound (`768` by default).
- `MOONDREAM_JPEG_QUALITY`: Moondream-only RGB JPEG quality (`75` by default).
- `MOONDREAM_FAILURE_THRESHOLD`: Consecutive non-500 failures before cooldown (`3` by default).
- `MOONDREAM_FAILURE_COOLDOWN_SECONDS`: Cooldown after a provider 500 or repeated failures (`60` by default).
- `STREAMING_VLM_MAX_DIMENSION`: Maximum dimension for streaming VLM images (`640` by default).
- `STREAMING_VLM_JPEG_QUALITY`: JPEG quality for streaming VLM images (`70` by default).
- `OCR_MAX_DIMENSION`: Maximum dimension for streaming OCR images (`1024` by default).
- `OCR_JPEG_QUALITY`: JPEG quality for streaming OCR images (`80` by default).
- `EVALUATOR_DEBUG_REASON`: Make a second text-only evaluator call and log its reason (`false` by default).
- `MIN_STREAMING_EXECUTION_INTERVAL`: Minimum seconds between the completion of one streaming execution and the start of the next (default: `2.0`). Frames received while waiting replace the pending frame, so the newest scene is analyzed when the interval expires.
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8081`)
- `PAUSE_DURATION`: Seconds to wait before creating issue (default: `5.0`)

## Usage

Run one direct smoke test using the same image preprocessing, short prompt adapter, request builder, and response parser as production:

```bash
cd backend
./.venv/bin/python scripts/test_moondream_provider.py
```

### Basic Usage

```bash
python stream_server.py
```

### With GitHub Integration and AI

```bash
export GITHUB_TOKEN="your_github_token_here"
export GEMINI_API_KEY="your_gemini_api_key_here"
export OPENAI_API_KEY="your_openai_key_here"
export GITHUB_REPO="owner/repo"
python stream_server.py
```

## How GitHub Integration Works

1. **Text Reception**: Server receives complete sentences from mobile app
2. **Pause Detection**: Monitors for 5-second pause after last sentence received
3. **AI Parsing**: Uses LiteLLM (configurable provider backends) to parse transcript and extract structured information
   - Determines if it's a bug report or feature request
   - Extracts relevant fields (title, description, steps, etc.)
4. **Template Filling**: Fills appropriate GitHub issue template with parsed data
5. **Issue Creation**: Creates GitHub issue with filled template and appropriate labels
6. **Reset**: Clears tracking and waits for next transcript

### Example Flow

```
User speaks: "The camera crashes when I try to take a photo. I click the photo button and the app freezes."
→ Complete sentences received by server
→ 5 seconds pass with no new text
→ AI parses transcript:
  - Type: Bug Report
  - Title: "Camera crashes when taking photo"
  - Description: "The camera crashes when I try to take a photo"
  - Actual: "App freezes when clicking photo button"
→ GitHub issue created with filled bug report template
```

### Sentence Detection

The mobile app now only streams complete sentences (ending with `.`, `!`, or `?`) to the server. This ensures:
- More coherent issue descriptions
- Better AI parsing accuracy
- Reduced noise from partial transcripts

## Message Format

### Text Messages from Client

```json
{
  "text": "Transcribed text here",
  "timestamp": 1234567890
}
```

### Frame Messages from Client

```json
{
  "frameNumber": 0,
  "timestamp": 1234567890,
  "data": {
    "base64Image": "data:image/jpeg;base64,...",
    "width": 1920,
    "height": 1080
  }
}
```

## Logging

All received text messages are logged to `./received_frames/received_texts.log` with timestamps.

## Multi-Turn Conversations

The server supports multi-turn conversations to gather complete information for GitHub issues:

### How it works:

1. **Initial Input**: User provides partial information (e.g., "The camera crashes")
2. **Missing Field Detection**: AI identifies what's missing (e.g., steps, expected behavior)
3. **Feedback Request**: Server sends feedback via WebSocket asking for missing information
4. **TTS Feedback**: Mobile app speaks the feedback to the user
5. **Follow-up Input**: User provides additional details
6. **Data Merging**: Server merges new information with previous data
7. **Completion**: Process repeats until all required fields are filled
8. **Issue Creation**: GitHub issue is created with complete information

### Example Conversation:

```
User: "The camera crashes."
→ App: 🔊 "I need steps to reproduce, expected behavior, and actual behavior."

User: "I open the camera and click the photo button. I expected to take a photo but it freezes."
→ Server merges data → Creates issue with all fields filled
→ App: 🔊 "Issue 42 created successfully."
```

### WebSocket Message Types:

**Feedback Message** (Server → Client):
```json
{
  "type": "feedback",
  "message": "I need steps to reproduce and expected behavior.",
  "missing_fields": ["steps", "expected"]
}
```

**Issue Created Message** (Server → Client):
```json
{
  "type": "issue_created",
  "message": "Issue #42 created successfully",
  "issue_number": 42,
  "issue_url": "https://github.com/owner/repo/issues/42"
}
```

## Security Notes

- Never commit your `GITHUB_TOKEN` or any provider API key (for example `GEMINI_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`) to source control
- Use environment variables or a secure secrets manager
- The GitHub token should have minimal required permissions (only `repo` scope)
- Provider API keys should be kept secure and rotated regularly
- Monitor your model provider usage to avoid unexpected costs
- Consider using GitHub Apps for production deployments

## Troubleshooting

### GitHub Issues Not Created

1. Check that `GITHUB_TOKEN` environment variable is set
2. Verify token has `repo` permissions
3. Check server logs for error messages
4. Ensure repository exists and token has access
5. Verify issue templates exist in `.github/ISSUE_TEMPLATE/`

### AI Parsing Not Working

1. Check that a provider API key is set (for example `GEMINI_API_KEY` or `OPENAI_API_KEY`) when using a provider-specific model
2. Verify the API key is valid and has sufficient quota
3. Check server logs for LiteLLM / provider errors
4. Server will fall back to simpler parsing if AI fails

### Sentence Detection Issues

1. Ensure mobile app is updated with sentence detection logic
2. Check that sentences end with proper punctuation (`.`, `!`, `?`)
3. Review `received_texts.log` to see what's being received

### Connection Issues

1. Verify firewall allows connections on port 8081
2. Check that client is connecting to correct server IP
3. Review server logs for connection errors

## Development

### Testing GitHub Integration

For testing without actually creating issues, you can:

1. Leave `GITHUB_TOKEN` unset (server will log warnings but continue)
2. Use a test repository for `GITHUB_REPO`
3. Monitor `received_texts.log` to verify text is being received

## Dependencies

- `websockets`: WebSocket server implementation
- `opencv-python`: Image processing
- `numpy`: Numerical operations
- `pillow`: Image handling
- `PyGithub`: GitHub API client
