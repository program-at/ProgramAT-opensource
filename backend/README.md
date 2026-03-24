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

## Configuration

The server uses environment variables for configuration:

### Required for GitHub Integration

- `GITHUB_TOKEN`: Your GitHub personal access token with `repo` scope
  - Create one at: https://github.com/settings/tokens
  - Required permissions: `repo` (to create issues)

- `GITHUB_REPO`: Repository in format `owner/repo` (default: `idea11y/ProgramAT`)

### Required for AI-Powered Template Filling

- `GEMINI_API_KEY`: Your Google Gemini API key for intelligent issue parsing
  - Create one at: https://aistudio.google.com/app/apikey
  - Required for automatic template filling
  - If not provided, falls back to simple parsing

### Optional

- `GEMINI_MODEL`: Gemini model to use (default: `gemini-3-flash-preview`)
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8080`)
- `PAUSE_DURATION`: Seconds to wait before creating issue (default: `5.0`)

## Usage

### Basic Usage

```bash
python stream_server.py
```

### With GitHub Integration and AI

```bash
export GITHUB_TOKEN="your_github_token_here"
export GEMINI_API_KEY="your_gemini_api_key_here"
export GITHUB_REPO="owner/repo"
python stream_server.py
```

## How GitHub Integration Works

1. **Text Reception**: Server receives complete sentences from mobile app
2. **Pause Detection**: Monitors for 5-second pause after last sentence received
3. **AI Parsing**: Uses Google Gemini to parse transcript and extract structured information
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

- Never commit your `GITHUB_TOKEN` or `GEMINI_API_KEY` to source control
- Use environment variables or a secure secrets manager
- The GitHub token should have minimal required permissions (only `repo` scope)
- The Gemini API key should be kept secure and rotated regularly
- Monitor your Gemini API usage to avoid unexpected costs
- Consider using GitHub Apps for production deployments

## Troubleshooting

### GitHub Issues Not Created

1. Check that `GITHUB_TOKEN` environment variable is set
2. Verify token has `repo` permissions
3. Check server logs for error messages
4. Ensure repository exists and token has access
5. Verify issue templates exist in `.github/ISSUE_TEMPLATE/`

### AI Parsing Not Working

1. Check that `GEMINI_API_KEY` environment variable is set
2. Verify API key is valid and has sufficient quota
3. Check server logs for Gemini API errors
4. Server will fallback to simple parsing if AI fails

### Sentence Detection Issues

1. Ensure mobile app is updated with sentence detection logic
2. Check that sentences end with proper punctuation (`.`, `!`, `?`)
3. Review `received_texts.log` to see what's being received

### Connection Issues

1. Verify firewall allows connections on port 8080
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
