# ProgramAT System Architecture

## System Overview

ProgramAT is a **mobile-first AI-powered issue management system** that combines camera streaming with text input to create and manage GitHub issues through natural language processing.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Mobile App (React Native)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Camera View  │  │  Text Input  │  │Issue Selector│      │
│  │ (Vision Cam) │  │ (with OS     │  │              │      │
│  │              │  │  dictation)  │  │              │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────┬───────┴──────────────────┘              │
│                    │                                         │
│         ┌──────────▼──────────┐                              │
│         │ WebSocket Service   │                              │
│         │   (Socket.IO)       │                              │
│         └──────────┬──────────┘                              │
└────────────────────┼────────────────────────────────────────┘
                     │
                     │ WebSocket (ws://server:8080)
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Python WebSocket Server                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Message Handler (async)                      │   │
│  │  • Receives frames (base64 images)                    │   │
│  │  • Receives text (with delta tracking)                │   │
│  │  • Routes to appropriate processor                    │   │
│  └─────────┬────────────────────────────────┬───────────┘   │
│            │                                 │               │
│  ┌─────────▼──────────┐         ┌───────────▼──────────┐    │
│  │  Frame Processor   │         │   Text Processor     │    │
│  │  • Save to disk    │         │  • Delta extraction  │    │
│  │  • OpenCV process  │         │  • Pause detection   │    │
│  └────────────────────┘         └───────────┬──────────┘    │
│                                              │               │
│                                  ┌───────────▼──────────┐    │
│                                  │   AI Parser          │    │
│                                  │  (Google Gemini)     │    │
│                                  │  • Parse transcript  │    │
│                                  │  • Extract fields    │    │
│                                  │  • Detect missing    │    │
│                                  └───────────┬──────────┘    │
│                                              │               │
│                                  ┌───────────▼──────────┐    │
│                                  │  GitHub Integration  │    │
│                                  │  • Create issues     │    │
│                                  │  • Update issues     │    │
│                                  │  • Mention @copilot  │    │
│                                  └──────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Frontend Architecture

### Technology Stack
- **Framework**: React Native (iOS & Android)
- **Language**: TypeScript
- **Camera**: react-native-vision-camera
- **Text-to-Speech**: react-native-tts (for server feedback)
- **WebSocket**: Socket.IO client
- **UI**: React Native core components

### Key Components

#### 1. **App.tsx** - Main Container
- Connection management (connect/disconnect)
- Mode switching (create vs update)
- Issue selection state
- Server feedback handling
- Coordinates all child components

#### 2. **TextInput.tsx** - Text Input Interface
- **Multi-line text input** with auto-expanding height
- **OS-level dictation support** (microphone icon on keyboard)
- **Auto-send complete sentences** as user types
- **Delta tracking** to avoid resending entire text
- Character counter
- Clear and Send buttons
- Server feedback display

#### 3. **CameraView.tsx** - Camera Frame Capture
- Real-time camera preview
- Periodic frame capture (configurable interval)
- Base64 encoding
- WebSocket streaming

#### 4. **IssueSelector.tsx** - Issue Browser
- Modal interface to browse open GitHub issues
- Search and filter capabilities
- Issue selection for update mode

#### 5. **WebSocketService.ts** - Communication Layer
- Socket.IO connection management
- Message sending (text, frames)
- Event handlers for server responses
- Automatic reconnection

#### 6. **TextToSpeechService.ts** - Voice Feedback
- Speaks server feedback messages
- Handles TTS queue
- Platform-specific implementations

### Input Flow (Text-Based)

```
User Input → TextInput Component → Extract Complete Sentences
    ↓
Sentence Detected (ends with . ! ?)
    ↓
WebSocket.sendText(sentence)
    ↓
Backend Processing
    ↓
Server Response (feedback/success)
    ↓
Display + Optional TTS Spoken Feedback
```

### OS-Level Dictation Support

The system leverages native OS dictation features:

- **iOS**: Microphone icon on keyboard (tap to dictate)
- **Android**: Voice input button on keyboard
- No custom voice recognition library needed
- User can mix typing and dictation seamlessly

## Backend Architecture

### Technology Stack
- **Language**: Python 3.11
- **WebSocket Server**: websockets library
- **AI**: Google Gemini API (gemini-3-flash-preview)
- **GitHub**: PyGithub
- **Image Processing**: OpenCV, NumPy, Pillow
- **Secrets**: GCP Secret Manager
- **Environment**: GCP VM instance

### Core Services

#### 1. **WebSocket Server** (`stream_server.py`)
- Async event loop with websockets
- Handles multiple concurrent connections
- 20MB max message size for large images
- Ping/pong keepalive (20s interval)

#### 2. **Message Router**
- Detects message type (frame, text, command)
- Routes to appropriate processor
- Sends acknowledgments
- Broadcasts stats periodically

#### 3. **Frame Processor**
- Decodes base64 images
- Saves to disk (`received_frames/`)
- OpenCV processing pipeline (ready for ML)
- Size validation

#### 4. **Text Processor**
- **Delta tracking**: Only new words are queued
- Prevents duplicate issue creation
- Logs all text to `received_texts.log`
- Triggers pause detection

#### 5. **AI Parser** (Google Gemini)
- Parses natural language into structured data
- Supports 3 issue types:
  - **Bug Report**: description, steps, expected, actual
  - **Feature Request**: description, problem, solution
  - **Personal Website**: name, bio, skills, education, etc.
- Multi-turn conversations to fill missing fields
- Context-aware follow-ups

#### 6. **GitHub Integration**
- Creates issues from templates
- Fills templates with AI-parsed data
- Updates existing issues with comments
- Mentions @copilot for code changes
- Associates with pull requests

### Processing Flow

```
Text Received
    ↓
Delta Extraction (new words only)
    ↓
Buffer Update (last_text['content'])
    ↓
Pause Detection (5 second timer)
    ↓
AI Parsing (parse_transcript_with_ai)
    ↓
Field Validation (missing_fields check)
    ↓
┌─────────────┬─────────────┐
│  Complete   │ Incomplete  │
│             │             │
│ Fill        │ Generate    │
│ Template    │ Feedback    │
│             │             │
│ Create      │ Send to     │
│ GitHub      │ Client      │
│ Issue       │             │
│             │ Wait for    │
│ Send        │ More Info   │
│ Success     │             │
└─────────────┴─────────────┘
```

### Dual Mode Operation

#### Create Mode (Default)
- Accumulates text via pause detection
- AI parses into issue data
- Asks for missing fields
- Creates new GitHub issue when complete

#### Update Mode
- Activated by "select issue X" command
- OR by browsing issue selector
- All subsequent text becomes comments
- Adds @copilot mention for code changes
- Stays in update mode until switched

### Issue Selection Intelligence

The AI can parse voice commands like:
- "select issue 42"
- "work on the bug about login"
- "update the camera issue"
- "create new issue" (switch back to create mode)
- "show issues" / "list issues"

## Communication Protocol

### WebSocket Message Types

#### Client → Server

| Type | Payload | Purpose |
|------|---------|---------|
| Text | `{data: {text: "..."}}` | Send text transcript |
| Frame | `{data: {base64Image: "..."}}` | Send camera frame |
| Request Issues | `{type: 'request_issue_list'}` | Fetch open issues |
| Ping | `{type: 'ping'}` | Keep-alive |

#### Server → Client

| Type | Payload | Purpose |
|------|---------|---------|
| `ack` | `{frame_number, results}` | Acknowledge receipt |
| `issue_created` | `{issue_number, issue_url}` | Issue created |
| `issue_updated` | `{issue_number, message}` | Comment added |
| `issue_selected` | `{issue_number, issue_title}` | Issue selected |
| `issue_list` | `{issues: [...]}` | List of open issues |
| `feedback` | `{message, missing_fields}` | Request more info |
| `stats` | `{total_frames, fps, ...}` | Server statistics |

## Data Flow Examples

### Example 1: Creating a Bug Report

1. **User types/dictates**: "There's a bug in the login screen. When I enter my password and tap submit, nothing happens. It should log me in."

2. **Backend receives** complete sentences incrementally

3. **AI parses** after 5-second pause:
```json
{
  "type": "bug",
  "title": "Login screen submit button not working",
  "description": "When entering password and tapping submit, nothing happens",
  "steps": "1. Enter password\n2. Tap submit button",
  "expected": "Should log me in",
  "actual": "Nothing happens",
  "missing_fields": []
}
```

4. **Template filled** and **GitHub issue created**

5. **Client receives** success notification with issue URL

### Example 2: Multi-turn Conversation

1. **User**: "I want a new feature"

2. **AI parses**, finds missing fields: `['problem', 'solution']`

3. **Server sends feedback**: "I need problem description."

4. **User**: "Users can't export their data to CSV"

5. **AI merges** with existing data, still missing: `['solution']`

6. **Server**: "I need proposed solution."

7. **User**: "Add an export button that downloads a CSV file"

8. **AI merges**, all fields complete

9. **GitHub issue created**

### Example 3: Updating an Issue

1. **User**: "select issue 42"

2. **Server switches** to update mode

3. **User**: "The login bug also happens on Android"

4. **Server adds comment** to issue #42 (no @copilot mention, just status)

5. **User**: "Fix the submit button handler"

6. **Server adds comment** with @copilot mention (code change requested)

## Deployment

### Frontend
- React Native app built for iOS/Android
- Deployed via App Store / Google Play
- Or distributed via TestFlight / APK

### Backend
- GCP VM instance (Compute Engine)
- Public IP: ws://34.144.178.116:8080
- Python 3.11 virtual environment
- Systemd service for auto-restart
- Logs to syslog

### Environment Variables

Backend `.env` file:
```bash
GITHUB_TOKEN=<GitHub PAT>
GITHUB_REPO=owner/repo
GEMINI_API_KEY=<Google AI API key>
GEMINI_MODEL=gemini-3-flash-preview
PAUSE_DURATION=5.0
```

Frontend `config.ts`:
```typescript
export const WS_URL = 'ws://34.144.178.116:8080';
```

## Key Improvements in Text-Based Version

### Before (Voice-Based)
- Required @react-native-voice/voice library
- Complex voice recognition setup
- Streaming partial results
- Platform-specific voice permissions
- Voice quality dependent

### After (Text-Based)
- Standard React Native TextInput
- OS-level dictation (built into keyboard)
- Simpler implementation
- User can type OR dictate
- Mix typing and voice seamlessly
- Better accessibility
- Works in any environment (quiet/noisy)

## Future Enhancements

1. **Frame Analysis**: Use Gemini Vision to analyze camera frames
2. **Issue Templates**: More template types
3. **Offline Mode**: Queue messages when disconnected
4. **Draft Saving**: Save incomplete issues locally
5. **Rich Text**: Markdown formatting in input
6. **Attachments**: Upload images from camera directly to issues
7. **Voice Commands**: "Send", "Clear", "New issue" voice shortcuts
8. **Multi-language**: Support for non-English input

## Security Considerations

- GitHub PAT stored in GCP Secret Manager
- WebSocket connection (consider WSS/TLS for production)
- Input validation on backend
- Rate limiting for API calls
- Frame storage cleanup (disk space management)

---

**Last Updated**: December 18, 2025
**Architecture Version**: 2.0 (Text-Based)
