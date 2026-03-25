# ProgramAT

This repository contains code for an AI-powered assistive technology platform, ProgramAT. ProgramAT equips blind or low-vision (BLV) users to create custom camera-based assistive technologies via natural language instructions. BLV users can test their camera-based ATs with input from their iPhone's camera. The system has 3 major components:
- a mobile app, installable via TestFlight. Link will be provided closer to the event.
- a server that runs the computation for the camera-based AT you build using the app. This interfaces with a GitHub repository, and runs the AI models necessary for your task. Instructions to set a server up will be updated shortly.
- ProgramAT GitHub repository. This is where your tools will live. It will be a fork of this repository for each individual user. We recommend forking closer to the event date.

> Notice: Setup instructions and TestFlight link will be updated closer to the event.


## What is the ProgramAT Mobile App?

This is a React Native app that facilitates AT creation, iteration, and testing of the created AT using your camera feed. You can also monitor the status of their creation. The app streams frames to a Python backend on your server, which executes pluggable tools — object detection, OCR, scene description, and more and returns spoken feedback.

## Getting Started

### Prerequisites

- A [GitHub account](https://github.com/). Refer to [screen reader friendly instructions by Jeff Bishop](https://community-access.org/git-going-with-github/docs/00-pre-workshop-setup.html).
- A computer with decent compute capacity. Exact specifications coming soon. A GPU is not required.
- A screen reader and an accessible web browser.
- iPhone 12 or higher running iOS 26. Apple Intelligence or AI-specific features for processors are not required.
- Python 3.11+ (for the backend server)
- Node.js >= 20
- Optional: React Native CLI development environment ([setup guide](https://reactnative.dev/docs/set-up-your-environment)). This is required if you want to run and install the app. You can skip this requirement if you are installing the app using our TestFlight link.
- Optional: For iOS: Xcode and CocoaPods. You can skip if you are building and running the app.
- For Android: Android Studio and Android SDK


### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/program-at/ProgramAT-opensource.git
   cd ProgramAT-opensource
   ```

2. Optional (required if you are building the app): **Install React Native dependencies**
   ```bash
   cd ProgramATApp
   npm install
   ```

3. Optional (required if you are building the app): **Install iOS dependencies (iOS only)**
   ```bash
   cd ios
   pod install
   cd ..
   ```

4. **Install backend dependencies**
   ```bash
   cd ../backend
   pip install -r requirements.txt
   ```

### Environment Variables

The backend uses environment variables for API keys and configuration:

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes (for VLM tools) | Google Gemini API key — used by scene description, clothing recognition, AI parsing, and Gemini Live |
| `GOOGLE_APPLICATION_CREDENTIALS` | For OCR tools | Google Cloud Vision API credentials (used by Live OCR) |
| `GITHUB_TOKEN` | For GitHub features | GitHub personal access token with `repo` scope |
| `GITHUB_REPO` | Yes (to be able to access your own tools) | Target repo in `owner/repo` format |
| `HOST` / `PORT` | Optional | Server bind address (default `0.0.0.0:8080`) |

### Running the Application (instructions to be updated)

1. **Start the backend server**
   ```bash
   cd backend
   export GEMINI_API_KEY="your_key_here"
   python stream_server.py
   ```
   The server starts on port 8080.

2. **Start the React Native app**. Skip if you are running the app from the TestFlight link.
   ```bash
   cd ProgramATApp
   npm start
   ```

3. **Run on your device**. Skip if you are running the app via TestFlight.

   For iOS:
   ```bash
   npm run ios
   ```

   For Android:
   ```bash
   npm run android
   ```

### Configuration (will be updated closer to event)

Server URLs are managed in `ProgramATApp/config.ts`. The app supports multiple named servers selectable via a secret code in Settings:

```typescript
export const SERVER_CONFIGS: Record<string, { url: string; name: string }> = {
  'default': { url: 'ws://YOUR_SERVER_IP:8080', name: 'Default Server' },
};
```

Switch between **Development** and **Production** modes in the Settings tab.

## Usage

1. **Connect** — The app auto-connects to the configured server on launch. Connection status is shown in the UI; a loading sound plays while connecting.

2. **Select a Tool** — Navigate to the **Tools** tab, browse the available tools, and tap one to select it.

3. **Run** — The Tool Runner opens with a live camera preview. Tap **Run** to execute the tool on single frames, or **Stream** to process frames continuously. Results are spoken aloud via TTS.

4. **Chat** — After a tool run, tap **Chat** to ask follow-up questions about the result (powered by Gemini).

5. **Development mode** — Use the **PRs** tab to browse open pull requests, select one to load its tools, and send text updates to GitHub issues.

Creation instructions coming soon.


 
## Supported Input Modes

- **Streaming mode** — Tools process frames continuously and return real-time audio feedback
- **Single-frame mode** — Capture one frame and get a detailed result
- **Real-time camera streaming** at configurable FPS via `react-native-vision-camera`
- **Conversation mode** — Ask follow-up questions about tool results via a Chat tab
- **Custom GPT-like tools** — Tools flagged as Custom GPT use Gemini Live for streaming multimodal conversations instead of executing code per frame

## Supported Feedback Modes

- **Text-to-Speech feedback** — all tool results are spoken aloud automatically
- **Rich audio output** — tools can return speech, beeps, haptic vibration, earcons, and more via the AudioOutputService
- **Speech-to-Text input** — voice input for follow-up questions using `@react-native-voice/voice` and OS-level dictation

## Usage Modes
### Development Mode (GitHub Integration)
- **PR browser** — List open pull requests, select one, and load its tools
- **Text input for issues** — Create or update GitHub issues with AI-powered parsing (Gemini)
- **Multi-turn conversations** — The server asks for missing fields until the issue is complete
- **Copilot session logs** — View AI coding session summaries per PR

### Production Mode
- Tools are pulled from the `main` branch only
- The PR browser tab is hidden; users go straight to the tool list

## Built-In Tools

| Tool | Description | Model |
|------|-------------|-------|
| **Object Recognition** | Detects and announces objects using YOLO11 + COCO | YOLOv11 |
| **Live OCR** | Reads visible text aloud in real time | Google Cloud Vision API |
| **Scene Description** | Generates a spoken description of the scene | Google Gemini Vision |
| **Camera Aiming** | Guides users to center an object for a well-framed photo | YOLOv11 |
| **Door Detection** | Detects doors/doorways with clock-face navigation cues | YOLOWorld |
| **Empty Seat Detection** | Finds unoccupied chairs and gives directional guidance | YOLOv11 |
| **Clothing Recognition** | Identifies the most prominent clothing item and its features | Google Gemini Vision |

New tools can be added by placing a Python file in the `tools/` directory. Each tool exposes a `main(image, input_data)` function and returns an audio-friendly string or dict.

## Project Structure

```
ProgramAT-opensource/
├── ProgramATApp/              # React Native mobile app
│   ├── App.tsx               # Root component, WebSocket setup, state management
│   ├── TabNavigator.tsx      # Bottom tabs: PRs, Tools, Chat, Settings
│   ├── ToolSelector.tsx      # Lists available tools
│   ├── ToolRunner.tsx        # Runs selected tool against camera feed
│   ├── CameraView.tsx        # Camera capture & frame streaming
│   ├── Chat.tsx              # Follow-up conversation interface
│   ├── PRsAndText.tsx        # PR browser & text input (dev mode)
│   ├── Settings.tsx          # Mode switching, server selection, theme
│   ├── WebSocketService.ts   # WebSocket connection manager
│   ├── AudioOutputService.ts # Multi-modal audio output (TTS, beeps, haptics)
│   ├── TextToSpeechService.ts
│   ├── BeepService.ts
│   ├── config.ts             # Server URLs, feature flags, mode config
│   └── __tests__/            # Jest test suite
├── backend/                   # Python WebSocket server
│   ├── stream_server.py      # Main server — message routing, tool execution, GitHub integration
│   ├── gemini_live.py        # Gemini Live API session manager for Custom GPT tools
│   ├── gemini_summarizer.py  # Summarizes Copilot session logs
│   ├── copilot_db.py         # SQLite storage for Copilot session data
│   ├── module_manager.py     # Auto-installs missing Python packages at runtime
│   └── requirements.txt      # Python dependencies
└── tools/                     # Pluggable vision/AI tools (Python)
    ├── object_recognition.py
    ├── live_ocr.py
    ├── scene_description.py
    ├── camera_aiming.py
    ├── door_detection.py
    ├── empty_seat_detection.py
    └── clothing_recognition.py
```


## Development

### Running Tests

```bash
cd ProgramATApp
npm test
```

### Adding a New Tool

1. Create a Python file in `tools/` (e.g., `tools/my_tool.py`).
2. Implement a `main(image, input_data)` function that returns an audio-friendly string or dict.
3. The tool is automatically discovered and available in the app when loaded from the server.

See [tools/MODEL_SETUP.md](tools/MODEL_SETUP.md) and the existing tools for examples.

### Building for Production

For iOS:
1. Open `ProgramATApp/ios/ProgramATApp.xcworkspace` in Xcode
2. Select your signing team and provisioning profile
3. Build for Release

For Android:
1. Generate a signing key
2. Configure `android/app/build.gradle`
3. Run `cd android && ./gradlew assembleRelease`

## Technology Stack

### Mobile App
- **React Native 0.82.1** — Bare (non-Expo) for direct native module access
- **TypeScript**
- **react-native-vision-camera** — Camera capture and frame streaming
- **@react-native-voice/voice** — Speech-to-text for follow-up questions
- **react-native-tts** — Text-to-speech for tool results
- **react-native-audio-api** — Programmatic audio generation (beeps, tones)
- **AsyncStorage** — Local persistence for settings and sessions

### Backend
- **Python 3.11** with async `websockets`
- **Google Gemini API** — AI parsing, scene description, clothing recognition, Gemini Live
- **Google Cloud Vision API** — OCR
- **Ultralytics (YOLOv11 / YOLOWorld)** — Object detection
- **OpenCV / NumPy / Pillow** — Image processing
- **PyGithub** — GitHub issue and PR integration

## License

See [LICENSE](LICENSE).

## Contributions


- [Ellie Seehorn](https://seehorne.github.io/) (PhD student at University of Michigan)
- [Venkatesh Potluri](https://venkateshpotluri.me/) (Assistant Professor, University of Michigan. Principal Investigator of the [Intelligent Developer Experiences for Accessibility Lab](https://idea11y.dev/))
- [Anhong Guo](https://guoanhong.com/) (Assistant Professor, University of Michigan and principal investigator of the [Human AI Lab](https://guoanhong.com/))

Found a problem? Please file an issue!