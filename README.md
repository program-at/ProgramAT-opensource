# ProgramAT

A React Native application that streams camera frames and audio transcriptions to a backend WebSocket server for processing.

## Features

- 📹 **Camera Frame Streaming**: Real-time video frame capture and streaming at 4 FPS
- 🎤 **Speech-to-Text**: Live audio transcription with sentence detection and streaming
- 🔌 **WebSocket Connection**: Persistent connection with automatic reconnection
- 📊 **Visual Indicators**: Connection status and streaming statistics
- 🔒 **Permission Management**: Handles camera and microphone permissions
- 🎫 **Intelligent GitHub Integration**: Automatically creates structured GitHub issues from voice transcriptions
  - AI-powered parsing of transcripts
  - Automatic template filling
  - Smart sentence detection (only sends complete sentences)

## Project Structure

```
ProgramAT/
├── ProgramATApp/          # React Native application
│   ├── App.tsx           # Main app component with connection management
│   ├── CameraView.tsx    # Camera component with frame streaming
│   ├── SpeechToText.tsx  # Speech recognition component with text streaming
│   ├── WebSocketService.ts  # WebSocket connection manager
│   ├── __tests__/        # Test files
│   └── STREAMING_GUIDE.md   # Detailed streaming documentation
└── backend/              # Backend server
    └── stream_server.py  # WebSocket server for receiving streams
```

## Getting Started

### Prerequisites

- Node.js >= 20
- React Native development environment set up
- For iOS: Xcode and CocoaPods
- For Android: Android Studio and Android SDK
- Python 3.x (for backend server)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/program-at/ProgramAT-opensource.git
   cd ProgramAT-opensource
   ```

2. **Install React Native dependencies**
   ```bash
   cd ProgramATApp
   npm install
   ```

3. **Install iOS dependencies (iOS only)**
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

### Running the Application

1. **Start the backend server**
   ```bash
   cd backend
   python stream_server.py
   ```
   The server will start on port 8080.
   
   **Optional - Enable Intelligent GitHub Issue Creation:**
   To automatically create structured GitHub issues from voice transcriptions with AI parsing:
   ```bash
   export GITHUB_TOKEN="your_github_token_here"
   export GEMINI_API_KEY="your_gemini_api_key_here"  # For AI-powered template filling
   export GITHUB_REPO="owner/repo"  # Optional, defaults to program-at/ProgramAT
   python stream_server.py
   ```
   
   **Without AI (Simple Mode):**
   ```bash
   export GITHUB_TOKEN="your_github_token_here"
   python stream_server.py
   ```
   Will use basic parsing without AI-powered template filling.
   
   See [backend/README.md](backend/README.md) for more details.

2. **Start the React Native app**
   ```bash
   cd ProgramATApp
   npm start
   ```

3. **Run on your device**
   
   For iOS:
   ```bash
   npm run ios
   ```
   
   For Android:
   ```bash
   npm run android
   ```

### Configuration

To change the WebSocket server URL, edit `ProgramATApp/WebSocketService.ts`:

```typescript
private serverUrl: string = 'ws://YOUR_SERVER_IP:8080';
```

## Usage

1. **Connect to Server**: The app automatically connects on launch. Use the Connect/Disconnect button in the status bar to manage the connection.

2. **Start Camera Streaming**:
   - Tap "Start Camera" to activate the camera
   - Tap "Stream Frames" to begin sending frames to the server
   - Frames are captured at 4 FPS and sent as base64-encoded JPEG images

3. **Start Audio Transcription**:
   - Tap "Start Recording" to begin speech recognition
   - Speak into the device
   - Tap "Stop Recording" to finalize the transcription
   - The transcription is automatically sent to the server

See [STREAMING_GUIDE.md](ProgramATApp/STREAMING_GUIDE.md) for detailed usage instructions.

## Development

### Running Tests

```bash
cd ProgramATApp
npm test
```

### Linting

```bash
cd ProgramATApp
npm run lint
```

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

- **React Native 0.82.1** - Mobile application framework
- **TypeScript** - Type-safe development
- **react-native-vision-camera** - Camera access and frame capture
- **@react-native-voice/voice** - Speech recognition
- **react-native-fs** - File system operations
- **WebSocket** - Real-time bidirectional communication
- **Python WebSocket Server** - Backend frame and text processing

## Note

This application is configured to work **without Expo**. It uses bare React Native for direct access to native modules required for camera and audio streaming.

