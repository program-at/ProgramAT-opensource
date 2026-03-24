# iOS Setup Guide

This guide provides instructions for setting up and running the speech-to-text and camera features on iOS devices.

## Prerequisites

- macOS computer (required for iOS development)
- Xcode installed (version 14.0 or later)
- CocoaPods installed
- iOS device or simulator

## Installation Steps

### 1. Install Dependencies

From the project root directory, install npm packages:

```bash
cd ProgramATApp
npm install
```

### 2. Install iOS Pods

Navigate to the iOS directory and install CocoaPods dependencies:

```bash
cd ios
bundle install
bundle exec pod install
```

If you don't have Bundler installed, you can install pods directly:

```bash
pod install
```

**Note**: The Podfile includes VisionCamera frame processor configuration which is required for the camera feature to work correctly. This is automatically applied during `pod install`.

### 3. Permissions Configuration

The required iOS permissions have already been added to `Info.plist`:

- **NSMicrophoneUsageDescription**: Required for audio recording
- **NSSpeechRecognitionUsageDescription**: Required for speech recognition
- **NSCameraUsageDescription**: Required for camera access

These permissions will prompt the user when the app first attempts to use these features.

### 4. Build and Run

You can build and run the app using either:

**Option A: Using React Native CLI**
```bash
npm run ios
```

**Option B: Using Xcode**
1. Open `ios/ProgramATApp.xcworkspace` in Xcode
2. Select your target device or simulator
3. Click the Run button (⌘R)

## Usage

### Speech-to-Text

1. Launch the app on your iOS device or simulator
2. The app opens on the "Speech to Text" tab by default
3. Grant microphone and speech recognition permissions when prompted
4. Tap "Start Recording" to begin speech recognition
5. Speak clearly into the device's microphone
6. Tap "Stop Recording" when finished
7. The transcribed text will appear in the transcript area

### Camera

1. Tap the "Camera" tab at the top of the screen
2. Grant camera permission when prompted
3. Tap "Start Camera" to activate the camera
4. The live camera preview will appear
5. Tap "Stop Camera" when finished

**Note**: Camera functionality requires a physical iOS device. The camera is not available in the iOS Simulator.

## Troubleshooting

### Permission Issues

If the app doesn't prompt for permissions:
1. Go to iOS Settings > Privacy & Security > Speech Recognition
2. Enable speech recognition for ProgramATApp
3. Go to iOS Settings > Privacy & Security > Microphone
4. Enable microphone access for ProgramATApp
5. Go to iOS Settings > Privacy & Security > Camera
6. Enable camera access for ProgramATApp

### Build Issues

If you encounter build errors:
1. Clean the build folder: In Xcode, Product > Clean Build Folder (⇧⌘K)
2. Delete `Pods` folder and `Podfile.lock`, then run `pod install` again
3. Delete `node_modules` and `package-lock.json`, then run `npm install` again

### Speech Recognition Not Working

- Ensure you have an active internet connection (iOS speech recognition requires network connectivity)
- Check that the device's language settings include English
- Verify that Siri is enabled in device settings (uses the same speech engine)

### Camera Not Working

- Ensure you are testing on a physical iOS device (camera not available in simulator)
- Check that camera permissions are granted in Settings
- Try restarting the app if the camera preview appears black

## Features

### Speech-to-Text
- **Real-time speech recognition**: Converts speech to text as you speak
- **iOS native integration**: Uses Apple's built-in Speech framework
- **Clean UI**: Simple, intuitive interface for recording and viewing transcripts
- **Error handling**: Displays clear error messages if issues occur

### Camera
- **Real-time camera preview**: Live camera feed display
- **Frame processing**: Built-in frame processor for future streaming
- **Socket-ready architecture**: Designed for streaming frames to server
- **iOS native integration**: Uses AVFoundation via VisionCamera library

## Technical Details

### Speech-to-Text
- **Library**: @react-native-voice/voice (v3.2.4)
- **Speech Engine**: iOS Speech Framework (Apple's native speech recognition)
- **Language**: English (US) - en-US
- **Platform**: iOS 13.0 or later

### Camera
- **Library**: react-native-vision-camera (v4.7.2)
- **Camera Engine**: AVFoundation via VisionCamera
- **Platform**: iOS 13.0 or later
- **Requirements**: Physical iOS device (not available in simulator)

## Notes

- The speech recognition feature requires an active internet connection on iOS
- Recognition quality depends on audio input quality and speaking clarity
- The app uses Apple's Speech framework, which may send audio data to Apple's servers for processing
- The camera feature works on physical iOS devices only (not in simulator)
- Both features are designed to support future streaming to a server via sockets
- Frame processor can be added in a future update for real-time frame streaming

## WebSocket Streaming (New)

The app now supports real-time streaming of camera frames and transcribed text to a backend WebSocket server.

### New Dependencies for Streaming

- **react-native-fs**: File system access for reading camera frames (v2.20.0)
- All dependencies are automatically linked via CocoaPods when you run `pod install`

### Streaming Setup

1. After installing dependencies, run:
   ```bash
   cd ios
   pod install
   ```

2. The new dependencies (react-native-fs) will be automatically linked

3. Build and run the app as usual

### Using Streaming Features

- The app connects to the WebSocket server on startup
- Connection status is shown at the top of the screen
- When camera is active, tap "Stream Frames" to begin sending frames to the server
- Transcriptions are automatically sent when you stop recording

See [STREAMING_GUIDE.md](STREAMING_GUIDE.md) for detailed streaming documentation.

## Additional Documentation

- See [CAMERA.md](CAMERA.md) for detailed camera feature documentation
- See [SPEECH_TO_TEXT.md](SPEECH_TO_TEXT.md) for detailed speech-to-text documentation
- See [STREAMING_GUIDE.md](STREAMING_GUIDE.md) for WebSocket streaming documentation
