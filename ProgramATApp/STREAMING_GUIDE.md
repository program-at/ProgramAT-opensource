# WebSocket Streaming Guide

This app now supports streaming camera frames and transcribed audio to a backend WebSocket server.

## Features

- **Real-time Frame Streaming**: Captures and streams camera frames at 2 FPS to the backend server
- **Audio Transcription Streaming**: Streams speech-to-text transcriptions in real-time
- **Connection Management**: Easy connect/disconnect controls with visual status indicators
- **Automatic Reconnection**: Attempts to reconnect automatically if connection is lost

## Server Configuration

The WebSocket server URL and other streaming settings are configured in `config.ts`:

```typescript
export const Config = {
  WEBSOCKET_SERVER_URL: 'ws://34.144.178.116:8080',
  FRAME_CAPTURE_INTERVAL_MS: 500, // 2 FPS
  FRAME_QUALITY_PRIORITIZATION: 'speed',
  IMAGE_QUALITY: 0.7, // 0.0 - 1.0 (lower = smaller files)
  MAX_RECONNECT_ATTEMPTS: 5,
  RECONNECT_DELAY_MS: 3000,
};
```

To change the server URL or other settings, edit `config.ts` before building the app.

### Performance Tuning

- **IMAGE_QUALITY**: Set between 0.6-0.8 for good balance between quality and file size
  - Higher values (0.9-1.0): Better quality, larger files, more bandwidth/memory
  - Lower values (0.5-0.6): Lower quality, smaller files, less bandwidth/memory
- **FRAME_CAPTURE_INTERVAL_MS**: Increase to reduce frame rate and bandwidth
  - 500ms = 2 FPS (default, good balance)
  - 1000ms = 1 FPS (lower bandwidth usage)
  - 250ms = 4 FPS (higher bandwidth, may cause performance issues)

## How to Use

### 1. Connect to Server

When the app starts, it will automatically attempt to connect to the WebSocket server. You can also:
- Manually connect using the "Connect" button in the status bar
- Disconnect using the "Disconnect" button

The status indicator shows:
- 🟢 Green dot = Connected
- 🔴 Red dot = Disconnected

### 2. Stream Camera Frames

1. Tap "Start Camera" to activate the camera
2. Once the camera is active, tap "Stream Frames" to begin streaming
3. Frames will be captured at 2 FPS and sent to the server
4. The frame counter shows how many frames have been sent
5. Tap "Stop Streaming" to pause frame streaming
6. Tap "Stop Camera" to deactivate the camera

### 3. Stream Audio Transcriptions

1. Tap "Start Recording" to begin speech recognition
2. Speak into the device microphone
3. When you tap "Stop Recording", the transcription will automatically be sent to the server
4. The transcript appears in the text area below

## Data Format

### Frame Messages

Frames are sent in the following format:

```json
{
  "frameNumber": 0,
  "timestamp": 1234567890,
  "data": {
    "base64Image": "data:image/jpeg;base64,/9j/4AAQ...",
    "width": 1920,
    "height": 1080
  }
}
```

### Text Messages

Transcriptions are sent in the following format:

```json
{
  "text": "Hello world",
  "timestamp": 1234567890
}
```

## Server Responses

The server sends acknowledgments for received frames and text:

```json
{
  "type": "ack",
  "frame_number": 0,
  "results": {
    "frame": {
      "status": "processed",
      "frame_shape": "1920x1080",
      "saved": "received_frames/frame_000000.jpg"
    },
    "text": {
      "status": "saved",
      "text_preview": "Hello world"
    }
  }
}
```

## Requirements

### Dependencies

The streaming functionality requires the following packages:
- `react-native-vision-camera` - For camera access
- `@react-native-voice/voice` - For speech recognition
- `react-native-fs` - For file operations

### Permissions

Ensure your app has the following permissions:
- Camera permission (for capturing frames)
- Microphone permission (for speech recognition)
- Network permission (for WebSocket connection)

## Troubleshooting

### Connection Issues

If you cannot connect to the server:
1. Verify the server is running at the configured URL
2. Check that port 8080 is open and accessible
3. Ensure your device has network connectivity
4. Check firewall settings

### Frame Streaming Issues

If frames are not being sent:
1. Ensure the WebSocket is connected (green status indicator)
2. Verify camera permissions are granted
3. Check that the camera is active before streaming
4. Look for error messages in the app UI
5. The app will display an error after 5 consecutive frame failures
6. Streaming will automatically stop after 20 consecutive failures
7. Check console logs for detailed error information

### Audio Transcription Issues

If transcriptions are not being sent:
1. Ensure the WebSocket is connected
2. Verify microphone permissions are granted
3. Make sure you stop recording to trigger the send
4. Check that speech was recognized (transcript appears)

## Backend Server

The backend server is located at `/backend/stream_server.py`. To run it:

```bash
cd backend
python stream_server.py
```

The server:
- Listens on port 8080
- Saves received frames to `received_frames/` directory
- Logs received text to `received_frames/received_texts.log`
- Broadcasts statistics every 5 seconds
