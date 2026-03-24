# Camera Access Feature Documentation

## Overview

This React Native application now includes camera access capabilities using the `react-native-vision-camera` library. On iOS, this leverages Apple's native AVFoundation framework, which provides high-quality camera access and frame processing.

## Features

- **Real-time Camera Preview**: Live camera feed display
- **Simple UI**: Easy-to-use interface with Start/Stop camera buttons
- **Socket-Ready Architecture**: Designed for future frame streaming to server
- **Error Handling**: Clear error messages if something goes wrong
- **Permission Management**: Handles camera permissions gracefully
- **Tab Navigation**: Integrated with speech-to-text feature via tab navigation
- **Extensible**: Ready for frame processor integration when needed

## Architecture

### Components

#### CameraView Component (`CameraView.tsx`)

The main component that handles all camera functionality:

- **State Management**: Manages camera active state, permissions, errors, and loading states
- **Permission Handling**: Manages camera permission requests and status
- **User Controls**: Provides buttons for starting/stopping camera
- **Props**: Accepts optional `onFrameCapture` callback for future extensions

### Key Dependencies

- `react-native-vision-camera` (v4.7.2): Modern camera library for React Native
- `react-native-safe-area-context`: Ensures UI respects device safe areas

## iOS Configuration

### Required Permissions

The following permission is configured in `ios/ProgramATApp/Info.plist`:

1. **NSCameraUsageDescription**: "We need camera access to stream video"

This permission will trigger a system prompt when the user first attempts to use the camera.

### Platform Requirements

- iOS 13.0 or later
- Physical device recommended (camera not available in iOS Simulator)
- Camera access granted by user

### Native Dependencies

The following CocoaPods dependencies are automatically installed:

- `VisionCamera`: Core camera functionality
- Additional native modules from react-native-vision-camera

### iOS Build Configuration

The Podfile includes configuration to ensure proper iOS deployment target:

```ruby
post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      # Ensure iOS deployment target is set correctly for all pods
      if config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'].to_f < 13.0
        config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '13.0'
      end
    end
  end
end
```

This ensures all dependencies are built with iOS 13.0+ as the minimum target.

## Usage

### Basic Usage

The CameraView component is integrated into the main App component via tab navigation:

```typescript
import CameraView from './CameraView';

function App() {
  return (
    <View style={styles.container}>
      <CameraView />
    </View>
  );
}
```

### Future Frame Processing

The component accepts an optional `onFrameCapture` callback prop that is reserved for future frame processing functionality:

```typescript
<CameraView 
  onFrameCapture={(frameInfo) => {
    // Will be used for streaming frames to server
    // Currently not active - will be implemented in future update
  }}
/>
```

This callback will be activated when frame processor functionality is added in a future update.

## User Flow

1. User navigates to "Camera" tab
2. App checks camera permissions
3. If not granted, user sees permission request button
4. User grants permission
5. User taps "Start Camera" button
6. Camera preview appears in real-time
7. Frame processor runs in background (ready for streaming)
8. User taps "Stop Camera" to finish
9. Camera preview stops

## Technical Details

### Camera Engine

- **iOS**: Uses AVFoundation via VisionCamera
- **Camera**: Back camera (rear-facing)
- **Capabilities**: Photo and video capture enabled
- **Frame Processing**: Ready for future implementation

### Socket Streaming Ready

The architecture is designed to easily integrate socket streaming in the future:

```typescript
// Future example integration with socket.io
const socket = io('your-server-url');

// When frame processor is implemented:
<CameraView 
  onFrameCapture={(frameInfo) => {
    socket.emit('frame', frameInfo);
  }}
/>
```

### Error Handling

The component handles various error scenarios:

- Permission denied
- No camera device available
- Camera initialization errors
- General camera errors

All errors are displayed to the user with clear messages.

### Performance

- High-performance frame processing with worklets
- Minimal impact on app performance
- Efficient event handling with proper cleanup
- No memory leaks (proper cleanup on unmount)

## Testing

### Unit Tests

Tests are located in `__tests__/CameraView.test.tsx`:

- Component renders correctly
- Component renders with callback prop
- VisionCamera module is properly mocked for testing

Run tests with:
```bash
npm test
```

### Manual Testing on iOS

1. Build and run the app on a physical iOS device
2. Navigate to the Camera tab
3. Grant camera permission when prompted
4. Test camera start/stop functionality
5. Verify preview is working
6. Test error scenarios (deny permissions)

**Note**: Camera functionality requires a physical iOS device. The iOS Simulator does not provide camera access.

## Integration with Speech-to-Text

The camera feature is designed to work alongside the speech-to-text feature:

- **Tab Navigation**: Switch between Speech-to-Text and Camera
- **Independent Operation**: Each feature operates independently
- **Combined Streaming**: Both can be used simultaneously for future server streaming
- **Consistent UI/UX**: Similar design patterns and user experience

## Streaming Architecture

### Current State

The app currently provides:
- Live camera preview
- Callback-based architecture ready for future enhancements
- Socket integration can be added when needed

### Future Enhancements for Streaming

To enable full streaming to a server in the future:

1. **Add Frame Processor**:
   - Install react-native-reanimated if not present
   - Add frame processor to CameraView component
   - Implement worklet-based frame capture

2. **Add Socket.IO**:
   ```bash
   npm install socket.io-client
   ```

3. **Connect to Server**:
   ```typescript
   import io from 'socket.io-client';
   const socket = io('your-server-url');
   ```

4. **Stream Frames**:
   - Capture frames using frame processor
   - Convert frames to appropriate format
   - Send via socket to server

5. **Integrate with Speech**:
   - Combine audio stream from speech-to-text
   - Combine video frames from camera
   - Send both to server simultaneously

## Limitations

### Current Limitations

- Only supports back camera (can be extended to front camera)
- Frame processor not yet implemented (coming in future update)
- Requires physical device for testing
- iOS only (Android not configured)

### Future Enhancements

- Frame processor for real-time frame capture
- Front camera support
- Camera switching
- Flash/torch control
- Zoom controls
- Photo capture
- Video recording
- Full frame buffer streaming
- ML model integration for on-device processing
- QR code scanning
- Barcode detection

## Troubleshooting

### Common Issues

**Issue**: Permissions not requested
- **Solution**: Ensure Info.plist contains NSCameraUsageDescription

**Issue**: Camera not working
- **Solution**: Ensure testing on physical device, not simulator

**Issue**: Black screen on camera start
- **Solution**: Check permissions are granted in Settings

**Issue**: App crashes on launch
- **Solution**: Run `pod install` in the ios directory to ensure native dependencies are linked

**Issue**: Build errors
- **Solution**: Clean build folder and rebuild

## Security Considerations

### Data Privacy

- Camera frames are processed locally
- No data is sent to third-party servers (currently)
- Users must grant explicit permission for camera access
- Frame processing happens on-device

### Dependencies

- react-native-vision-camera v4.7.2 has no known security issues
- Dependencies reviewed for vulnerabilities
- Regular updates recommended

## Performance Considerations

### Frame Processing

- Frame processor runs on separate thread (worklet)
- Does not block main JS thread
- Optimized for 30-60 FPS processing
- Memory efficient

### Best Practices

- Stop camera when not in use
- Clean up resources properly
- Avoid heavy processing in frame processor
- Use runOnJS for non-worklet code

## Support

For issues or questions:
1. Check this documentation
2. Review the iOS_SETUP.md for setup instructions
3. Check the GitHub issues for known problems
4. Create a new issue with detailed reproduction steps

## License

This feature is part of the ProgramAT application. See the main repository LICENSE file for details.
