# Speech-to-Text Feature Documentation

## Overview

This React Native application now includes speech-to-text capabilities using the `@react-native-voice/voice` library. On iOS, this leverages Apple's native Speech framework, which provides high-quality speech recognition.

## Features

- **Real-time Speech Recognition**: Converts spoken words to text as you speak
- **Simple UI**: Easy-to-use interface with Start/Stop recording buttons
- **Live Transcript Display**: See your words appear in real-time
- **Error Handling**: Clear error messages if something goes wrong
- **Visual Feedback**: Recording indicator shows when the app is listening
- **Clear Function**: Easily clear the transcript to start fresh

## Architecture

### Components

#### SpeechToText Component (`SpeechToText.tsx`)

The main component that handles all speech recognition functionality:

- **State Management**: Manages recording state, transcript, errors, and loading states
- **Event Listeners**: Handles speech recognition events (start, end, results, errors)
- **User Controls**: Provides buttons for starting/stopping recording and clearing transcripts
- **Props**: Accepts optional `onTranscriptChange` callback for parent components

### Key Dependencies

- `@react-native-voice/voice` (v3.2.4): Cross-platform speech recognition library
- `react-native-safe-area-context`: Ensures UI respects device safe areas

## iOS Configuration

### Required Permissions

The following permissions are configured in `ios/ProgramATApp/Info.plist`:

1. **NSMicrophoneUsageDescription**: "We need microphone access to record audio"
2. **NSSpeechRecognitionUsageDescription**: "We need speech recognition to convert your voice to text"

These permissions will trigger system prompts when the user first attempts to use speech recognition.

### Platform Requirements

- iOS 13.0 or later
- Active internet connection (iOS speech recognition requires network connectivity)
- Microphone access granted by user
- Speech recognition access granted by user

## Usage

### Basic Usage

The SpeechToText component is integrated into the main App component:

```typescript
import SpeechToText from './SpeechToText';

function App() {
  return (
    <View style={styles.container}>
      <SpeechToText />
    </View>
  );
}
```

### With Callback

You can pass a callback to receive transcript updates:

```typescript
<SpeechToText 
  onTranscriptChange={(transcript) => {
    console.log('New transcript:', transcript);
  }}
/>
```

## User Flow

1. User taps "Start Recording" button
2. App requests microphone and speech recognition permissions (first time only)
3. User grants permissions
4. Recording begins - a red indicator appears
5. User speaks into the device microphone
6. Speech is converted to text in real-time
7. Transcript appears in the display area
8. User taps "Stop Recording" to finish
9. User can tap "Clear" to reset the transcript

## Technical Details

### Speech Recognition Engine

- **iOS**: Uses Apple's Speech framework
- **Language**: English (US) - en-US
- **Network**: Requires internet connection for processing
- **Privacy**: Audio may be sent to Apple's servers for processing

### Error Handling

The component handles various error scenarios:

- Permission denied
- Network unavailable
- Speech recognition service unavailable
- General recognition errors

All errors are displayed to the user with clear messages.

### Performance

- Minimal impact on app performance
- Efficient event handling with proper cleanup
- No memory leaks (proper listener cleanup on unmount)

## Testing

### Unit Tests

Tests are located in `__tests__/SpeechToText.test.tsx`:

- Component renders correctly
- Component renders with callback prop
- Voice module is properly mocked for testing

Run tests with:
```bash
npm test
```

### Manual Testing on iOS

1. Build and run the app on a physical iOS device or simulator
2. Grant required permissions when prompted
3. Test basic recording functionality
4. Test error scenarios (deny permissions, turn off internet)
5. Test UI responsiveness and feedback

## Limitations

### Current Limitations

- Only supports English (US) language
- Requires internet connection on iOS
- Dependent on Apple's speech recognition service availability
- May have rate limits imposed by Apple

### Future Enhancements

- Support for multiple languages
- Offline speech recognition (where available)
- Custom vocabulary/domain-specific recognition
- Punctuation and formatting improvements
- Continuous recognition mode
- Voice command integration

## Troubleshooting

### Common Issues

**Issue**: Permissions not requested
- **Solution**: Ensure Info.plist contains the required permission keys

**Issue**: Speech recognition not working
- **Solution**: Check internet connection and verify permissions are granted

**Issue**: Poor recognition quality
- **Solution**: Ensure microphone is not obstructed and speak clearly

**Issue**: App crashes on launch
- **Solution**: Run `pod install` in the ios directory to ensure native dependencies are linked

## Security Considerations

### Data Privacy

- Audio data may be sent to Apple's servers for speech processing
- Transcripts are stored only in app memory (not persisted)
- No data is sent to third-party servers
- Users must grant explicit permission for microphone and speech recognition

### Dependencies

- All dependencies have been checked for known vulnerabilities
- @react-native-voice/voice v3.2.4 has no known security issues
- Transitive dependencies in build tools have some vulnerabilities but do not affect runtime

## Support

For issues or questions:
1. Check the iOS_SETUP.md for detailed setup instructions
2. Review this documentation
3. Check the GitHub issues for known problems
4. Create a new issue with detailed reproduction steps

## License

This feature is part of the ProgramAT application. See the main repository LICENSE file for details.
