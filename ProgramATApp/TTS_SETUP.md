# Text-to-Speech Setup Instructions

The app uses `react-native-tts` for voice feedback. After pulling this code, you need to install the native dependencies.

## iOS Setup

1. Navigate to the iOS directory:
```bash
cd ProgramATApp/ios
```

2. Install pods:
```bash
pod install
```

3. Build and run:
```bash
cd ..
npx react-native run-ios
```

## Android Setup

The module should auto-link on Android. Just build:
```bash
npx react-native run-android
```

## Troubleshooting

### iOS Error: "Objective-C bool isn't valid"

This error occurs when the native module isn't properly linked. To fix:

1. Clean the build:
```bash
cd ios
rm -rf Pods Podfile.lock build
pod install
cd ..
```

2. Clean React Native cache:
```bash
npx react-native start --reset-cache
```

3. Rebuild:
```bash
npx react-native run-ios
```

### TTS Not Working

The app will gracefully degrade if TTS isn't available:
- Feedback messages will still display in the UI
- Console will show: `[TTS] Would speak (TTS unavailable): <message>`
- No crashes or errors - the feature just won't speak aloud

This allows development to continue even if TTS setup is incomplete.
