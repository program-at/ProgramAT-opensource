# Migration Guide: Voice-Based to Text-Based Input

## Overview

This guide documents the migration from a voice-based speech recognition system to a text input system with OS-level dictation support.

## What Changed

### Removed
- **SpeechToText.tsx** component (voice-based)
- Dependency on `@react-native-voice/voice` library
- Voice recording state management
- Voice permission handling
- Complex partial results streaming

### Added
- **TextInput.tsx** component (text-based)
- Standard React Native TextInput
- OS keyboard dictation support
- Simpler state management

### Modified
- **App.tsx**: Import and usage of TextInput instead of SpeechToText
- Component styling (renamed `speechSection` to `inputSection`)

## Frontend Changes

### 1. New TextInput Component

**Location**: `/ProgramATApp/TextInput.tsx`

**Key Features**:
- Multi-line text input with auto-expanding height
- Character counter
- Auto-send complete sentences
- Delta tracking (avoid resending)
- Clear and Send buttons
- Server feedback display
- Platform-specific dictation hints

**Usage**:
```tsx
import TextInput from './TextInput';

<TextInput serverFeedback={spokenFeedback} />
```

### 2. App.tsx Changes

**Before**:
```tsx
import SpeechToText from './SpeechToText';

<View style={styles.speechSection}>
  <SpeechToText serverFeedback={spokenFeedback} />
</View>
```

**After**:
```tsx
import TextInput from './TextInput';

<View style={styles.inputSection}>
  <TextInput serverFeedback={spokenFeedback} />
</View>
```

### 3. Behavior Changes

#### Input Method
- **Before**: Tap "Start Recording" → speak → auto-send on sentence completion
- **After**: Type in text box OR use keyboard's microphone icon → auto-send on sentence completion OR tap "Send"

#### Dictation
- **Before**: Custom voice recognition with @react-native-voice/voice
- **After**: OS-level dictation via keyboard (iOS microphone icon, Android voice button)

#### Text Streaming
- **Before**: Partial results streamed as user speaks
- **After**: Complete sentences auto-sent as user types/dictates

## Backend Changes

**No changes required!** The backend already processes text via WebSocket and doesn't care whether it came from voice recognition or text input.

The backend continues to:
- Receive text via `data.text` or `data.caption`
- Track deltas to avoid reprocessing
- Detect pauses (5 seconds)
- Parse with AI
- Create/update GitHub issues

## User Experience Changes

### Before (Voice-Based)

1. User taps "Start Recording" button
2. Speaks into microphone
3. Sees partial transcription appear in real-time
4. System auto-sends complete sentences
5. User taps "Stop Recording" when done
6. Remaining text sent to server

### After (Text-Based)

1. User taps in text input box
2. Either types manually OR taps keyboard microphone icon to dictate
3. Sees text appear as typed/dictated
4. System auto-sends complete sentences (ending with . ! ?)
5. User taps "Send" button when done OR just waits for auto-send
6. Can mix typing and dictation seamlessly

### Advantages

✅ **Simpler**: No custom voice recognition library
✅ **Flexible**: User can type, dictate, or mix both
✅ **Accessible**: Works in noisy environments
✅ **Familiar**: Standard text input UX
✅ **Editable**: User can edit before sending
✅ **Cross-platform**: Consistent across iOS/Android

### Trade-offs

⚠️ **Manual activation**: User must tap keyboard mic (vs auto-start)
⚠️ **Two-step dictation**: Tap mic → speak (vs single button)

## Migration Steps

### For Developers

1. **Install new component**:
   - File already created: `ProgramATApp/TextInput.tsx`

2. **Update App.tsx**:
   - Changes already applied
   - Import changed from SpeechToText to TextInput
   - Component usage updated

3. **Remove old dependencies** (optional):
   ```bash
   cd ProgramATApp
   npm uninstall @react-native-voice/voice
   ```

4. **Test the changes**:
   ```bash
   npm run ios    # or npm run android
   ```

5. **Update documentation**:
   - README files updated
   - Architecture documentation created

### For Users

1. **Update the app** (when deployed)

2. **New workflow**:
   - Instead of "Start Recording" button, you'll see a text input box
   - Type your issue description, or tap the microphone on your keyboard to dictate
   - The app will auto-send complete sentences
   - Or tap "Send" when you're done

3. **Tips**:
   - **iOS**: Tap the microphone icon on the keyboard to dictate
   - **Android**: Tap the voice input button on the keyboard
   - You can edit text before it's sent
   - Complete sentences (ending with . ! ?) are sent automatically
   - Tap "Clear" to start over

## Testing Checklist

- [ ] Text input appears and is editable
- [ ] Character counter updates
- [ ] Auto-send works for complete sentences
- [ ] Send button sends remaining text
- [ ] Clear button resets input
- [ ] Server feedback displays correctly
- [ ] TTS speaks server feedback
- [ ] WebSocket communication works
- [ ] Issue creation flow works end-to-end
- [ ] Issue update mode works
- [ ] iOS dictation works (tap keyboard mic)
- [ ] Android dictation works (voice input button)
- [ ] Can mix typing and dictation

## Rollback Plan

If you need to revert to voice-based input:

1. **Restore SpeechToText component**:
   ```bash
   git checkout HEAD~1 -- ProgramATApp/SpeechToText.tsx
   ```

2. **Revert App.tsx changes**:
   ```tsx
   import SpeechToText from './SpeechToText';
   
   <View style={styles.speechSection}>
     <SpeechToText serverFeedback={spokenFeedback} />
   </View>
   ```

3. **Reinstall voice library**:
   ```bash
   npm install @react-native-voice/voice
   ```

## Known Issues

None currently. Please report issues via GitHub.

## FAQ

**Q: Can I still use voice input?**
A: Yes! Use your keyboard's built-in dictation feature (microphone icon on iOS, voice button on Android).

**Q: Why make this change?**
A: Simpler architecture, more flexibility (type or dictate), better accessibility, and easier maintenance.

**Q: Does this affect the backend?**
A: No, the backend works the same way. It receives text and doesn't care how it was input.

**Q: Can I still get voice feedback?**
A: Yes, the app still speaks server feedback messages via text-to-speech.

**Q: What if my keyboard doesn't have dictation?**
A: You can type manually. Most modern iOS and Android keyboards include dictation features.

**Q: Will this work offline?**
A: Typing works offline, but you need internet to send to the server (same as before).

---

**Migration Date**: December 18, 2025
**Version**: 2.0
