# Accessibility Features for ProgramAT iOS App

## Overview
Comprehensive screen reader accessibility has been added to the ProgramAT iOS app to support VoiceOver users. All interactive elements, status indicators, and dynamic content are now properly labeled and announced.

## Features Added

### 1. **App.tsx - Main Application**

#### Connection Status
- **Status indicator**: Announces "Server connection status: Connected" or "Server connection status: Disconnected"
- **Connect/Disconnect button**: 
  - Label: "Connect to server" / "Disconnect from server"
  - Hint: Explains what will happen when tapped
  - Role: Properly identified as a button

#### Error Handling
- **Error banner**: Uses `accessibilityRole="alert"` and `accessibilityLiveRegion="polite"` to announce errors immediately
- Announces: "Error: [error message]"

#### Section Labels
- Camera and Speech sections are properly labeled for context

---

### 2. **CameraView.tsx - Camera Controls**

#### Permission Request
- **Grant Permission button**:
  - Label: "Grant camera permission"
  - Hint: "Double tap to open settings and grant camera access"

#### Camera Controls
- **Start Camera button**:
  - Label: "Start camera"
  - Hint: "Double tap to activate the camera preview"
  - State: Announces when disabled (loading)

- **Stop Camera button**:
  - Label: "Stop camera"
  - Hint: "Double tap to deactivate the camera"

- **Stream Frames button**:
  - Label: "Stream frames"
  - Hint: "Double tap to start streaming camera frames to the server"

- **Stop Streaming button**:
  - Label: "Stop streaming"
  - Hint: "Double tap to stop streaming camera frames"

#### Status Indicators
- **Camera status**: Dynamically announces camera state and frame count
  - Example: "Camera active, streaming, 42 frames sent"
  - Uses `accessibilityLiveRegion="polite"` for real-time updates

- **Error messages**: Announced with `accessibilityRole="alert"` and `accessibilityLiveRegion="assertive"` for immediate attention

#### Camera Preview
- **Live preview**: Labeled as "Camera preview" with hint "Live camera feed displaying what the camera sees"
- **Placeholder**: When camera is off, placeholder text is accessible

---

### 3. **SpeechToText.tsx - Speech Recognition**

#### Recording Controls
- **Start Recording button**:
  - Label: "Start recording"
  - Hint: "Double tap to begin speech recognition"
  - State: Announces when disabled (loading)

- **Stop Recording button**:
  - Label: "Stop recording"
  - Hint: "Double tap to stop speech recognition"

- **Clear button**:
  - Label: "Clear transcript"
  - Hint: "Double tap to clear the current transcript"

#### Status Indicators
- **Recording status**: 
  - Label: "Recording in progress"
  - Uses `accessibilityLiveRegion="polite"` for announcement

- **Error messages**: 
  - Announced with `accessibilityRole="alert"` and `accessibilityLiveRegion="assertive"`

- **Server feedback**: 
  - Announced with `accessibilityRole="alert"` and `accessibilityLiveRegion="polite"`
  - Example: "Server feedback: I need steps to reproduce"

#### Transcript Display
- **Transcript text**: 
  - Uses `accessibilityLiveRegion="polite"` to announce updates as text is transcribed
  - Provides fallback message when empty: "No transcript yet. Press start recording to begin."

---

## Accessibility Properties Used

### `accessible`
- Set to `true` for interactive and informative elements
- Set to `false` for decorative containers to reduce noise

### `accessibilityRole`
- **"button"**: For all touchable controls
- **"header"**: For section titles
- **"text"**: For informational text
- **"alert"**: For errors and important notifications

### `accessibilityLabel`
- Clear, concise description of the element
- Includes dynamic state information where relevant
- Example: "Camera active, streaming, 42 frames sent"

### `accessibilityHint`
- Explains what will happen when the user interacts
- Uses action-oriented language: "Double tap to..."
- Only provided for interactive elements

### `accessibilityState`
- **disabled**: Indicates when buttons are disabled (e.g., during loading)

### `accessibilityLiveRegion`
- **"polite"**: For status updates that don't require immediate attention
- **"assertive"**: For errors that need immediate user awareness

---

## Testing with VoiceOver

### How to Enable VoiceOver on iOS
1. Go to **Settings > Accessibility > VoiceOver**
2. Toggle VoiceOver on
3. Or use Siri: "Hey Siri, turn on VoiceOver"

### Gestures
- **Single tap**: Select an item
- **Double tap**: Activate the selected item
- **Swipe right**: Move to next element
- **Swipe left**: Move to previous element
- **Two-finger swipe down**: Read all content from current position

### Expected Behavior
1. All buttons announce their purpose and provide hints
2. Status changes (connection, recording, streaming) are announced automatically
3. Error messages interrupt and are announced immediately
4. Frame counts and transcript updates are announced politely without interruption
5. No decorative elements (dots, containers) clutter the experience
6. All interactive elements can be discovered and activated via VoiceOver

---

## Best Practices Followed

1. ✅ **Meaningful labels**: Every interactive element has a clear, descriptive label
2. ✅ **Action hints**: Users understand what will happen before they tap
3. ✅ **Live regions**: Dynamic content updates are announced appropriately
4. ✅ **Proper roles**: Screen readers understand the purpose of each element
5. ✅ **State management**: Disabled states are properly announced
6. ✅ **Error handling**: Errors get immediate attention with "assertive" live region
7. ✅ **Reduced noise**: Decorative elements are hidden from screen readers
8. ✅ **Contextual information**: Status messages include relevant details (e.g., frame count)

---

## Future Improvements

Consider these enhancements:
- Add accessibility actions for alternative interaction methods
- Implement custom rotor items for quick navigation
- Add haptic feedback for important state changes
- Support Dynamic Type for text scaling
- Test with other assistive technologies (Switch Control, Voice Control)

---

## Resources

- [Apple VoiceOver Documentation](https://developer.apple.com/documentation/accessibility/voiceover)
- [React Native Accessibility API](https://reactnative.dev/docs/accessibility)
- [iOS Accessibility Guidelines](https://developer.apple.com/accessibility/ios/)

---

**Last Updated**: November 25, 2025
