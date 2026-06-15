/**
 * TextInput Component
 * Provides text input capabilities with OS-level dictation support
 * Replaces voice-based SpeechToText component
 *
 * @format
 */

import React, { useState, useRef } from 'react';
import {
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ScrollView,
  TextInput as RNTextInput,
  Platform,
  Keyboard,
  TouchableWithoutFeedback,
  KeyboardAvoidingView,
  ActivityIndicator,
} from 'react-native';
import { useTheme } from './ThemeContext';
import WebSocketService from './WebSocketService';
import TextToSpeechService from './TextToSpeechService';
import VideoRecorderModal from './VideoRecorderModal';

interface TextInputProps {
  serverFeedback?: string;
  selectedIssue?: {number: number; title: string} | null;
  onNewIssue?: () => void;
  onBack?: () => void;
  showBackButton?: boolean;
  onViewPRs?: () => void;
  showPRsButton?: boolean;
}

export default function TextInputComponent({ 
  serverFeedback, 
  selectedIssue, 
  onNewIssue,
  onBack,
  showBackButton = false,
  onViewPRs,
  showPRsButton = false
}: TextInputProps) {
  const { theme } = useTheme();
  const [inputText, setInputText] = useState('');
  const [error, setError] = useState('');
  const inputRef = useRef<RNTextInput>(null);

  // Video attachment state (create mode only)
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [isVideoRecorderOpen, setIsVideoRecorderOpen] = useState(false);
  const [isSending, setIsSending] = useState(false);

  const isCreateMode = !selectedIssue;

  /** Convert the WebSocket URL to an HTTP base URL for REST endpoints. */
  const getHttpBaseUrl = (): string => {
    const wsUrl = WebSocketService.getServerUrl(); // e.g. 'ws://1.2.3.4:8081'
    return wsUrl.replace(/^wss?/, 'http');
  };

  const handleSubmitWithVideo = async () => {
    const baseUrl = getHttpBaseUrl();
    if (!baseUrl) {
      setError('Server URL not configured');
      return;
    }

    setIsSending(true);
    setError('');

    let shouldFallbackToWebSocket = false;

    try {
      const formData = new FormData();
      formData.append('metadata', JSON.stringify({ text: inputText.trim() }));
      if (videoUri) {
        formData.append('video', {
          uri: videoUri,
          type: 'video/mp4',
          name: 'recording.mp4',
        } as any);
      }

      const response = await fetch(`${baseUrl}/submit-creation`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (result.status === 'created') {
        const videoNote = result.video_summary ? '' : videoUri ? ' Video summarization was skipped.' : '';
        TextToSpeechService.speak(`Issue ${result.issue_number} created.${videoNote}`);
        setInputText('');
        setVideoUri(null);
        Keyboard.dismiss();
        return;
      }

      // Server returned an error — fall back to WebSocket so AI parsing still applies
      shouldFallbackToWebSocket = true;
    } catch {
      // Network error — fall back to WebSocket so AI parsing still applies
      shouldFallbackToWebSocket = true;
    } finally {
      setIsSending(false);
    }

    if (shouldFallbackToWebSocket) {
      if (!WebSocketService.isConnected()) {
        setError('Submission failed and server not connected. Please try again.');
        return;
      }
      TextToSpeechService.speak('Video submission failed. Processing your text.');
      setVideoUri(null);
      WebSocketService.sendIssueSelection('create');
      WebSocketService.sendText(inputText.trim());
      setInputText('');
      Keyboard.dismiss();
    }
  };

  const handleTextChange = (text: string) => {
    setInputText(text);
    setError('');
  };

  const handleSend = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text');
      return;
    }

    // Create mode + video attached → submit via HTTP multipart
    if (isCreateMode && videoUri) {
      await handleSubmitWithVideo();
      return;
    }

    if (!WebSocketService.isConnected()) {
      setError('Not connected to server');
      return;
    }

    setError('');

    // If switching to create mode or already in create mode, send mode first
    if (isCreateMode) {
      WebSocketService.sendIssueSelection('create');
    }

    // Send the text (without the prefix, backend now knows the mode)
    console.log('[TextInput] Sending text in', isCreateMode ? 'CREATE' : 'UPDATE', 'mode:', inputText);
    WebSocketService.sendText(inputText.trim());

    // Clear input after sending
    setInputText('');

    // Dismiss keyboard
    Keyboard.dismiss();
  };

  const handleNewIssue = () => {
    if (!WebSocketService.isConnected()) {
      setError('Not connected to server');
      return;
    }
    
    // Send mode switch to backend
    WebSocketService.sendIssueSelection('create');
    
    // Call the parent callback to update UI
    if (onNewIssue) {
      onNewIssue();
    }
    
    setError('');
  };

  const handleClear = () => {
    setInputText('');
    setError('');
    
    // Dismiss keyboard
    Keyboard.dismiss();
  };

  return (
    <KeyboardAvoidingView 
      style={[styles.container, { backgroundColor: theme.background }]}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 64 : 0}
      accessible={false}>
      <ScrollView 
        style={styles.innerContainer}
        keyboardShouldPersistTaps="handled"
        accessible={false}>
          {/* Back Button */}
          {showBackButton && onBack && (
            <View style={styles.backButtonContainer}>
              <TouchableOpacity
                style={[styles.backButton, { backgroundColor: theme.backgroundSecondary }]}
                onPress={onBack}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Back to PR list"
                accessibilityHint="Double tap to return to pull request list">
                <Text style={[styles.backButtonText, { color: theme.primary }]}>← Back to PRs</Text>
              </TouchableOpacity>
            </View>
          )}
          
          <View style={[styles.modeBar, { borderBottomColor: theme.border }]} accessible={false}>
            <View style={styles.modeInfo}>
              <Text style={[styles.modeLabel, { color: theme.textSecondary }]} accessible={false}>Mode:</Text>
              <View 
                style={[styles.modeBadge, isCreateMode ? styles.createModeBadge : styles.updateModeBadge, { backgroundColor: isCreateMode ? theme.success : theme.primary }]}
                accessible={true}
                accessibilityRole="text"
                accessibilityLabel={isCreateMode ? 'Create new mode' : `Update mode. ${selectedIssue?.title}`}>
                <Text style={styles.modeBadgeText} accessible={false} numberOfLines={1} ellipsizeMode="tail">
                  {isCreateMode ? '✨ Create New' : `🔄 ${selectedIssue?.title}`}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.header}>
            <View style={styles.headerLeft}>
              <Text 
                style={[styles.headerText, { color: theme.text }]}
                accessible={true}
                accessibilityRole="header"
                accessibilityLabel={isCreateMode ? 'New Visual AT Tool' : `Update ${selectedIssue?.title}`}
                numberOfLines={2}
                ellipsizeMode="tail">
                {isCreateMode ? 'New Visual AT Tool' : `Update ${selectedIssue?.title}`}
              </Text>
              <Text style={[styles.hintText, { color: theme.textTertiary }]} accessible={false}>
                {Platform.OS === 'ios' 
                  ? 'Tap outside to close keyboard • Mic icon for dictation' 
                  : 'Tap outside to close keyboard • Voice button for dictation'}
              </Text>
            </View>
            {serverFeedback !== '' && (
              <View style={[styles.feedbackBadge, { backgroundColor: theme.success }]}>
                <Text style={styles.feedbackBadgeText} numberOfLines={1}>
                  {serverFeedback}
                </Text>
              </View>
            )}
          </View>

          <View style={styles.inputSection} accessible={false}>
            <RNTextInput
              ref={inputRef}
              style={[styles.textInput, { 
                backgroundColor: theme.inputBackground, 
                borderColor: theme.inputBorder, 
                color: theme.text 
              }]}
              placeholder="Describe the visual assistive technology you'd like..."
              placeholderTextColor={theme.inputPlaceholder}
              multiline
              numberOfLines={8}
              value={inputText}
              onChangeText={handleTextChange}
              autoCorrect={true}
              autoCapitalize="sentences"
              textAlignVertical="top"
              returnKeyType="default"
              blurOnSubmit={false}
              scrollEnabled={true}
              accessible={true}
              accessibilityLabel="Text input"
              accessibilityHint="Type your message here. Use context menu to copy or paste"
              editable={true}
              contextMenuHidden={false}
              selectTextOnFocus={false}
              selection={undefined}
            />

            <View style={styles.charCount} accessible={false}>
              <Text style={[styles.charCountText, { color: theme.textTertiary }]} accessible={false}>
                {inputText.length} characters
              </Text>
            </View>
          </View>

          {error !== '' && (
            <View style={styles.errorContainer}>
              <Text style={[styles.errorText, { color: theme.error }]}>{error}</Text>
            </View>
          )}

          {/* Video attachment row — create mode only */}
          {isCreateMode && (
            <View style={styles.videoRow}>
              {videoUri ? (
                <>
                  <View style={[styles.videoAttachedBadge, { backgroundColor: theme.backgroundSecondary, borderColor: theme.border }]}>
                    <Text style={[styles.videoAttachedText, { color: theme.text }]}>📹 Video attached</Text>
                  </View>
                  <TouchableOpacity
                    onPress={() => setIsVideoRecorderOpen(true)}
                    accessible={true}
                    accessibilityRole="button"
                    accessibilityLabel="Re-record video">
                    <Text style={[styles.videoActionText, { color: theme.primary }]}>Re-record</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    onPress={() => setVideoUri(null)}
                    accessible={true}
                    accessibilityRole="button"
                    accessibilityLabel="Remove video attachment">
                    <Text style={[styles.videoActionText, { color: theme.error }]}>Remove</Text>
                  </TouchableOpacity>
                </>
              ) : (
                <TouchableOpacity
                  style={[styles.addVideoButton, { borderColor: theme.border, backgroundColor: theme.backgroundSecondary }]}
                  onPress={() => setIsVideoRecorderOpen(true)}
                  accessible={true}
                  accessibilityRole="button"
                  accessibilityLabel="Add video recording"
                  accessibilityHint="Opens camera to record a video to attach to this submission">
                  <Text style={[styles.addVideoText, { color: theme.text }]}>📹  Add Video</Text>
                </TouchableOpacity>
              )}
            </View>
          )}

          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={[styles.button, styles.clearButton, { backgroundColor: theme.backgroundSecondary, borderColor: theme.border }]}
              onPress={handleClear}
              disabled={!inputText.trim()}
              accessible={true}
              accessibilityLabel="Clear text"
              accessibilityHint="Clears all text from the input field"
              accessibilityRole="button">
              <Text style={[styles.buttonText, { color: theme.text }]}>Clear</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button, 
                styles.sendButton, 
                { backgroundColor: theme.primary }, 
                (!inputText.trim() || isSending) && styles.buttonDisabled
              ]}
              onPress={handleSend}
              disabled={!inputText.trim() || isSending}
              accessible={true}
              accessibilityLabel={isSending ? 'Submitting…' : (isCreateMode && videoUri ? 'Submit with video' : 'Send text')}
              accessibilityHint="Sends the text to the server"
              accessibilityRole="button"
              accessibilityState={{ disabled: !inputText.trim() || isSending }}>
              {isSending
                ? <ActivityIndicator size="small" color="#fff" />
                : <Text style={[styles.buttonText, styles.sendButtonText]}>
                    {isCreateMode && videoUri ? 'Submit' : 'Send'}
                  </Text>
              }
            </TouchableOpacity>
          </View>
        </ScrollView>

        {/* Video recorder overlay — create mode only */}
        <VideoRecorderModal
          visible={isVideoRecorderOpen}
          onVideoRecorded={(path) => {
            setVideoUri(path);
            setIsVideoRecorderOpen(false);
          }}
          onCancel={() => setIsVideoRecorderOpen(false)}
        />
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  innerContainer: {
    flex: 1,
    padding: 16,
  },
  backButtonContainer: {
    marginBottom: 12,
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    alignSelf: 'flex-start',
  },
  backButtonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  modeBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
    paddingBottom: 12,
    borderBottomWidth: 2,
  },
  modeInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    flex: 1,
    minWidth: 0, // Allow shrinking
  },
  modeLabel: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    flexShrink: 0, // Don't shrink the label
  },
  modeBadge: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 12,
    flexShrink: 1, // Allow badge to shrink
    maxWidth: '80%', // Prevent taking full width
  },
  createModeBadge: {
  },
  updateModeBadge: {
  },
  modeBadgeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#fff',
  },
  modeButtonsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  viewPRsButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  viewPRsButtonText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#fff',
  },
  switchModeButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  switchModeButtonText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#fff',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  headerLeft: {
    flex: 1,
  },
  headerText: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  hintText: {
    fontSize: 12,
    fontStyle: 'italic',
  },
  feedbackBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    borderLeftWidth: 2,
    borderLeftColor: '#fff',
    maxWidth: 150,
    marginLeft: 8,
  },
  feedbackBadgeText: {
    fontSize: 11,
    color: '#fff',
    fontWeight: '500',
  },
  inputSection: {
    flex: 1,
    borderWidth: 1,
    borderRadius: 8,
    marginBottom: 12,
  },
  scrollView: {
    flex: 1,
  },
  textInput: {
    flex: 1,
    padding: 12,
    fontSize: 16,
    minHeight: 150,
  },
  charCount: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderTopWidth: 1,
  },
  charCountText: {
    fontSize: 11,
    textAlign: 'right',
  },
  errorContainer: {
    backgroundColor: '#ffebee',
    padding: 10,
    borderRadius: 6,
    marginBottom: 12,
  },
  errorText: {
    fontSize: 13,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  button: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 8,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  clearButton: {
  },
  sendButton: {
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  sendButtonText: {
    color: '#fff',
  },
  // Video attachment row
  videoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 12,
  },
  addVideoButton: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  addVideoText: {
    fontSize: 15,
    fontWeight: '600',
  },
  videoAttachedBadge: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 8,
    borderWidth: 1,
    alignItems: 'center',
  },
  videoAttachedText: {
    fontSize: 14,
    fontWeight: '600',
  },
  videoActionText: {
    fontSize: 14,
    fontWeight: '600',
    paddingVertical: 4,
  },
});
