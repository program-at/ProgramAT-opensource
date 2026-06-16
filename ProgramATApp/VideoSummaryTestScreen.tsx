/**
 * VideoSummaryTestScreen
 * Developer testing screen: record a video OR pick one from the library, send it
 * to the server, and display the Gemini summary — without creating a GitHub issue.
 * Accessed from Settings.
 */

import React, { useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { launchImageLibrary } from 'react-native-image-picker';
import { useTheme } from './ThemeContext';
import WebSocketService from './WebSocketService';
import VideoRecorderModal from './VideoRecorderModal';

interface VideoSummaryTestScreenProps {
  onBack: () => void;
}

export default function VideoSummaryTestScreen({ onBack }: VideoSummaryTestScreenProps) {
  const { theme } = useTheme();
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [videoSource, setVideoSource] = useState<'recorded' | 'library' | null>(null);
  const [isRecorderOpen, setIsRecorderOpen] = useState(false);
  const [isPickingFromLibrary, setIsPickingFromLibrary] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const clearVideo = () => {
    setVideoUri(null);
    setVideoSource(null);
    setSummary(null);
    setErrorMsg(null);
  };

  const handlePickFromLibrary = async () => {
    setIsPickingFromLibrary(true);
    try {
      const result = await launchImageLibrary({
        mediaType: 'video',
        videoQuality: 'high',
      });
      if (result.didCancel || !result.assets?.length) return;
      const asset = result.assets[0];
      if (asset.uri) {
        setVideoUri(asset.uri);
        setVideoSource('library');
        setSummary(null);
        setErrorMsg(null);
      }
    } finally {
      setIsPickingFromLibrary(false);
    }
  };

  const getHttpBase = (): string =>
    WebSocketService.getServerUrl().replace(/^wss?/, 'http');

  const handleTest = async () => {
    if (!videoUri) return;

    const baseUrl = getHttpBase();
    if (!baseUrl) {
      setErrorMsg('Server URL not configured — connect in Settings first.');
      return;
    }

    setIsTesting(true);
    setSummary(null);
    setErrorMsg(null);

    try {
      const formData = new FormData();
      formData.append('video', {
        uri: videoUri,
        type: 'video/mp4',
        name: 'test.mp4',
      } as any);

      const response = await fetch(`${baseUrl}/test-video-summary`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (result.status === 'ok') {
        setSummary(result.summary || '(No summary returned)');
      } else {
        setErrorMsg(result.error ?? 'Server returned an error.');
      }
    } catch {
      setErrorMsg('Could not reach the server. Check your connection.');
    } finally {
      setIsTesting(false);
    }
  };

  return (
    <SafeAreaView
      style={[styles.container, { backgroundColor: theme.background }]}
      edges={['bottom']}>
      {/* Header */}
      <View style={[styles.header, { borderBottomColor: theme.border }]}>
        <TouchableOpacity
          onPress={onBack}
          style={styles.backButton}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel="Back to Settings">
          <Text style={[styles.backText, { color: theme.primary }]}>‹ Settings</Text>
        </TouchableOpacity>
        <Text style={[styles.title, { color: theme.text }]}>Video Summary Test</Text>
        <Text style={[styles.subtitle, { color: theme.textSecondary }]}>
          Record a video and verify the Gemini summary without creating an issue.
        </Text>
      </View>

      <ScrollView style={styles.body} contentContainerStyle={styles.bodyContent}>

        {/* Step 1 — Add Video */}
        <View style={[styles.card, { backgroundColor: theme.card, borderColor: theme.border }]}>
          <Text style={[styles.stepLabel, { color: theme.textSecondary }]}>Step 1 — Add Video</Text>

          {isPickingFromLibrary ? (
            <View style={styles.pickingRow}>
              <ActivityIndicator size="small" color={theme.primary} />
              <Text style={[styles.pickingText, { color: theme.textSecondary }]}>
                Attaching video from library…
              </Text>
            </View>
          ) : videoUri ? (
            <View style={styles.videoReadyRow}>
              <Text style={[styles.videoReadyText, { color: theme.text }]}>
                {videoSource === 'library' ? '📂  Library video ready' : '📹  Recorded video ready'}
              </Text>
              <TouchableOpacity
                onPress={() => setIsRecorderOpen(true)}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Re-record video">
                <Text style={[styles.actionLink, { color: theme.primary }]}>Re-record</Text>
              </TouchableOpacity>
              <TouchableOpacity
                onPress={handlePickFromLibrary}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Pick different video from library">
                <Text style={[styles.actionLink, { color: theme.primary }]}>Library</Text>
              </TouchableOpacity>
              <TouchableOpacity
                onPress={clearVideo}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Remove video">
                <Text style={[styles.actionLink, { color: theme.error }]}>Remove</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <View style={styles.sourceRow}>
              <TouchableOpacity
                style={[styles.sourceButton, { backgroundColor: theme.primary }]}
                onPress={() => setIsRecorderOpen(true)}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Record a new video"
                accessibilityHint="Opens the camera to record a test video">
                <Text style={styles.sourceButtonText}>📹  Record</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.sourceButton, { backgroundColor: theme.backgroundSecondary, borderWidth: 1, borderColor: theme.border }]}
                onPress={handlePickFromLibrary}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Choose a video from your photo library"
                accessibilityHint="Opens your photo library to select an existing video">
                <Text style={[styles.sourceButtonText, { color: theme.text }]}>📂  Choose from Library</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>

        {/* Step 2 — Test */}
        <View style={[styles.card, { backgroundColor: theme.card, borderColor: theme.border }]}>
          <Text style={[styles.stepLabel, { color: theme.textSecondary }]}>Step 2 — Generate Summary</Text>
          <TouchableOpacity
            style={[
              styles.testButton,
              { backgroundColor: theme.success },
              (!videoUri || isTesting) && styles.buttonDisabled,
            ]}
            onPress={handleTest}
            disabled={!videoUri || isTesting}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel={isTesting ? 'Generating summary…' : 'Send to Gemini and summarize'}
            accessibilityState={{ disabled: !videoUri || isTesting }}>
            {isTesting ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <Text style={styles.testButtonText}>Send to Gemini</Text>
            )}
          </TouchableOpacity>
        </View>

        {/* Result */}
        {(summary !== null || errorMsg !== null) && (
          <View style={[
            styles.card,
            styles.resultCard,
            {
              backgroundColor: theme.card,
              borderColor: errorMsg ? theme.error : theme.success,
            },
          ]}>
            <Text style={[styles.stepLabel, { color: theme.textSecondary }]}>
              {errorMsg ? 'Error' : 'Summary'}
            </Text>
            <Text
              style={[
                styles.resultText,
                { color: errorMsg ? theme.error : theme.text },
              ]}
              selectable>
              {errorMsg ?? summary}
            </Text>
          </View>
        )}
      </ScrollView>

      {/* Camera overlay */}
      <VideoRecorderModal
        visible={isRecorderOpen}
        onVideoRecorded={path => {
          setVideoUri(path);
          setIsRecorderOpen(false);
          setSummary(null);
          setErrorMsg(null);
        }}
        onCancel={() => setIsRecorderOpen(false)}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  // Header
  header: {
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
  },
  backButton: {
    marginBottom: 8,
    alignSelf: 'flex-start',
  },
  backText: {
    fontSize: 17,
    fontWeight: '500',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 13,
  },
  // Body
  body: { flex: 1 },
  bodyContent: {
    padding: 20,
    gap: 16,
  },
  card: {
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  resultCard: {
    borderWidth: 2,
  },
  stepLabel: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 12,
  },
  // Video row
  videoReadyRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  videoReadyText: {
    flex: 1,
    fontSize: 15,
    fontWeight: '600',
  },
  actionLink: {
    fontSize: 14,
    fontWeight: '600',
    paddingVertical: 2,
  },
  // Buttons
  recordButton: {
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  recordButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  sourceRow: {
    gap: 10,
  },
  sourceButton: {
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  sourceButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  pickingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 6,
  },
  pickingText: {
    fontSize: 15,
    fontStyle: 'italic',
  },
  testButton: {
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  testButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  // Result
  resultText: {
    fontSize: 15,
    lineHeight: 22,
  },
});
