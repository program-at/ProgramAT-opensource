/**
 * VideoRecorderModal
 * Full-screen camera overlay for recording a short video to attach to a creation submission.
 * Uses react-native-vision-camera v4 for recording.
 */

import React, { useRef, useState, useEffect } from 'react';
import {
  Modal,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  Alert,
  Linking,
} from 'react-native';
import { Camera, useCameraDevice, useCameraPermission, useMicrophonePermission } from 'react-native-vision-camera';

interface VideoRecorderModalProps {
  visible: boolean;
  onVideoRecorded: (filePath: string) => void;
  onCancel: () => void;
}

export default function VideoRecorderModal({
  visible,
  onVideoRecorded,
  onCancel,
}: VideoRecorderModalProps) {
  const cameraRef = useRef<Camera>(null);
  const device = useCameraDevice('back');
  const { hasPermission: hasCameraPermission, requestPermission: requestCameraPermission } = useCameraPermission();
  const { hasPermission: hasMicPermission, requestPermission: requestMicPermission } = useMicrophonePermission();
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Request both camera and microphone permissions when modal opens
  useEffect(() => {
    if (!visible) return;
    const requestAll = async () => {
      if (!hasCameraPermission) await requestCameraPermission();
      if (!hasMicPermission) await requestMicPermission();
    };
    requestAll();
  }, [visible, hasCameraPermission, hasMicPermission, requestCameraPermission, requestMicPermission]);

  // Reset state when modal opens; clean up timer on unmount
  useEffect(() => {
    if (visible) {
      setIsRecording(false);
      setElapsed(0);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [visible]);

  const startRecording = () => {
    if (!cameraRef.current) return;
    setIsRecording(true);
    setElapsed(0);
    timerRef.current = setInterval(() => setElapsed(e => e + 1), 1000);

    cameraRef.current.startRecording({
      onRecordingFinished: video => {
        if (timerRef.current) clearInterval(timerRef.current);
        setIsRecording(false);
        // vision-camera returns an absolute path on Android, file:// URI on iOS
        const path = video.path.startsWith('file://') ? video.path : `file://${video.path}`;
        onVideoRecorded(path);
      },
      onRecordingError: error => {
        if (timerRef.current) clearInterval(timerRef.current);
        setIsRecording(false);
        Alert.alert('Recording error', error.message ?? 'Could not record video');
      },
    });
  };

  const stopRecording = async () => {
    await cameraRef.current?.stopRecording();
    // onRecordingFinished fires automatically after stop
  };

  const handleCancel = async () => {
    if (isRecording) {
      await cameraRef.current?.stopRecording().catch(() => {});
    }
    if (timerRef.current) clearInterval(timerRef.current);
    setIsRecording(false);
    onCancel();
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  if (!visible) return null;

  if (!hasCameraPermission || !hasMicPermission) {
    const missingCamera = !hasCameraPermission;
    const missingMic = !hasMicPermission;
    const message = missingCamera && missingMic
      ? 'Camera and microphone access are required to record video. Please enable them in Settings.'
      : missingCamera
        ? 'Camera permission is required to record a video. Please enable it in Settings.'
        : 'Microphone permission is required to record video with audio. Please enable it in Settings.';
    return (
      <Modal visible={visible} animationType="slide">
        <SafeAreaView style={styles.fallback}>
          <Text style={styles.fallbackText}>{message}</Text>
          <TouchableOpacity
            style={styles.cancelPill}
            onPress={() => Linking.openURL('app-settings:')}>
            <Text style={styles.cancelPillText}>Open Settings</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.cancelPill, { marginTop: 12 }]} onPress={onCancel}>
            <Text style={styles.cancelPillText}>Cancel</Text>
          </TouchableOpacity>
        </SafeAreaView>
      </Modal>
    );
  }

  if (!device) {
    return (
      <Modal visible={visible} animationType="slide">
        <SafeAreaView style={styles.fallback}>
          <Text style={styles.fallbackText}>No camera found on this device.</Text>
          <TouchableOpacity style={styles.cancelPill} onPress={onCancel}>
            <Text style={styles.cancelPillText}>Cancel</Text>
          </TouchableOpacity>
        </SafeAreaView>
      </Modal>
    );
  }

  return (
    <Modal visible={visible} animationType="slide" statusBarTranslucent>
      <View style={styles.container}>
        {/* Camera preview */}
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={visible}
          video
          audio
        />

        {/* Recording indicator */}
        {isRecording && (
          <SafeAreaView style={styles.timerRow} pointerEvents="none">
            <View style={styles.timerBadge}>
              <View style={styles.recDot} />
              <Text style={styles.timerText}>{formatTime(elapsed)}</Text>
            </View>
          </SafeAreaView>
        )}

        {/* Bottom controls */}
        <SafeAreaView style={styles.controls}>
          <TouchableOpacity
            style={styles.sidePill}
            onPress={handleCancel}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Cancel"
            accessibilityHint="Discard recording and return to creation form">
            <Text style={styles.cancelPillText}>Cancel</Text>
          </TouchableOpacity>

          {/* Record / Stop button */}
          <TouchableOpacity
            style={[styles.recordRing, isRecording && styles.recordRingActive]}
            onPress={isRecording ? stopRecording : startRecording}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel={isRecording ? 'Stop recording' : 'Start recording'}
            accessibilityHint={
              isRecording ? 'Double tap to stop and save video' : 'Double tap to begin recording'
            }>
            <View style={[styles.recordCore, isRecording && styles.recordCoreStop]} />
          </TouchableOpacity>

          {/* Spacer keeps button centred */}
          <View style={styles.sidePill} />
        </SafeAreaView>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  fallback: {
    flex: 1,
    backgroundColor: '#000',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
  },
  fallbackText: {
    color: '#fff',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 24,
  },
  // ---- Timer overlay ----
  timerRow: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    alignItems: 'center',
    paddingTop: 56,
  },
  timerBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.55)',
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 20,
    gap: 8,
  },
  recDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#ff3b30',
  },
  timerText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  // ---- Bottom controls ----
  controls: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 32,
    paddingBottom: 44,
  },
  sidePill: {
    width: 70,
    alignItems: 'center',
  },
  cancelPill: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    backgroundColor: 'rgba(0,0,0,0.45)',
    borderRadius: 20,
  },
  cancelPillText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  recordRing: {
    width: 76,
    height: 76,
    borderRadius: 38,
    borderWidth: 4,
    borderColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  recordRingActive: {
    borderColor: '#ff3b30',
  },
  recordCore: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#ff3b30',
  },
  recordCoreStop: {
    width: 28,
    height: 28,
    borderRadius: 6,
  },
});
