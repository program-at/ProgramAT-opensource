// 
/**
 * OnDeviceTest - On-device LLM test screen with text and multimodal (photo) support
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Button,
  TextInput,
  ScrollView,
  ActivityIndicator,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTheme } from './ThemeContext';
import { useModel, GEMMA_4_E2B_IT } from 'react-native-litert-lm';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';

interface OnDeviceTestProps {
  onBack: () => void;
}

export default function OnDeviceTest({ onBack }: OnDeviceTestProps) {
  const { theme } = useTheme();

  const {
    model,
    isReady,
    downloadProgress,
    error,
  } = useModel(GEMMA_4_E2B_IT, {
    backend: 'cpu',
    autoLoad: true,
    systemPrompt: 'You are a helpful assistant.',
    enableMemoryTracking: true,
  });

  const [input, setInput] = useState('Describe what is in this image.');
  const [response, setResponse] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [capturedImageUri, setCapturedImageUri] = useState<string | null>(null);

  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const cameraRef = useRef<Camera>(null);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, []);

  const takePhoto = async () => {
    if (!cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePhoto({ flash: 'off' });
      setCapturedImageUri(`file://${photo.path}`);
      setShowCamera(false);
    } catch (err) {
      Alert.alert('Camera error', String(err));
    }
  };

  const generate = async () => {
    if (!model || !input.trim()) return;

    try {
      setIsGenerating(true);
      setResponse('');

      let result: string;

      if (capturedImageUri) {
        // Strip file:// prefix — sendMessageWithImage requires a bare absolute path
        const imagePath = capturedImageUri.replace(/^file:\/\//, '');
        result = await model.sendMessageWithImage(input, imagePath);
      } else {
        result = await model.sendMessage(input);
      }

      setResponse(typeof result === 'string' ? result : JSON.stringify(result, null, 2));
    } catch (err) {
      setResponse(`Error: ${String(err)}`);
    } finally {
      setIsGenerating(false);
    }
  };

  if (showCamera && device) {
    return (
      <View style={{ flex: 1, backgroundColor: '#000' }}>
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={true}
          photo={true}
        />
        <SafeAreaView style={styles.cameraControls}>
          <TouchableOpacity style={styles.captureBtn} onPress={takePhoto}>
            <Text style={styles.captureBtnText}>📸 Capture</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.cancelBtn} onPress={() => setShowCamera(false)}>
            <Text style={styles.cancelBtnText}>Cancel</Text>
          </TouchableOpacity>
        </SafeAreaView>
      </View>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.background }]}>
      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        <Text style={[styles.title, { color: theme.text }]}>On-Device Test</Text>

        {error && (
          <Text style={{ color: 'red', marginTop: 8 }}>{String(error)}</Text>
        )}

        {!isReady ? (
          <View style={{ marginTop: 20, alignItems: 'center' }}>
            <ActivityIndicator />
            <Text style={{ color: theme.text, marginTop: 8 }}>
              Loading model... {Math.round(downloadProgress * 100)}%
            </Text>
          </View>
        ) : (
          <View style={{ marginTop: 16, gap: 12 }}>
            {/* Photo section */}
            <View style={styles.row}>
              <TouchableOpacity
                style={[styles.photoBtn, { borderColor: theme.primary }]}
                onPress={() => setShowCamera(true)}>
                <Text style={[styles.photoBtnText, { color: theme.primary }]}>
                  {capturedImageUri ? '📷 Retake Photo' : '📷 Take Photo'}
                </Text>
              </TouchableOpacity>
              {capturedImageUri && (
                <TouchableOpacity onPress={() => setCapturedImageUri(null)}>
                  <Text style={{ color: 'red', marginLeft: 12 }}>✕ Clear</Text>
                </TouchableOpacity>
              )}
            </View>

            {capturedImageUri && (
              <Text style={{ color: theme.textSecondary, fontSize: 12 }}>
                ✅ Photo ready — model will answer about the image
              </Text>
            )}

            <TextInput
              value={input}
              onChangeText={setInput}
              placeholder={capturedImageUri ? 'Ask about the photo...' : 'Type a message...'}
              multiline
              style={[styles.input, { color: theme.text, borderColor: theme.textSecondary }]}
              placeholderTextColor={theme.textSecondary}
            />

            <Button
              title={isGenerating ? 'Generating...' : 'Generate'}
              onPress={generate}
              disabled={isGenerating}
            />

            {response.length > 0 && (
              <View style={[styles.responseContainer, { borderColor: theme.border, backgroundColor: theme.card }]}>
                <Text style={{ color: theme.text }}>{response}</Text>
              </View>
            )}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: 20, paddingBottom: 40 },
  title: { fontSize: 24, fontWeight: '700', marginBottom: 4 },
  row: { flexDirection: 'row', alignItems: 'center' },
  photoBtn: {
    borderWidth: 1.5,
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 16,
  },
  photoBtnText: { fontSize: 15, fontWeight: '600' },
  input: {
    borderWidth: 1,
    borderRadius: 8,
    padding: 10,
    fontSize: 15,
    minHeight: 80,
  },
  responseContainer: {
    borderWidth: 1,
    borderRadius: 8,
    padding: 12,
    marginTop: 4,
  },
  cameraControls: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    alignItems: 'center',
    gap: 12,
  },
  captureBtn: {
    backgroundColor: '#fff',
    borderRadius: 40,
    paddingVertical: 14,
    paddingHorizontal: 32,
  },
  captureBtnText: { fontSize: 18, fontWeight: '700', color: '#000' },
  cancelBtn: { marginTop: 8 },
  cancelBtnText: { color: '#fff', fontSize: 16 },
});

