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
  Image,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTheme } from './ThemeContext';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';

let useModel: any = null;
let GEMMA_4_E2B_IT: any = null;
let checkMultimodalSupport: any = null;
try {
  const litert = require('react-native-litert-lm');
  useModel = litert.useModel;
  GEMMA_4_E2B_IT = litert.GEMMA_4_E2B_IT;
  checkMultimodalSupport = litert.checkMultimodalSupport;
} catch (err) {
  console.warn('react-native-litert-lm not available:', err);
}

interface OnDeviceTestProps {
  onBack: () => void;
}

export default function OnDeviceTest({ onBack }: OnDeviceTestProps) {
  const { theme } = useTheme();

  if (!useModel || !GEMMA_4_E2B_IT) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: theme.background }]}>
        <View style={{ padding: 20, justifyContent: 'center', alignItems: 'center', flex: 1 }}>
          <Text style={[styles.title, { color: theme.text, marginBottom: 12 }]}>On-Device Test</Text>
          <Text style={{ color: 'red', textAlign: 'center', lineHeight: 24 }}>
            ⚠️ The react-native-litert-lm module is not available. This requires native module setup.{'\n\n'}
            Please ensure:
            {'\n'}1. Run: npm install
            {'\n'}2. Run: npx react-native doctor
            {'\n'}3. Rebuild: npm run ios (or npm run android)
          </Text>
          <TouchableOpacity style={{ marginTop: 20 }} onPress={onBack}>
            <Text style={{ color: theme.primary, fontSize: 16, fontWeight: '600' }}>← Go Back</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  const {
    model,
    isReady,
    downloadProgress,
    error,
  } = useModel(GEMMA_4_E2B_IT, {
    backend: 'cpu',
    autoLoad: true,
    //systemPrompt: 'You are a helpful assistant.',
    //enableMemoryTracking: true,
   // multimodal: true,
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
      const path = `${photo.path}`;
      console.log('[OnDeviceTest] Photo captured at:', path);
      setCapturedImageUri(path);
      setShowCamera(false);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.error('[OnDeviceTest] Camera error:', errorMsg);
      Alert.alert('Camera error', errorMsg);
    }
  };

  const generate = async () => {
    console.log('this is the model in use');
    //console.log(Object.keys(litert));
    console.log(model)
    console.log(GEMMA_4_E2B_IT);
    if (!model || !input.trim()) {
      setResponse('Error: Model not ready or no input provided');
      return;
    }

    try {
      setIsGenerating(true);
      setResponse('Generating response...');

      let result: any;

      if (capturedImageUri) {
        // Check if multimodal is supported on this platform
        const warning = checkMultimodalSupport?.();
        console.log('[OnDeviceTest] Platform:', Platform.OS, '| Multimodal supported:', !warning);
        
        if (warning) {
          console.warn('[OnDeviceTest] Multimodal not supported:', warning);
          setResponse(`⚠️ Image support not available on this platform (${Platform.OS}). Responding with text-only mode.`);
          // Fallback to text-only if multimodal not supported
          result = await Promise.race([
            model.sendMessage(input),
            new Promise((_, reject) => 
              setTimeout(() => reject(new Error('Model inference timeout (>30s)')), 30000)
            )
          ]);
        } else {
          try {
            console.log('[OnDeviceTest] Loading image into ArrayBuffer from:', capturedImageUri);
            
            // Use zero-copy ArrayBuffer method (recommended)
            // Fetch the image file and convert to ArrayBuffer
            //file://
            const response = await fetch(`${capturedImageUri}`);
            if (!response.ok) {
              throw new Error(`Failed to fetch image: ${response.statusText}`);
            }
            
            const imageBuffer = await response.arrayBuffer();
            console.log('[OnDeviceTest] Image loaded, buffer size:', imageBuffer.byteLength, 'bytes');
            console.log({
              path: capturedImageUri,
              size: imageBuffer.byteLength,
              firstBytes: Array.from(new Uint8Array(imageBuffer.slice(0, 16)))
            });
            // Use the zero-copy multimodal message API with proper format
            console.log('[OnDeviceTest] Calling sendMultimodalMessage with proper format');
            result = await Promise.race([
              model.sendMultimodalMessage([
                { type: 'image', imageBuffer },
                { type: 'text', text:  'What is in this image?' }
              ]),
              new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Model inference timeout (>30s)')), 30000)
              )
            ]);
          } catch (imageErr) {
            console.error('[OnDeviceTest] Image inference failed:', imageErr);
            // Fallback to text-only
            console.log('[OnDeviceTest] Falling back to text-only mode');
            setResponse('⚠️ Image inference failed. Responding with text-only mode.');
            result = await Promise.race([
              model.sendMessage(input),
              new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Model inference timeout (>30s)')), 30000)
              )
            ]);
          }
        }
      } else {
        console.log('[OnDeviceTest] Calling sendMessage with:', { input });
        
        if (!model.sendMessage) {
          throw new Error('sendMessage method not available on model');
        }
        
        result = await Promise.race([
          model.sendMessage(input),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Model inference timeout (>30s)')), 30000)
          )
        ]);
      }

      const responseText = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
      setResponse(responseText);
      console.log('[OnDeviceTest] Generation success:', responseText.substring(0, 100));
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.error('[OnDeviceTest] Generation error:', errorMsg);
      setResponse(`Error: ${errorMsg}`);
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
                ✅ Photo ready — using zero-copy ArrayBuffer for processing
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

