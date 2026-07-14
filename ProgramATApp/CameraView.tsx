/**
 * CameraView Component
 * Provides camera access and video capture capabilities
 * Built to support future frame streaming to server via sockets
 *
 * @format
 */

import React, { useEffect, useState, useRef, forwardRef, useImperativeHandle } from 'react';
import {
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ActivityIndicator,
  Linking,
  Platform,
  Alert,
  AccessibilityInfo,
  NativeModules,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import RNFS from 'react-native-fs';
import WebSocketService from './WebSocketService';
import Config from './config';
import BeepService from './BeepService';
import { CameraSource } from './CameraSource';

// Native bridge for the Meta Ray-Ban camera source. Only the shared-pipeline
// methods are used here; the low-level debug methods live in Settings.
const { MetaWearablesModule } = NativeModules as {
  MetaWearablesModule?: {
    startRayBanStream?: () => Promise<boolean>;
    stopRayBanStream?: () => Promise<boolean>;
    captureRayBanFrame?: () => Promise<{ base64: string; width: number; height: number }>;
  };
};

interface CameraViewProps {
  onFrameCapture?: (frame: any) => void;
}

export interface CameraViewHandle {
  captureFrame: () => Promise<{ base64: string; width: number; height: number } | null>;
  startStreaming: () => void;
  stopStreaming: () => void;
  isStreaming: () => boolean;
}

const CameraView = forwardRef<CameraViewHandle, CameraViewProps>(({ onFrameCapture }, ref) => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [cameraSource, setCameraSource] = useState<CameraSource>(CameraSource.Phone);
  const cameraRef = useRef<Camera>(null);
  const frameIntervalRef = useRef<any>(null);
  const errorCountRef = useRef<number>(0);
  const lastErrorTime = useRef<number>(0);
  const frameSkipCounterRef = useRef<number>(0);
  // Mirror of cameraSource so interval/handle closures read the live value.
  const cameraSourceRef = useRef<CameraSource>(CameraSource.Phone);

  // Camera permissions
  const { hasPermission, requestPermission } = useCameraPermission();

  // Get back camera device
  const device = useCameraDevice('back');

  // Handle permission request with better UX
  const handlePermissionRequest = async () => {
    const permission = await requestPermission();
    
    if (!permission) {
      // Permission denied - guide user to settings
      Alert.alert(
        'Camera Permission Required',
        'ProgramAT needs camera access to function. Please enable camera permission in your device settings.',
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Open Settings',
            onPress: () => {
              if (Platform.OS === 'ios') {
                Linking.openURL('app-settings:');
              } else {
                Linking.openSettings();
              }
            },
          },
        ]
      );
    }
  };

  useEffect(() => {
    // Request permission on mount if not already granted
    if (hasPermission === false) {
      handlePermissionRequest();
    }
  }, [hasPermission]);

  useEffect(() => {
    // Clean up frame capture interval on unmount, and stop the Ray-Ban stream
    // if it was the active source so the glasses don't keep streaming.
    return () => {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }
      if (cameraSourceRef.current === CameraSource.RayBan) {
        // Fire-and-forget on unmount; native still tears down to STOPPED.
        MetaWearablesModule?.stopRayBanStream?.().catch(() => {});
      }
    };
  }, []);

  // Fetch the most recent Ray-Ban frame from the native module. Returns the
  // same { base64, width, height } shape as the phone capture path so callers
  // (and the tools downstream) cannot tell the sources apart.
  const captureRayBanFrameJS = async (): Promise<{ base64: string; width: number; height: number } | null> => {
    try {
      const frame = await MetaWearablesModule?.captureRayBanFrame?.();
      if (!frame || !frame.base64) {
        return null;
      }
      return { base64: frame.base64, width: frame.width, height: frame.height };
    } catch (err) {
      console.warn('[CameraView] captureRayBanFrame failed:', err);
      return null;
    }
  };

  // Loading sound effect for camera operations
  useEffect(() => {
    if (isLoading) {
      console.log('[CameraView] Starting loading sound');
      BeepService.playLoadingSound();
    }
  }, [isLoading]);

  // Expose captureFrame method to parent components
  useImperativeHandle(ref, () => ({
    captureFrame: async () => {
      if (!isCameraActive) {
        console.warn('[CameraView] Cannot capture frame: camera not active');
        return null;
      }

      // Ray-Ban source: return the most recent frame from the glasses.
      if (cameraSourceRef.current === CameraSource.RayBan) {
        return await captureRayBanFrameJS();
      }

      if (!cameraRef.current) {
        console.warn('[CameraView] Cannot capture frame: phone camera not ready');
        return null;
      }

      try {
        const photo = await cameraRef.current.takePhoto({
          enableShutterSound: false,
        });

        // Read the file and convert to base64
        const base64Image = await RNFS.readFile(photo.path, 'base64');
        
        return {
          base64: `data:image/jpeg;base64,${base64Image}`,
          width: photo.width,
          height: photo.height,
        };
      } catch (error) {
        console.error('[CameraView] Error capturing frame:', error);
        return null;
      }
    },
    startStreaming: () => {
      console.log('[CameraView] Starting frame streaming from parent');
      startFrameStreaming();
    },
    stopStreaming: () => {
      console.log('[CameraView] Stopping frame streaming from parent');
      stopFrameStreaming();
    },
    isStreaming: () => {
      return isStreaming;
    },
  }), [isCameraActive, isStreaming]);

  // Start capturing and streaming frames at regular intervals
  const startFrameStreaming = () => {
    if (!WebSocketService.isActiveConnected()) {
      setError('WebSocket not connected. Please connect to server first.');
      return;
    }

    setIsStreaming(true);
    setFrameCount(0);
    errorCountRef.current = 0;
    frameSkipCounterRef.current = 0; // Reset frame skip counter when starting
    setError('');
    
    // The hosted multiframe experiment needs enough source frames to populate
    // its short rolling window. Other modes retain the existing cadence.
    const captureIntervalMs =
      WebSocketService.getNvidiaStreamingMode() === 'hosted_multiframe'
        ? WebSocketService.getStreamingFrameIntervalMs() || Config.FRAME_CAPTURE_INTERVAL_MS
        : Config.FRAME_CAPTURE_INTERVAL_MS;

    // Capture frames using configured interval
    frameIntervalRef.current = setInterval(() => {
      captureAndSendFrame();
    }, captureIntervalMs);
  };

  const stopFrameStreaming = () => {
    setIsStreaming(false);
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
  };

  const captureAndSendFrame = async () => {
    const inReviewMode = Config.APP_MODE === 'review';
    const connected = inReviewMode
      ? WebSocketService.isReviewConnected()
      : WebSocketService.isConnected();

    const isRayBan = cameraSourceRef.current === CameraSource.RayBan;

    // Phone source needs a live camera ref; Ray-Ban frames come from native.
    if ((!isRayBan && !cameraRef.current) || !connected) {
      return;
    }

    // Increment frame skip counter
    frameSkipCounterRef.current += 1;

    const hostedMultiframe =
      !inReviewMode && WebSocketService.getNvidiaStreamingMode() === 'hosted_multiframe';

    // Preserve the existing every-third-frame behavior outside the isolated
    // hosted experiment. hosted_multiframe sends each capture into its buffer.
    if (!hostedMultiframe && frameSkipCounterRef.current % 3 !== 0) {
      console.log(`[CameraView] Skipping frame ${frameSkipCounterRef.current} (only sending every 3rd frame)`);
      return;
    }

    try {
      // Acquire the frame from the active source, then send it through the
      // identical WebSocket path so downstream tools are source-agnostic.
      let base64WithPrefix: string;
      let width: number;
      let height: number;
      let phoneFrame: any = null;

      if (isRayBan) {
        const frame = await captureRayBanFrameJS();
        if (!frame) {
          return;
        }
        base64WithPrefix = frame.base64;
        width = frame.width;
        height = frame.height;
      } else {
        const photo = await cameraRef.current!.takePhoto({
          enableShutterSound: false,
        });

        // Read the file and convert to base64 using react-native-fs
        const base64Image = await RNFS.readFile(photo.path, 'base64');

        // Add data URL prefix to match server expectations
        base64WithPrefix = `data:image/jpeg;base64,${base64Image}`;
        width = photo.width;
        height = photo.height;
        phoneFrame = photo;
      }

      // Send to server — route to review server when in review mode
      const sent = inReviewMode
        ? WebSocketService.sendFrameToReview(base64WithPrefix, width, height)
        : WebSocketService.sendFrame(base64WithPrefix, width, height);

      if (sent) {
        setFrameCount(prev => prev + 1);
        console.log(`[CameraView] Sent frame ${frameSkipCounterRef.current} (processed frame #${frameCount + 1})`);
        // Reset error count on successful send
        if (errorCountRef.current > 0) {
          errorCountRef.current = 0;
          setError('');
        }
        if (onFrameCapture && phoneFrame) {
          onFrameCapture(phoneFrame);
        }
      }
    } catch (err) {
      // Increment error count using ref to avoid stale closure issues
      errorCountRef.current += 1;
      const newErrorCount = errorCountRef.current;
      
      // Show error message if we've had multiple consecutive failures
      // and enough time has passed since the last error notification
      const now = Date.now();
      if (newErrorCount >= 5 && now - lastErrorTime.current > 5000) {
        const errorMsg = `Frame capture issues (${newErrorCount} errors). Check connection.`;
        setError(errorMsg);
        lastErrorTime.current = now;
        console.error('Frame capture error:', err);
      }
      
      // Stop streaming if too many consecutive errors
      if (newErrorCount >= 20) {
        stopFrameStreaming();
        setError('Streaming stopped due to repeated errors. Please try again.');
      }
    }
  };

  const startCamera = async () => {
    try {
      setError('');
      setIsLoading(true);

      if (!hasPermission) {
        const granted = await requestPermission();
        if (!granted) {
          setError('Camera permission denied');
          setIsLoading(false);
          return;
        }
      }

      if (!device) {
        setError('No camera device found');
        setIsLoading(false);
        return;
      }

      cameraSourceRef.current = CameraSource.Phone;
      setCameraSource(CameraSource.Phone);
      setIsCameraActive(true);
      setIsLoading(false);
    } catch (err) {
      setIsLoading(false);
      setError('Failed to start camera');
      console.error(err);
    }
  };

  // Start the Meta Ray-Ban glasses as the active camera source. Once started,
  // the rest of the pipeline (streaming + take photo) is identical to phone.
  const startRayBanCamera = async () => {
    try {
      setError('');
      setIsLoading(true);

      if (typeof MetaWearablesModule?.startRayBanStream !== 'function') {
        throw new Error('Meta Ray-Ban camera is not available on this build.');
      }

      await MetaWearablesModule.startRayBanStream();

      cameraSourceRef.current = CameraSource.RayBan;
      setCameraSource(CameraSource.RayBan);
      setIsCameraActive(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(`Failed to start Meta Ray-Ban: ${message}`);
      AccessibilityInfo.announceForAccessibility(`Failed to start Meta Ray-Ban. ${message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const stopCamera = async () => {
    stopFrameStreaming();

    // For Ray-Ban, Stop is a full disconnect: wait for the native session to
    // reach STOPPED and release before we let the source switch back to phone.
    if (cameraSourceRef.current === CameraSource.RayBan) {
      setIsLoading(true);
      setError('');
      try {
        await MetaWearablesModule?.stopRayBanStream?.();
      } catch (err) {
        console.warn('[CameraView] stopRayBanStream failed:', err);
      } finally {
        setIsLoading(false);
      }
    }

    cameraSourceRef.current = CameraSource.Phone;
    setCameraSource(CameraSource.Phone);
    setIsCameraActive(false);
    setError('');
  };

  if (!hasPermission) {
    return (
      <View style={styles.container} accessible={false}>
        <Text 
          style={styles.title}
          accessibilityRole="header"
          accessible={true}>
          Camera Access
        </Text>
        <View style={styles.permissionContainer}>
          <Text 
            style={styles.permissionText}
            accessible={true}
            accessibilityRole="text">
            Camera permission is required to use this feature.
          </Text>
          <TouchableOpacity
            style={[styles.button, styles.startButton]}
            onPress={handlePermissionRequest}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Grant camera permission"
            accessibilityHint="Double tap to request camera access or open settings">
            <Text style={styles.buttonText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.container} accessible={false}>
        <Text 
          style={styles.title}
          accessibilityRole="header"
          accessible={true}>
          Camera Access
        </Text>
        <View style={styles.errorContainer}>
          <Text 
            style={styles.errorText}
            accessible={true}
            accessibilityRole="alert">
            No camera device available
          </Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container} accessible={false}>
      {/* Compact header row with title and buttons */}
      <View style={styles.headerRow}>
        <Text style={styles.title} accessibilityRole="header" accessible={true}>
          Camera
        </Text>
        
        <View style={styles.buttonContainer} accessible={false}>
          {!isCameraActive ? (
            isLoading ? (
              <ActivityIndicator
                size="small"
                color="#2196F3"
                accessible={true}
                accessibilityLabel="Connecting to camera" />
            ) : (
              <>
                <TouchableOpacity
                  style={[styles.button, styles.compactButton, styles.startButton]}
                  onPress={startCamera}
                  accessible={true}
                  accessibilityRole="button"
                  accessibilityLabel="Start Phone Camera"
                  accessibilityHint="Use the phone's back camera as the source for this tool">
                  <Text style={styles.buttonText}>Phone</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={[styles.button, styles.compactButton, styles.rayBanButton]}
                  onPress={startRayBanCamera}
                  accessible={true}
                  accessibilityRole="button"
                  accessibilityLabel="Start Meta Ray-Ban"
                  accessibilityHint="Use your Meta Ray-Ban glasses as the source for this tool">
                  <Text style={styles.buttonText}>Ray-Ban</Text>
                </TouchableOpacity>
              </>
            )
          ) : (
            <TouchableOpacity
              style={[styles.button, styles.stopButton]}
              onPress={stopCamera}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel={cameraSource === CameraSource.RayBan ? 'Stop Meta Ray-Ban camera' : 'Stop phone camera'}>
              <Text style={styles.buttonText}>Stop</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>

      {error !== '' && (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      <View style={styles.cameraContainer} accessible={false}>
        {isCameraActive ? (
          cameraSource === CameraSource.Phone ? (
            <Camera
              ref={cameraRef}
              style={styles.camera}
              device={device}
              isActive={isCameraActive}
              photo={true}
              video={true}
              accessible={true}
              accessibilityLabel="Camera preview"
              accessibilityHint="Live camera feed displaying what the camera sees"
            />
          ) : (
            <View style={styles.cameraPlaceholder}>
              <Text
                style={styles.placeholderText}
                accessible={true}
                accessibilityRole="text">
                Meta Ray-Ban camera active
              </Text>
              <Text
                style={styles.placeholderSubtext}
                accessible={true}
                accessibilityRole="text">
                Frames stream from your glasses to this tool
              </Text>
            </View>
          )
        ) : (
          <View style={styles.cameraPlaceholder}>
            <Text
              style={styles.placeholderText}
              accessible={true}
              accessibilityRole="text">
              Camera preview will appear here
            </Text>
            <Text
              style={styles.placeholderSubtext}
              accessible={true}
              accessibilityRole="text">
              Choose "Phone" or "Ray-Ban" to begin
            </Text>
          </View>
        )}
      </View>
    </View>
  );
});

export default CameraView;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 8,
    backgroundColor: '#f5f5f5',
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  title: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  buttonContainer: {
    flexDirection: 'row',
    gap: 6,
  },
  button: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    minWidth: 60,
    alignItems: 'center',
  },
  compactButton: {
    paddingHorizontal: 10,
    minWidth: 36,
  },
  startButton: {
    backgroundColor: '#4CAF50',
  },
  rayBanButton: {
    backgroundColor: '#00897B',
  },
  testButton: {
    backgroundColor: '#673AB7',
  },
  physicalButton: {
    backgroundColor: '#00897B',
  },
  debugButton: {
    backgroundColor: '#455A64',
  },
  mockButton: {
    backgroundColor: '#009688',
  },
  stopButton: {
    backgroundColor: '#f44336',
  },
  streamButton: {
    backgroundColor: '#2196F3',
  },
  stopStreamButton: {
    backgroundColor: '#FF9800',
  },
  buttonText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '600',
  },
  cameraIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 6,
  },
  cameraDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#4CAF50',
    marginRight: 6,
  },
  cameraText: {
    fontSize: 12,
    color: '#4CAF50',
    fontWeight: '600',
  },
  errorContainer: {
    backgroundColor: '#ffebee',
    padding: 8,
    borderRadius: 6,
    marginBottom: 6,
  },
  errorText: {
    color: '#c62828',
    fontSize: 12,
    textAlign: 'center',
  },
  physicalStatusContainer: {
    backgroundColor: '#e8f5e9',
    padding: 8,
    borderRadius: 6,
    marginBottom: 6,
  },
  physicalStatusErrorContainer: {
    backgroundColor: '#ffebee',
  },
  physicalStatusText: {
    color: '#1b5e20',
    fontSize: 12,
    textAlign: 'center',
  },
  physicalStatusErrorText: {
    color: '#c62828',
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  permissionText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 20,
  },
  cameraContainer: {
    flex: 1,
    backgroundColor: '#000',
    borderRadius: 6,
    overflow: 'hidden',
    marginBottom: 8,
  },
  camera: {
    flex: 1,
  },
  cameraPlaceholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#333',
  },
  placeholderText: {
    fontSize: 16,
    color: '#fff',
    marginBottom: 8,
  },
  placeholderSubtext: {
    fontSize: 14,
    color: '#aaa',
  },
});
