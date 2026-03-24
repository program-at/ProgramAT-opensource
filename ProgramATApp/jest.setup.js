/* eslint-disable no-undef */
// Mock react-native-vision-camera
jest.mock('react-native-vision-camera', () => ({
  Camera: 'Camera',
  useCameraDevice: jest.fn(() => ({
    id: 'mock-camera-id',
    position: 'back',
    hasFlash: true,
    hasTorch: true,
    minZoom: 1,
    maxZoom: 10,
    neutralZoom: 1,
  })),
  useCameraPermission: jest.fn(() => ({
    hasPermission: true,
    requestPermission: jest.fn(() => Promise.resolve(true)),
  })),
}));

// Mock react-native-fs
jest.mock('react-native-fs', () => ({
  readFile: jest.fn(() => Promise.resolve('mockBase64Data')),
}));

// Mock react-native-tts
jest.mock('react-native-tts', () => ({
  setDefaultLanguage: jest.fn(() => Promise.resolve()),
  setDefaultRate: jest.fn(() => Promise.resolve()),
  speak: jest.fn(() => Promise.resolve()),
  stop: jest.fn(() => Promise.resolve()),
}));

