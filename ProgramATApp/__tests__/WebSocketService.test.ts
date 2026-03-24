/**
 * WebSocketService Tests
 * @format
 */

import WebSocketService from '../WebSocketService';

describe('WebSocketService', () => {
  beforeEach(() => {
    // Reset service state before each test
    WebSocketService.disconnect();
  });

  afterEach(() => {
    WebSocketService.disconnect();
  });

  test('service is initially not connected', () => {
    expect(WebSocketService.isConnected()).toBe(false);
  });

  test('sendFrame returns false when not connected', () => {
    const result = WebSocketService.sendFrame('base64data');
    expect(result).toBe(false);
  });

  test('sendText returns false when not connected', () => {
    const result = WebSocketService.sendText('test text');
    expect(result).toBe(false);
  });

  test('sendFrameWithText returns false when not connected', () => {
    const result = WebSocketService.sendFrameWithText('base64data', 'test text');
    expect(result).toBe(false);
  });

  test('callbacks can be set without errors', () => {
    const mockConnectionCallback = jest.fn();
    const mockMessageCallback = jest.fn();
    const mockErrorCallback = jest.fn();

    WebSocketService.onConnectionChange(mockConnectionCallback);
    WebSocketService.onMessage(mockMessageCallback);
    WebSocketService.onError(mockErrorCallback);

    // Verify no errors occurred
    expect(mockConnectionCallback).not.toHaveBeenCalled();
    expect(mockMessageCallback).not.toHaveBeenCalled();
    expect(mockErrorCallback).not.toHaveBeenCalled();
  });

  test('resetFrameCounter does not throw error', () => {
    expect(() => {
      WebSocketService.resetFrameCounter();
    }).not.toThrow();
  });
});
