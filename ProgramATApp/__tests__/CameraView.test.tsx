/**
 * CameraView Component Tests
 * @format
 */

import React from 'react';
import ReactTestRenderer from 'react-test-renderer';
import CameraView from '../CameraView';

describe('CameraView', () => {
  test('renders correctly', async () => {
    await ReactTestRenderer.act(() => {
      ReactTestRenderer.create(<CameraView />);
    });
  });

  test('renders with callback prop', async () => {
    const mockCallback = jest.fn();
    await ReactTestRenderer.act(() => {
      ReactTestRenderer.create(<CameraView onFrameCapture={mockCallback} />);
    });
  });
});
