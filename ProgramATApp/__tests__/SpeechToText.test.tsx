/**
 * @format
 */

import React from 'react';
import ReactTestRenderer from 'react-test-renderer';
import SpeechToText from '../SpeechToText';

// Mock the Voice module
jest.mock('@react-native-voice/voice', () => ({
  __esModule: true,
  default: {
    onSpeechStart: jest.fn(),
    onSpeechEnd: jest.fn(),
    onSpeechResults: jest.fn(),
    onSpeechError: jest.fn(),
    start: jest.fn(() => Promise.resolve()),
    stop: jest.fn(() => Promise.resolve()),
    destroy: jest.fn(() => Promise.resolve()),
    removeAllListeners: jest.fn(),
  },
}));

// Mock WebSocketService
jest.mock('../WebSocketService', () => ({
  __esModule: true,
  default: {
    isConnected: jest.fn(() => true),
    sendText: jest.fn(),
    onMessage: jest.fn(),
  },
}));

describe('SpeechToText Component', () => {
  test('renders correctly', async () => {
    await ReactTestRenderer.act(() => {
      ReactTestRenderer.create(<SpeechToText />);
    });
  });

  test('renders with onTranscriptChange callback', async () => {
    const mockCallback = jest.fn();
    await ReactTestRenderer.act(() => {
      ReactTestRenderer.create(<SpeechToText onTranscriptChange={mockCallback} />);
    });
  });
});

describe('Sentence Extraction Logic', () => {
  const extractCompleteSentences = (text: string): { sentences: string[]; remainder: string } => {
    const sentences: string[] = [];
    let remaining = text;
    
    while (remaining.length > 0) {
      const sentenceRegex = /^(.*?[.!?])(\s+|$)/;
      const match = remaining.match(sentenceRegex);
      
      if (match) {
        sentences.push(match[1].trim());
        remaining = remaining.substring(match[0].length).trim();
      } else {
        break;
      }
    }
    
    return { sentences, remainder: remaining };
  };

  test('extracts single sentence ending with period', () => {
    const result = extractCompleteSentences('Hello world.');
    expect(result.sentences).toEqual(['Hello world.']);
    expect(result.remainder).toBe('');
  });

  test('extracts sentence ending with question mark', () => {
    const result = extractCompleteSentences('How are you?');
    expect(result.sentences).toEqual(['How are you?']);
    expect(result.remainder).toBe('');
  });

  test('extracts sentence ending with exclamation mark', () => {
    const result = extractCompleteSentences('Great job!');
    expect(result.sentences).toEqual(['Great job!']);
    expect(result.remainder).toBe('');
  });

  test('extracts sentence with remainder text', () => {
    const result = extractCompleteSentences('Hello world. This is');
    expect(result.sentences).toEqual(['Hello world.']);
    expect(result.remainder).toBe('This is');
  });

  test('returns no sentences for incomplete text', () => {
    const result = extractCompleteSentences('Hello world');
    expect(result.sentences).toEqual([]);
    expect(result.remainder).toBe('Hello world');
  });

  test('extracts multiple sentences correctly', () => {
    const result = extractCompleteSentences('First sentence. Second sentence.');
    expect(result.sentences).toEqual(['First sentence.', 'Second sentence.']);
    expect(result.remainder).toBe('');
  });

  test('extracts multiple sentences with remainder', () => {
    const result = extractCompleteSentences('First sentence. Second sentence. Third');
    expect(result.sentences).toEqual(['First sentence.', 'Second sentence.']);
    expect(result.remainder).toBe('Third');
  });

  test('handles empty string', () => {
    const result = extractCompleteSentences('');
    expect(result.sentences).toEqual([]);
    expect(result.remainder).toBe('');
  });

  test('handles sentence with multiple spaces', () => {
    const result = extractCompleteSentences('Hello world.   Next sentence!');
    expect(result.sentences).toEqual(['Hello world.', 'Next sentence!']);
    expect(result.remainder).toBe('');
  });

  test('extracts mixed punctuation sentences', () => {
    const result = extractCompleteSentences('What? Really! Yes.');
    expect(result.sentences).toEqual(['What?', 'Really!', 'Yes.']);
    expect(result.remainder).toBe('');
  });
});
