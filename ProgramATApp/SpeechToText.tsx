/**
 * SpeechToText Component
 * Provides speech-to-text capabilities using React Native Voice
 *
 * @format
 */

import React, { useEffect, useState, useRef } from 'react';
import {
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import Voice, {
  SpeechResultsEvent,
  SpeechErrorEvent,
} from '@react-native-voice/voice';
import WebSocketService from './WebSocketService';
import TextToSpeechService from './TextToSpeechService';

interface SpeechToTextProps {
  onTranscriptChange?: (transcript: string) => void;
  serverFeedback?: string;
}

export default function SpeechToText({ onTranscriptChange, serverFeedback }: SpeechToTextProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // Use ref to track position without triggering re-renders
  const lastSentIndexRef = useRef(0);

  // Helper function to extract all complete sentences from text
  const extractCompleteSentences = (text: string): { sentences: string[]; remainder: string } => {
    const sentences: string[] = [];
    let remaining = text;
    
    // Keep extracting sentences until we can't find any more
    while (remaining.length > 0) {
      // Match sentences ending with period, question mark, or exclamation mark
      const sentenceRegex = /^(.*?[.!?])(\s+|$)/;
      const match = remaining.match(sentenceRegex);
      
      if (match) {
        sentences.push(match[1].trim());
        remaining = remaining.substring(match[0].length).trim();
      } else {
        // No more complete sentences found
        break;
      }
    }
    
    return { sentences, remainder: remaining };
  };

  useEffect(() => {
    const onSpeechStart = () => {
      setIsLoading(false);
      setError('');
      lastSentIndexRef.current = 0; // Reset sent tracking on new recording
    };

    const onSpeechEnd = () => {
      setIsRecording(false);
      setIsLoading(false);
      
      // Send any remaining text that hasn't been sent when recording ends
      setTranscript((currentTranscript) => {
        if (currentTranscript && lastSentIndexRef.current < currentTranscript.length && WebSocketService.isConnected()) {
          const unsent = currentTranscript.substring(lastSentIndexRef.current).trim();
          if (unsent) {
            console.log('[SpeechToText] Sending remaining text on end:', unsent);
            WebSocketService.sendText(unsent);
          }
        }
        return currentTranscript;
      });
    };

    // Handle interim results while user is speaking
    const onSpeechPartialResults = (event: SpeechResultsEvent) => {
      if (event.value && event.value.length > 0) {
        const currentTranscript = event.value[0];
        console.log('[SpeechToText] Partial result:', currentTranscript);
        
        // Update the displayed transcript
        setTranscript(currentTranscript);
        
        // Get only the new portion since last send
        const newPortion = currentTranscript.substring(lastSentIndexRef.current);
        console.log('[SpeechToText] New portion:', newPortion);
        
        if (newPortion) {
          // Extract complete sentences from the new portion
          const { sentences, remainder } = extractCompleteSentences(newPortion);
          console.log('[SpeechToText] Extracted sentences:', sentences);
          console.log('[SpeechToText] Remainder:', remainder);
          
          if (sentences.length > 0) {
            // Combine all complete sentences
            const completedText = sentences.join(' ');
            
            // Send the completed sentences
            if (WebSocketService.isConnected()) {
              console.log('[SpeechToText] Sending to server:', completedText);
              WebSocketService.sendText(completedText);
            }
            
            // Update the index to the start of the remainder
            // New index = old index + length of what we just sent
            const sentLength = newPortion.length - remainder.length;
            lastSentIndexRef.current = lastSentIndexRef.current + sentLength;
            console.log('[SpeechToText] Updated lastSentIndexRef.current to:', lastSentIndexRef.current);
          }
        }
        
        if (onTranscriptChange) {
          onTranscriptChange(currentTranscript);
        }
      }
    };

    const onSpeechResults = (event: SpeechResultsEvent) => {
      // This fires when speech recognition completes
      // We still handle it to catch any final results
      if (event.value && event.value.length > 0) {
        const currentTranscript = event.value[0];
        console.log('[SpeechToText] Final result:', currentTranscript);
        
        setTranscript(currentTranscript);
        
        // Get only the new portion since last send
        const newPortion = currentTranscript.substring(lastSentIndexRef.current);
        
        if (newPortion) {
          // Extract complete sentences from the new portion
          const { sentences, remainder } = extractCompleteSentences(newPortion);
          
          if (sentences.length > 0) {
            // Combine all complete sentences
            const completedText = sentences.join(' ');
            
            // Send the completed sentences
            if (WebSocketService.isConnected()) {
              console.log('[SpeechToText] Sending final sentences to server:', completedText);
              WebSocketService.sendText(completedText);
            }
            
            // Update the index to the start of the remainder
            const sentLength = newPortion.length - remainder.length;
            lastSentIndexRef.current = lastSentIndexRef.current + sentLength;
          }
        }
        
        if (onTranscriptChange) {
          onTranscriptChange(currentTranscript);
        }
      }
    };

    const onSpeechError = (event: SpeechErrorEvent) => {
      setIsRecording(false);
      setIsLoading(false);
      setError(event.error?.message || 'Speech recognition error');
    };

    // Set up voice event listeners
    Voice.onSpeechStart = onSpeechStart;
    Voice.onSpeechEnd = onSpeechEnd;
    Voice.onSpeechPartialResults = onSpeechPartialResults; // Interim results while speaking
    Voice.onSpeechResults = onSpeechResults; // Final results
    Voice.onSpeechError = onSpeechError;

    return () => {
      // Clean up voice instance with proper error handling
      Voice.destroy()
        .then(Voice.removeAllListeners)
        .catch((err) => {
          console.error('Error cleaning up Voice:', err);
          Voice.removeAllListeners();
        });
    };
  }, [onTranscriptChange]); // Only depend on onTranscriptChange, not state variables

  // Initialize TTS on mount
  useEffect(() => {
    TextToSpeechService.initialize();
  }, []);

  const startRecording = async () => {
    try {
      setError('');
      setIsLoading(true);
      await Voice.start('en-US'); // Using US English
      setIsRecording(true);
    } catch (err) {
      setIsLoading(false);
      setError('Failed to start recording');
      console.error(err);
    }
  };

  const stopRecording = async () => {
    try {
      await Voice.stop();
      setIsRecording(false);
      setIsLoading(false);
    } catch (err) {
      setError('Failed to stop recording');
      console.error(err);
    }
  };

  const clearTranscript = () => {
    setTranscript('');
    lastSentIndexRef.current = 0;
    setError('');
    if (onTranscriptChange) {
      onTranscriptChange('');
    }
  };

  return (
    <View style={styles.container} accessible={false}>
      <Text 
        style={styles.title}
        accessibilityRole="header"
        accessible={true}>
        Speech to Text
      </Text>

      <View style={styles.buttonContainer} accessible={false}>
        {!isRecording ? (
          <TouchableOpacity
            style={[styles.button, styles.startButton]}
            onPress={startRecording}
            disabled={isLoading}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Start recording"
            accessibilityHint="Double tap to begin speech recognition"
            accessibilityState={{ disabled: isLoading }}>
            {isLoading ? (
              <ActivityIndicator color="#fff" accessible={false} />
            ) : (
              <Text style={styles.buttonText}>Start Recording</Text>
            )}
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={[styles.button, styles.stopButton]}
            onPress={stopRecording}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Stop recording"
            accessibilityHint="Double tap to stop speech recognition">
            <Text style={styles.buttonText}>Stop Recording</Text>
          </TouchableOpacity>
        )}

        {transcript !== '' && (
          <TouchableOpacity
            style={[styles.button, styles.clearButton]}
            onPress={clearTranscript}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Clear transcript"
            accessibilityHint="Double tap to clear the current transcript">
            <Text style={styles.buttonText}>Clear</Text>
          </TouchableOpacity>
        )}
      </View>

      {isRecording && (
        <View 
          style={styles.recordingIndicator}
          accessible={true}
          accessibilityRole="text"
          accessibilityLabel="Recording in progress"
          accessibilityLiveRegion="polite">
          <View style={styles.recordingDot} accessible={false} />
          <Text style={styles.recordingText}>Recording...</Text>
        </View>
      )}

      {error !== '' && (
        <View 
          style={styles.errorContainer}
          accessible={true}
          accessibilityRole="alert"
          accessibilityLabel={`Error: ${error}`}
          accessibilityLiveRegion="assertive">
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      {serverFeedback && serverFeedback !== '' && (
        <View 
          style={styles.feedbackContainer}
          accessible={true}
          accessibilityRole="alert"
          accessibilityLabel={`Server feedback: ${serverFeedback}`}
          accessibilityLiveRegion="polite">
          <Text style={styles.feedbackText}>🔊 {serverFeedback}</Text>
        </View>
      )}

      <ScrollView 
        style={styles.transcriptContainer}
        accessible={false}
        accessibilityLabel="Transcript scroll view">
        <Text 
          style={styles.transcriptLabel}
          accessible={true}
          accessibilityRole="header">
          Transcript:
        </Text>
        <Text 
          style={styles.transcriptText}
          accessible={true}
          accessibilityRole="text"
          accessibilityLabel={transcript || 'No transcript yet. Press start recording to begin.'}
          accessibilityLiveRegion="polite">
          {transcript || 'No transcript yet. Press "Start Recording" to begin.'}
        </Text>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 12,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 12,
    textAlign: 'center',
    color: '#333',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 10,
    marginBottom: 12,
    flexWrap: 'wrap',
  },
  button: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
    minWidth: 140,
    alignItems: 'center',
  },
  startButton: {
    backgroundColor: '#4CAF50',
  },
  stopButton: {
    backgroundColor: '#f44336',
  },
  clearButton: {
    backgroundColor: '#2196F3',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  recordingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  recordingDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#f44336',
    marginRight: 8,
  },
  recordingText: {
    fontSize: 16,
    color: '#f44336',
    fontWeight: '600',
  },
  errorContainer: {
    backgroundColor: '#ffebee',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
  },
  errorText: {
    color: '#c62828',
    fontSize: 14,
  },
  feedbackContainer: {
    backgroundColor: '#e3f2fd',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#2196f3',
  },
  feedbackText: {
    color: '#1565c0',
    fontSize: 14,
    fontWeight: '500',
  },
  transcriptContainer: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  transcriptLabel: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 8,
    color: '#333',
  },
  transcriptText: {
    fontSize: 16,
    lineHeight: 24,
    color: '#666',
  },
});
