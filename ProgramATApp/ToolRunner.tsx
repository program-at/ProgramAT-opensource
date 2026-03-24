/**
 * ToolRunner Component
 * Displays and runs the selected tool
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  ScrollView,
  Keyboard,
  TouchableWithoutFeedback,
  Platform,
  AccessibilityInfo,
  findNodeHandle,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';
import CameraView, { CameraViewHandle } from './CameraView';
import WebSocketService from './WebSocketService';
import AudioOutputService from './AudioOutputService';
import BeepService from './BeepService';
import TextToSpeechService from './TextToSpeechService';
import Voice, {
  SpeechResultsEvent,
  SpeechErrorEvent,
} from '@react-native-voice/voice';

// Configuration for text similarity filtering
const SIMILARITY_THRESHOLDS = {
  STREAMING: 0.8,   // 80% similarity threshold for streaming updates 
  CONSERVATIVE: 0.95, // 95% for very chatty streams  
  AGGRESSIVE: 0.75,   // 75% for fewer updates
};

interface Tool {
  name: string;
  path: string;
  description?: string;
  code?: string;
  source_code?: string;
  language?: string;
  pr_number?: number;
  pr_title?: string;
  branch_name?: string;
  custom_gpt?: boolean;
  gpt_query?: string;
  system_instruction?: string;
  query_interval?: number;
}

interface ToolRunnerProps {
  selectedTool: Tool | null;
  onBack?: () => void;
  showBackButton?: boolean;
  onNavigateToChat?: (conversationId?: string) => void; // Callback to navigate to chat tab with optional conversation ID
  isActive?: boolean; // Whether the parent tab is currently active
}

export default function ToolRunner({ 
  selectedTool,
  onBack,
  showBackButton = false,
  onNavigateToChat,
  isActive = true,
}: ToolRunnerProps) {
  const [toolOutput, setToolOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const isStreamingRef = useRef(false); // Ref to avoid stale closures in cleanup effects
  const [audioEnabled, setAudioEnabled] = useState(true); // Toggle audio output
  const [conversationMode, setConversationMode] = useState(false); // Toggle conversation mode
  const conversationModeRef = useRef(false); // Ref to track conversation mode for WebSocket handler
  const conversationIdRef = useRef<string | null>(null); // Ref to track conversation ID
  const [lastCapturedImage, setLastCapturedImage] = useState<string | null>(null); // Track last captured image for chat
  const lastCapturedImageRef = useRef<string | null>(null); // Ref to track image for WebSocket handler
  const [lastStreamingText, setLastStreamingText] = useState(''); // Track last streaming text for similarity
  const lastStreamingTextRef = useRef(''); // Backup ref to persist across re-renders
  
  // Custom GPT follow-up state
  const [isCustomGptStreaming, setIsCustomGptStreaming] = useState(false);
  const [isListeningFollowup, setIsListeningFollowup] = useState(false);
  const [followupTranscript, setFollowupTranscript] = useState('');
  const [isProcessingFollowup, setIsProcessingFollowup] = useState(false);
  
  // Keep isStreamingRef in sync with isStreaming state
  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  // Voice event listeners for follow-up speech-to-text
  useEffect(() => {
    Voice.onSpeechPartialResults = (e: SpeechResultsEvent) => {
      if (e.value && e.value[0]) {
        setFollowupTranscript(e.value[0]);
      }
    };
    Voice.onSpeechResults = (e: SpeechResultsEvent) => {
      if (e.value && e.value[0]) {
        const transcript = e.value[0];
        setFollowupTranscript(transcript);
        console.log('[ToolRunner] Follow-up speech final transcript:', transcript);
        // Just store the final transcript — user taps "Stop & Send" to actually send
      }
    };
    Voice.onSpeechError = (e: SpeechErrorEvent) => {
      console.error('[ToolRunner] Follow-up speech error:', e.error);
      setIsListeningFollowup(false);
      // Cancel Voice to clean up
      Voice.cancel().catch(() => {});
      // Resume query loop if we errored out
      const ws = (WebSocketService as any).ws;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'resume_live_query' }));
      }
    };
    return () => {
      Voice.destroy().then(Voice.removeAllListeners);
    };
  }, []);

  // Follow-up handler functions
  const startFollowupListening = async () => {
    if (!isCustomGptStreaming || !isStreaming) return;
    console.log('[ToolRunner] Starting follow-up listening');
    // Always cancel any existing Voice session first to avoid "already started" error
    try {
      await Voice.cancel();
    } catch (_) {
      // Ignore - no session to cancel
    }
    // Stop any ongoing TTS so it doesn't interfere with mic input
    try {
      TextToSpeechService.stop();
    } catch (_) {
      // Ignore
    }
    // Pause the query loop on server
    const ws = (WebSocketService as any).ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'pause_live_query' }));
    }
    setFollowupTranscript('');
    setIsListeningFollowup(true);
    try {
      // Play a short ping to indicate recording has started
      await BeepService.playBeep(880, 150);
      await Voice.start('en-US');
      console.log('[ToolRunner] Voice.start succeeded - listening for follow-up');
    } catch (e) {
      console.error('[ToolRunner] Failed to start Voice:', e);
      setIsListeningFollowup(false);
      // Resume query loop since we failed to start
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'resume_live_query' }));
      }
    }
  };

  // Use a ref to access current transcript from stopFollowupListening
  const followupTranscriptRef = useRef('');
  useEffect(() => {
    followupTranscriptRef.current = followupTranscript;
  }, [followupTranscript]);

  const stopFollowupListening = async () => {
    console.log('[ToolRunner] Stopping follow-up listening');
    setIsListeningFollowup(false);
    try {
      // Use Voice.stop() to get final results (cancel discards them)
      await Voice.stop();
    } catch (e) {
      console.error('[ToolRunner] Failed to stop Voice:', e);
    }
    // Give a moment for the final onSpeechResults callback to fire and update the ref
    await new Promise<void>(resolve => setTimeout(() => resolve(), 300));
    // Send whatever transcript we have
    const currentTranscript = followupTranscriptRef.current;
    console.log('[ToolRunner] Follow-up transcript after stop:', JSON.stringify(currentTranscript));
    if (currentTranscript.trim()) {
      sendFollowup(currentTranscript);
    } else {
      console.log('[ToolRunner] No follow-up transcript captured, resuming query loop');
      // Nothing captured, just resume the query loop
      const ws = (WebSocketService as any).ws;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'resume_live_query' }));
      }
    }
  };

  const sendFollowup = (text: string) => {
    if (!text.trim()) {
      // Nothing to send, resume query loop
      const ws = (WebSocketService as any).ws;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'resume_live_query' }));
      }
      setIsListeningFollowup(false);
      return;
    }
    console.log('[ToolRunner] Sending follow-up:', text);
    setIsProcessingFollowup(true);
    setIsListeningFollowup(false);
    const ws = (WebSocketService as any).ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'live_followup',
        text: text.trim(),
      }));
    }
  };

  // Track last one-shot result for conversation follow-up
  const [lastOneShotResult, setLastOneShotResult] = useState<{
    result: string;
    image: string;
    conversationId: string;
  } | null>(null);

  // Function to register a conversation image with the backend
  const registerConversationImage = (conversationId: string, imageBase64: string) => {
    console.log('[ToolRunner] Registering conversation image with backend');
    console.log('[ToolRunner] Conversation ID:', conversationId);
    console.log('[ToolRunner] Image base64 length (before):', imageBase64.length);
    console.log('[ToolRunner] Image base64 preview:', imageBase64.substring(0, 100));

    const ws = (WebSocketService as any).ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      // Extract just the base64 part (remove data:image/jpeg;base64, prefix if present)
      // Use a more robust approach: find the comma and take everything after it
      let base64Data = imageBase64;
      const commaIndex = imageBase64.indexOf(',');
      if (commaIndex !== -1) {
        base64Data = imageBase64.substring(commaIndex + 1);
        console.log('[ToolRunner] Stripped prefix at comma index:', commaIndex);
      } else {
        console.log('[ToolRunner] No comma found, using full string as base64');
      }
      
      // Remove any whitespace
      base64Data = base64Data.trim().replace(/\s/g, '');
      
      console.log('[ToolRunner] Image base64 length (after cleanup):', base64Data.length);
      console.log('[ToolRunner] Base64 preview (cleaned):', base64Data.substring(0, 100));

      const message = {
        type: 'register_conversation_image',
        conversation_id: conversationId,
        image_base64: base64Data,
        timestamp: Date.now(),
      };

      ws.send(JSON.stringify(message));
      console.log('[ToolRunner] Sent conversation image registration to backend');
    } else {
      console.error('[ToolRunner] WebSocket not open, cannot register conversation image');
    }
  };

  // Function to save result to chat when in conversation mode
  const saveToChat = (result: string, conversationId?: string, imageUri?: string) => {
    console.log('[ToolRunner] saveToChat called');
    console.log('[ToolRunner] Result:', result.substring(0, 100));
    console.log('[ToolRunner] Conversation ID:', conversationId);
    console.log('[ToolRunner] Image URI length:', imageUri?.length || 0);
    console.log('[ToolRunner] Tool name:', selectedTool?.name);
    
    const storageKey = 'chatSessions';
    console.log('[ToolRunner] Using storage key:', storageKey);
    
    // Get existing chat sessions
    AsyncStorage.getItem(storageKey)
      .then((sessionsData) => {
        const sessions = sessionsData ? JSON.parse(sessionsData) : [];
        console.log('[ToolRunner] Existing sessions count:', sessions.length);

        // Create descriptive title: "Tool Name - Date at Time"
        const now = new Date();
        const dateStr = now.toLocaleDateString();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const toolName = selectedTool?.name || 'Unknown Tool';
        const title = `${toolName} - ${dateStr} at ${timeStr}`;

        // Create new chat session
        const newChatId = Date.now().toString();
        const newChat: any = {
          id: newChatId,
          title,
          toolName,
          messages: [],
          conversationId: conversationId, // Store conversation ID for follow-up questions
          imageUri: imageUri, // Store image URI for display
          createdAt: new Date().toISOString(),
          lastUpdated: new Date().toISOString(),
        };

        // Add assistant message with the result
        newChat.messages.push({
          id: `${Date.now()}_assistant`,
          type: 'assistant',
          content: result,
          timestamp: new Date().toISOString(),
        });

        console.log('[ToolRunner] Chat has', newChat.messages.length, 'messages');
        console.log('[ToolRunner] Message types:', newChat.messages.map((m: any) => m.type).join(', '));
        
        // Save updated sessions
        const updatedSessions = [newChat, ...sessions];
        console.log('[ToolRunner] About to save', updatedSessions.length, 'sessions to AsyncStorage');
        console.log('[ToolRunner] New chat ID:', newChatId);
        console.log('[ToolRunner] Conversation ID stored:', conversationId);
        console.log('[ToolRunner] Image URI stored:', !!imageUri);
        
        return AsyncStorage.setItem(storageKey, JSON.stringify(updatedSessions));
      })
      .then(() => {
        console.log('[ToolRunner] Result saved to chat session successfully');
      })
      .catch((error) => {
        console.error('[ToolRunner] Error saving to chat:', error);
      });
  };

  // Wrapper for setLastStreamingText with logging and ref backup
  const setLastStreamingTextWithLogging = (text: string) => {
    console.log('[ToolRunner] setLastStreamingText called with:', JSON.stringify(text));
    console.log('[ToolRunner] Stack trace for debugging:', new Error().stack?.split('\n').slice(0, 5));
    setLastStreamingText(text);
    lastStreamingTextRef.current = text; // Also store in ref
    console.log('[ToolRunner] Also set ref to:', JSON.stringify(text));
  };

  // Track component lifecycle and state changes
  useEffect(() => {
    console.log('[ToolRunner] Component mounted/re-rendered, lastStreamingText:', JSON.stringify(lastStreamingText));
    return () => {
      console.log('[ToolRunner] Component cleanup, lastStreamingText was:', JSON.stringify(lastStreamingText));
    };
  }, []);

  useEffect(() => {
    console.log('[ToolRunner] lastStreamingText state changed to:', JSON.stringify(lastStreamingText));
  }, [lastStreamingText]);

  // Clear saved one-shot result when tool changes
  useEffect(() => {
    setLastOneShotResult(null);
    console.log('[ToolRunner] Tool changed, cleared saved one-shot result');
  }, [selectedTool?.name]);

  // Function to calculate cosine similarity between two texts
  const calculateCosineSimilarity = (text1: string, text2: string): number => {
    if (!text1 || !text2) return 0;
    if (text1.trim() === text2.trim()) {
      console.log('[Similarity Debug] Texts are identical, returning 1.0');
      return 1.0; // Exact match
    }
    
    // Normalize and tokenize - split by spaces, punctuation, and convert to lowercase
    const normalize = (text: string) => text.toLowerCase()
      .replace(/[^\w\s]/g, ' ') // Replace punctuation with spaces
      .split(/\s+/)
      .filter(word => word.length > 0);
    
    const words1 = normalize(text1);
    const words2 = normalize(text2);
    
    console.log('[Similarity Debug] Text1 tokens:', words1.slice(0, 5), '... length:', words1.length);
    console.log('[Similarity Debug] Text2 tokens:', words2.slice(0, 5), '... length:', words2.length);
    
    if (words1.length === 0 || words2.length === 0) {
      console.log('[Similarity Debug] One text has no tokens, returning 0');
      return 0;
    }
    
    // Check if the normalized word lists are identical
    if (words1.length === words2.length && words1.every((word, i) => word === words2[i])) {
      console.log('[Similarity Debug] Normalized texts are identical, returning 1.0');
      return 1.0;
    }
    
    // Create word frequency maps for better accuracy
    const getWordFreq = (words: string[]) => {
      const freq: { [key: string]: number } = {};
      words.forEach(word => freq[word] = (freq[word] || 0) + 1);
      return freq;
    };
    
    const freq1 = getWordFreq(words1);
    const freq2 = getWordFreq(words2);
    
    // Get all unique words
    const allWords = Array.from(new Set([...Object.keys(freq1), ...Object.keys(freq2)]));
    console.log('[Similarity Debug] Unique words:', allWords.slice(0, 5), '... total:', allWords.length);
    
    if (allWords.length === 0) {
      console.log('[Similarity Debug] No unique words found, returning 0');
      return 0;
    }
    
    // Create frequency vectors
    const vector1 = allWords.map(word => freq1[word] || 0);
    const vector2 = allWords.map(word => freq2[word] || 0);
    
    // Calculate cosine similarity (NOT distance)
    const dotProduct = vector1.reduce((sum, val, i) => sum + val * vector2[i], 0);
    const magnitude1 = Math.sqrt(vector1.reduce((sum, val) => sum + val * val, 0));
    const magnitude2 = Math.sqrt(vector2.reduce((sum, val) => sum + val * val, 0));
    
    console.log('[Similarity Debug] Vectors1:', vector1.slice(0, 5));
    console.log('[Similarity Debug] Vectors2:', vector2.slice(0, 5));
    console.log('[Similarity Debug] Dot product:', dotProduct, 'Mag1:', magnitude1.toFixed(2), 'Mag2:', magnitude2.toFixed(2));
    
    if (magnitude1 === 0 || magnitude2 === 0) {
      console.log('[Similarity Debug] Zero magnitude detected, returning 0');
      return 0;
    }
    
    const similarity = dotProduct / (magnitude1 * magnitude2);
    console.log('[Similarity Debug] Raw similarity:', similarity, 'Should be between 0 and 1');
    
    // Ensure result is between 0 and 1 (cosine similarity should be)
    const clampedSimilarity = Math.max(0, Math.min(1, similarity));
    console.log('[Similarity Debug] Final clamped similarity:', clampedSimilarity);
    return clampedSimilarity;
  };

  // Function to announce text with similarity check
  const announceIfDifferent = (newText: string, threshold: number = 0.8): void => {
    console.log('[ToolRunner] === announceIfDifferent called ===');
    console.log('[ToolRunner] newText length:', newText.length, 'content:', JSON.stringify(newText));
    console.log('[ToolRunner] threshold:', threshold);
    console.log('[ToolRunner] audioEnabled:', audioEnabled, 'conversationMode:', conversationMode);
    console.log('[ToolRunner] lastStreamingText (state) length:', lastStreamingText.length, 'content:', JSON.stringify(lastStreamingText));
    console.log('[ToolRunner] lastStreamingTextRef (ref) length:', lastStreamingTextRef.current.length, 'content:', JSON.stringify(lastStreamingTextRef.current));
    
    // In conversation mode, we don't speak results - they go to chat instead
    // In audio mode, we speak if audio is enabled
    if (conversationMode) {
      console.log('[ToolRunner] Conversation mode active - not speaking streaming results');
      return;
    }
    
    if (!audioEnabled || !newText?.trim()) {
      console.log('[ToolRunner] Exiting early - audioEnabled:', audioEnabled, 'conversationMode:', conversationMode, 'newText valid:', !!newText?.trim());
      return;
    }
    
    // Use ref as backup if state is empty
    const lastText = lastStreamingText || lastStreamingTextRef.current;
    console.log('[ToolRunner] Using lastText:', JSON.stringify(lastText));
    
    // Check for exact match first
    const lastTrimmed = lastText.trim();
    const newTrimmed = newText.trim();
    console.log('[ToolRunner] Exact match check - lastTrimmed === newTrimmed?', lastTrimmed === newTrimmed);
    console.log('[ToolRunner] lastTrimmed:', JSON.stringify(lastTrimmed));
    console.log('[ToolRunner] newTrimmed:', JSON.stringify(newTrimmed));
    
    if (lastTrimmed === newTrimmed) {
      console.log('[ToolRunner] Texts are IDENTICAL - skipping announcement');
      return;
    }
    
    // If this is the first announcement, always announce
    if (!lastText.trim()) {
      console.log('[ToolRunner] First streaming announcement, always announcing');
      AccessibilityInfo.announceForAccessibility(newText);
      setLastStreamingTextWithLogging(newText);
      return;
    }
    
    // For very short texts, use simple string comparison as fallback
    if (newText.length < 10 || lastText.length < 10) {
      const areVeryDifferent = newText.trim() !== lastText.trim();
      console.log('[ToolRunner] Short text fallback, different?', areVeryDifferent);
      if (areVeryDifferent) {
        AccessibilityInfo.announceForAccessibility(newText);
        setLastStreamingTextWithLogging(newText);
      }
      return;
    }
    
    const similarity = calculateCosineSimilarity(lastText, newText);
    console.log('[ToolRunner] Comparing texts:');
    console.log('[ToolRunner] Previous FULL:', lastText);
    console.log('[ToolRunner] Current FULL: ', newText);
    console.log('[ToolRunner] Text similarity:', similarity.toFixed(3), 'threshold:', threshold);
    console.log('[ToolRunner] Will announce?', similarity < threshold ? 'YES (different enough)' : 'NO (too similar)');
    
    // Only announce if the text is sufficiently different
    if (similarity < threshold) {
      console.log('[ToolRunner] Announcing new text (similarity below threshold)');
      AccessibilityInfo.announceForAccessibility(newText);
      setLastStreamingTextWithLogging(newText);
    } else {
      console.log('[ToolRunner] Skipping announcement (text too similar, similarity:', similarity.toFixed(3), ')');
    }
  };
  const toolNameRef = useRef<Text>(null);
  const cameraViewRef = useRef<CameraViewHandle>(null);

  // Set accessibility focus to tool name when component mounts or tool changes
  useEffect(() => {
    if (!selectedTool) return;
    
    const timeout = setTimeout(() => {
      if (toolNameRef.current) {
        const reactTag = findNodeHandle(toolNameRef.current);
        if (reactTag) {
          AccessibilityInfo.setAccessibilityFocus(reactTag);
        }
      }
    }, 100);
    
    return () => clearTimeout(timeout);
  }, [selectedTool?.name]);

  // Cleanup: stop streaming when component unmounts or tool changes
  useEffect(() => {
    return () => {
      if (isStreaming) {
        console.log('[ToolRunner] Cleaning up: stopping camera streaming');
        
        if (cameraViewRef.current) {
          cameraViewRef.current.stopStreaming();
        }
      }
    };
  }, [isStreaming]);

  // Loading sound effect - play when running one-shot tools, but NOT when streaming (too much noise)
  useEffect(() => {
    let beepTimer: ReturnType<typeof setTimeout> | null = null;
    const isLoading = isRunning; // Only beep for one-shot tools, not streaming
    
    if (isLoading) {
      console.log('[ToolRunner] Tool running, will beep after 3 seconds if still processing');
      // Wait 3 seconds before starting beep
      beepTimer = setTimeout(() => {
        console.log('[ToolRunner] 3 seconds elapsed, starting loading sound');
        BeepService.startLoadingSound();
      }, 3000);
    } else {
      console.log('[ToolRunner] Stopping loading sound');
      BeepService.stopLoadingSound();
    }

    // Cleanup on unmount or when loading state changes
    return () => {
      if (beepTimer) {
        clearTimeout(beepTimer);
      }
      BeepService.stopLoadingSound();
    };
  }, [isRunning, isStreaming]);

  useEffect(() => {
    // Listen for tool results from backend
    const handleMessage = (event: any) => {
      try {
        const message = JSON.parse(event.data);
        console.log('[ToolRunner] Received message type:', message.type);
        
        if (message.type === 'tool_result') {
          console.log('[ToolRunner] Tool result received:', message.status);
          console.log('[ToolRunner] Raw message.result length:', message.result?.length);
          console.log('[ToolRunner] Result/Error:', message.result || message.error);
          console.log('[ToolRunner] Audio config:', message.audio);
          
          setIsRunning(false);
          if (message.status === 'success') {
            const result = message.result || 'Tool executed successfully';
            setToolOutput(result);
            
            // In conversation mode, save to chat instead of speaking
            // CRITICAL: Use ref.current, not state, because this handler is a closure
            console.log('[ToolRunner] Tool result received. conversationMode state:', conversationMode, 'ref:', conversationModeRef.current);
            console.log('[ToolRunner] Image state length:', lastCapturedImage?.length || 0, 'ref length:', lastCapturedImageRef.current?.length || 0);
            if (conversationModeRef.current) {
              console.log('[ToolRunner] ===== CONVERSATION MODE ACTIVE =====');
              console.log('[ToolRunner] Calling saveToChat with:');
              console.log('[ToolRunner]   - result length:', result.length);
              console.log('[ToolRunner]   - result preview:', result.substring(0, 100));
              console.log('[ToolRunner]   - conversationId:', conversationIdRef.current);
              console.log('[ToolRunner]   - imageUri length:', lastCapturedImageRef.current?.length || 0);
              console.log('[ToolRunner] =====================================');
              saveToChat(result, conversationIdRef.current || undefined, lastCapturedImageRef.current || undefined);
              
              // Handle audio output from tool result
              const textToAnnounce = (message.audio && message.audio.text) || result;
              if (message.audio && message.audio.type) {
                // Tool returned advanced audio config - play via AudioOutputService
                console.log('[ToolRunner] Playing audio type:', message.audio.type);
                AudioOutputService.play({
                  type: message.audio.type,
                  text: textToAnnounce,
                  rate: message.audio.rate,
                  interrupt: message.audio.interrupt,
                });
              }
              // Announce via VoiceOver with queue option for better reliability
              console.log('[ToolRunner] Announcing conversation result via accessibility (' + textToAnnounce.length + ' chars)');
              AccessibilityInfo.announceForAccessibilityWithOptions(textToAnnounce, { queue: true });
              
              // Reset conversation mode after saving to chat
              setConversationMode(false);
              conversationModeRef.current = false;
              conversationIdRef.current = null;
              console.log('[ToolRunner] Reset conversationMode state and ref to false');
            } else {
              // One-shot mode: Save result for potential conversation follow-up
              if (lastCapturedImageRef.current) {
                const convId = `conv_${Date.now()}`;
                console.log('[ToolRunner] Saving one-shot result for conversation follow-up');
                console.log('[ToolRunner]   - Image ref length:', lastCapturedImageRef.current.length);
                console.log('[ToolRunner]   - Image ref preview:', lastCapturedImageRef.current.substring(0, 100));
                console.log('[ToolRunner]   - Is data URI?', lastCapturedImageRef.current.startsWith('data:'));
                setLastOneShotResult({
                  result,
                  image: lastCapturedImageRef.current,
                  conversationId: convId
                });
                console.log('[ToolRunner] Saved one-shot result for conversation follow-up');
              }
              
              if (audioEnabled) {
                // Check if we have custom audio config from tool result
                const textToAnnounce = (message.audio && message.audio.text) || result;
                
                // Handle audio type (beep_high, success, etc.) if present
                if (message.audio && message.audio.type) {
                  console.log('[ToolRunner] Playing audio type:', message.audio.type);
                  AudioOutputService.play({
                    type: message.audio.type,
                    text: textToAnnounce,
                    rate: message.audio.rate,
                    interrupt: message.audio.interrupt,
                  });
                }
                
                // Announce via VoiceOver with queue option for better reliability
                console.log('[ToolRunner] Announcing tool result via accessibility (' + textToAnnounce.length + ' chars)');
                AccessibilityInfo.announceForAccessibilityWithOptions(textToAnnounce, { queue: true });
                // Don't update streaming text tracking for one-shot results
              }
            }
          } else {
            const error = `Error: ${message.error || 'Unknown error'}`;
            setToolOutput(error);
            
            // In conversation mode, save error to chat; otherwise announce
            if (conversationMode) {
              console.log('[ToolRunner] Saving error to chat (conversation mode)');
              saveToChat(error, lastCapturedImage || undefined);
              
              // Also announce the error aloud
              console.log('[ToolRunner] Announcing conversation error via accessibility:', error);
              AccessibilityInfo.announceForAccessibility(error);
              
              // Reset conversation mode after saving to chat
              setConversationMode(false);
            } else if (audioEnabled) {
              // Announce error via accessibility (always announce errors)
              console.log('[ToolRunner] Announcing error via accessibility:', error);
              AccessibilityInfo.announceForAccessibility(error);
              // Don't update streaming text tracking for errors
            }
          }
        } else if (message.type === 'tool_stream_result') {
          // Streaming result - update continuously
          console.log('[ToolRunner] Stream result:', message.result?.substring(0, 100));
          console.log('[ToolRunner] Stream audio config:', message.audio);
          console.log('[ToolRunner] audioEnabled state:', audioEnabled);
          const result = message.result || 'Processing...';
          setToolOutput(result);
          
          // Use accessibility announcement for streaming results with similarity checking
          console.log('[ToolRunner] About to check streaming result for similarity...');
          if (conversationMode) {
            // In conversation mode, we don't speak streaming results but will save final result to chat
            console.log('[ToolRunner] Conversation mode active - not speaking streaming results');
          } else if (audioEnabled && result) {
            console.log('[ToolRunner] Checking streaming result for announcement:', result.substring(0, 50));
            
            // Check if we have custom audio config
            const textToAnnounce = (message.audio && message.audio.text) || result;
            console.log('[ToolRunner] Full text to announce:', textToAnnounce);
            
            // Handle audio type (beep_high, success, etc.) if present
            if (message.audio && message.audio.type) {
              console.log('[ToolRunner] Playing streaming audio type:', message.audio.type);
              AudioOutputService.play({
                type: message.audio.type,
                text: textToAnnounce,
                rate: message.audio.rate,
                interrupt: message.audio.interrupt,
              });
            }
            
            // Use similarity check for streaming results (VoiceOver)
            console.log('[ToolRunner] Calling announceIfDifferent with threshold:', SIMILARITY_THRESHOLDS.STREAMING);
            announceIfDifferent(textToAnnounce, SIMILARITY_THRESHOLDS.STREAMING);
          } else {
            console.log('[ToolRunner] NOT checking similarity - audioEnabled:', audioEnabled, 'result exists:', !!result);
          }
        } else if (message.type === 'streaming_started') {
          console.log('[ToolRunner] Server confirmed streaming started');
          const mode = message.mode || 'code_execution';
          console.log('[ToolRunner] Streaming mode:', mode);
          setIsStreaming(true);
          setIsCustomGptStreaming(mode === 'gemini_live');
          setToolOutput(`Streaming started: ${message.tool_name || 'tool'}${mode === 'gemini_live' ? ' (Gemini Live)' : ''}`);
          
          if (!conversationMode && audioEnabled) {
            AudioOutputService.play({
              type: 'success' as any,
              text: mode === 'gemini_live' ? 'Gemini Live streaming started' : 'Streaming started',
            });
          }
        } else if (message.type === 'live_followup_response') {
          // Handle Gemini Live follow-up responses
          console.log('[ToolRunner] Live follow-up response:', message.text?.substring(0, 100));
          setIsProcessingFollowup(false);
          setFollowupTranscript('');
          if (!message.error && message.text) {
            setToolOutput(message.text);
            if (audioEnabled) {
              AudioOutputService.play({
                type: 'speech' as any,
                text: message.text,
                rate: 1.0,
                interrupt: false,
              });
              announceIfDifferent(message.text, SIMILARITY_THRESHOLDS.STREAMING);
            }
          } else if (message.error) {
            console.error('[ToolRunner] Live follow-up error:', message.text);
          }
        } else if (message.type === 'streaming_stopped') {
          console.log('[ToolRunner] Server confirmed streaming stopped');
          setIsStreaming(false);
          setIsCustomGptStreaming(false);
          setIsListeningFollowup(false);
          setFollowupTranscript('');
          setIsProcessingFollowup(false);
          
          // In conversation mode, save the final streaming result to chat
          if (conversationMode && toolOutput && toolOutput !== 'Streaming stopped') {
            console.log('[ToolRunner] Saving final streaming result to chat:', toolOutput.substring(0, 100));
            saveToChat(toolOutput, lastCapturedImage || undefined);
          }
          
          setToolOutput('Streaming stopped');
          
          // Stop camera streaming when server confirms stop
          if (cameraViewRef.current) {
            cameraViewRef.current.stopStreaming();
          }
          
          if (!conversationMode && audioEnabled) {
            AccessibilityInfo.announceForAccessibility('Streaming stopped');
            // Don't update streaming text tracking for status messages
          }
        } else if (message.type === 'tool_stream_error') {
          console.log('[ToolRunner] Stream error:', message.error);
          const error = `Stream Error: ${message.error}`;
          setToolOutput(error);
          
          // Stop streaming on error
          setIsStreaming(false);
          
          if (cameraViewRef.current) {
            cameraViewRef.current.stopStreaming();
          }
          
          // In conversation mode, save error to chat; otherwise announce
          if (conversationMode) {
            console.log('[ToolRunner] Saving stream error to chat (conversation mode)');
            saveToChat(error, lastCapturedImage || undefined);
          } else if (audioEnabled) {
            AccessibilityInfo.announceForAccessibility(error);
            // Don't update streaming text tracking for stream errors
          }
        } else if (message.type === 'module_installing') {
          console.log('[ToolRunner] Module installing:', message.message);
          const msg = `Installing module: ${message.message}`;
          setToolOutput(msg);
          
          if (!conversationMode && audioEnabled) {
            AccessibilityInfo.announceForAccessibility('Installing module');
            // Don't update streaming text tracking for status messages
          }
        } else if (message.type === 'module_installed') {
          console.log('[ToolRunner] Module installed:', message.module);
          const msg = `Module installed: ${message.module}. Resuming...`;
          setToolOutput(msg);
          
          if (!conversationMode && audioEnabled) {
            const announcement = `Module ${message.module} installed`;
            AccessibilityInfo.announceForAccessibility(announcement);
            // Don't update streaming text tracking for status messages
          }
        } else if (message.type === 'module_install_failed') {
          console.log('[ToolRunner] Module install failed:', message.error);
          const error = `Module install failed: ${message.error}`;
          setToolOutput(error);
          setIsStreaming(false);  // Stop streaming on install failure
          
          // Stop camera streaming on module install failure
          if (cameraViewRef.current) {
            cameraViewRef.current.stopStreaming();
          }
          
          // In conversation mode, save error to chat; otherwise announce
          if (conversationMode) {
            console.log('[ToolRunner] Saving module install error to chat (conversation mode)');
            saveToChat(error, lastCapturedImage || undefined);
          } else if (audioEnabled) {
            AccessibilityInfo.announceForAccessibility(error);
            // Don't update streaming text tracking for errors
          }
        }
      } catch (error) {
        console.error('[ToolRunner] Error parsing WebSocket message:', error);
      }
    };

    const ws = (WebSocketService as any).ws;
    console.log('[ToolRunner] Setting up message listener, ws exists:', !!ws);
    
    //use websockets to listen for tool results
    if (ws) {
      ws.addEventListener('message', handleMessage);
      return () => {
        ws.removeEventListener('message', handleMessage);
      };
    }
  }, []);

  const handleRunTool = async (isConversation: boolean = false) => {
    if (!selectedTool) return;

    console.log('[ToolRunner] Starting tool execution, isConversation:', isConversation);
    console.log('[ToolRunner] Tool:', selectedTool.name);
    console.log('[ToolRunner] Has code:', !!selectedTool.code);
    console.log('[ToolRunner] Language:', selectedTool.language);

    // Set conversation mode if this is a conversation run - update BOTH state and ref
    if (isConversation) {
      setConversationMode(true);
      conversationModeRef.current = true;
      console.log('[ToolRunner] Set conversationMode state and ref to true');
    } else {
      setConversationMode(false);
      conversationModeRef.current = false;
      console.log('[ToolRunner] Set conversationMode state and ref to false');
    }

    setIsRunning(true);
    setToolOutput('Capturing frame...');

    // Capture a single frame for one-shot execution
    const frameData = await cameraViewRef.current?.captureFrame();
    
    console.log('[ToolRunner] captureFrame returned:', frameData ? 'data' : 'null/undefined');
    if (frameData) {
      console.log('[ToolRunner] Frame data keys:', Object.keys(frameData));
      console.log('[ToolRunner] Frame has base64:', !!frameData.base64);
      console.log('[ToolRunner] Frame base64 length:', frameData.base64?.length || 0);
      console.log('[ToolRunner] Frame dimensions:', frameData.width, 'x', frameData.height);
    }
    
    if (!frameData) {
      const errorMsg = 'Error: Could not capture camera frame. Please ensure camera is active.';
      setToolOutput(errorMsg);
      setIsRunning(false);
      
      // Announce error via VoiceOver
      AccessibilityInfo.announceForAccessibility('Could not capture camera frame. Please ensure camera is active.');
      
      if (!conversationMode && audioEnabled) {
        AudioOutputService.play({
          type: 'error' as any,
          text: 'Could not capture camera frame',
        });
      }
      return;
    }

    // Always store the captured image (needed for conversation mode and potential future features)
    if (frameData.base64) {
      // frameData.base64 already includes the data URI prefix from CameraView
      const imageUri = frameData.base64;
      setLastCapturedImage(imageUri);
      lastCapturedImageRef.current = imageUri; // Also update ref for WebSocket handler
      console.log('[ToolRunner] Stored captured image in state and ref');
      console.log('[ToolRunner]   - Image URI length:', imageUri.length);
      console.log('[ToolRunner]   - Image URI prefix:', imageUri.substring(0, 50));
      console.log('[ToolRunner]   - conversationMode:', conversationMode);
      console.log('[ToolRunner]   - conversationModeRef.current:', conversationModeRef.current);
    } else {
      lastCapturedImageRef.current = null;
      console.log('[ToolRunner] No base64 data in frameData');
    }

    setToolOutput('Running tool...');

    // Generate conversation ID if this is a conversation run
    const conversationId = isConversation ? `conv_${Date.now()}` : undefined;
    if (conversationId) {
      conversationIdRef.current = conversationId; // Store in ref for WebSocket handler
      console.log('[ToolRunner] Generated and stored conversation ID:', conversationId);
    } else {
      conversationIdRef.current = null;
    }

    // Send tool execution request to backend with captured frame
    const message = {
      type: 'run_tool',
      tool_name: selectedTool.name,
      tool_path: selectedTool.path,
      tool_code: selectedTool.code,
      tool_language: selectedTool.language,
      input: '',
      frame: {
        base64: frameData.base64,
        width: frameData.width,
        height: frameData.height,
      },
      conversation_id: conversationId, // Include conversation ID if in conversation mode
      timestamp: Date.now(),
    };

    const ws = (WebSocketService as any).ws;
    console.log('[ToolRunner] WebSocket state:', ws ? ws.readyState : 'no ws');
    
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
      console.log('[ToolRunner] Sent tool execution request with frame:', selectedTool.name);
      console.log('[ToolRunner] Frame size:', frameData.width, 'x', frameData.height);
    } else {
      console.error('[ToolRunner] WebSocket not open. ReadyState:', ws?.readyState);
      const errorMsg = 'Error: Not connected to server';
      setToolOutput(errorMsg);
      setIsRunning(false);
      
      // Announce error via VoiceOver
      AccessibilityInfo.announceForAccessibility('Not connected to server. Please connect first.');
    }
  };

  const startStreamingTool = () => {
    if (!selectedTool) return;

    console.log('[ToolRunner] Starting streaming mode');
    
    // Start camera frame streaming automatically
    if (cameraViewRef.current) {
      console.log('[ToolRunner] Starting camera frame streaming');
      cameraViewRef.current.startStreaming();
    } else {
      console.warn('[ToolRunner] Camera ref not available');
    }
    
    // Set state optimistically for immediate UI feedback
    setIsStreaming(true);
    setToolOutput('Starting stream...');
    
    const message: any = {
      type: 'start_streaming_tool',
      tool_name: selectedTool.name,
      tool_code: selectedTool.code,
      tool_language: selectedTool.language,
      input: '',
      throttle_ms: 1000, // Process 1 frame per second
    };

    // Forward custom GPT / Gemini Live fields if present
    if (selectedTool.custom_gpt) {
      message.custom_gpt = true;
      message.gpt_query = selectedTool.gpt_query || '';
      message.system_instruction = selectedTool.system_instruction || '';
      message.query_interval = selectedTool.query_interval || 3.0;
      console.log('[ToolRunner] Starting Gemini Live streaming mode, query:', selectedTool.gpt_query);
    }

    const ws = (WebSocketService as any).ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
      console.log('[ToolRunner] Sent streaming start request');
      
      // Set a timeout in case server doesn't respond
      setTimeout(() => {
        if (isStreaming) {
          console.log('[ToolRunner] Server did not confirm streaming started, assuming started');
        }
      }, 2000);
    } else {
      console.error('[ToolRunner] WebSocket not open');
      const errorMsg = 'Error: Not connected to server';
      setToolOutput(errorMsg);
      setIsStreaming(false);  // Revert state on failure
      
      // Announce error via VoiceOver
      AccessibilityInfo.announceForAccessibility('Not connected to server. Please connect first.');
      
      // Stop camera streaming if websocket failed
      if (cameraViewRef.current) {
        cameraViewRef.current.stopStreaming();
      }
    }
  };

  const stopStreamingTool = () => {
    console.log('[ToolRunner] Stopping streaming mode');
    
    // Stop camera frame streaming automatically
    if (cameraViewRef.current) {
      console.log('[ToolRunner] Stopping camera frame streaming');
      cameraViewRef.current.stopStreaming();
    }
    
    // Set state optimistically for immediate UI feedback
    setIsStreaming(false);
    setToolOutput('Streaming stopped');
    
    // Clear custom GPT / follow-up state
    setIsCustomGptStreaming(false);
    setIsListeningFollowup(false);
    setFollowupTranscript('');
    setIsProcessingFollowup(false);
    
    const message = { 
      type: 'stop_streaming_tool',
    };

    const ws = (WebSocketService as any).ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
      console.log('[ToolRunner] Sent streaming stop request');
    } else {
      console.error('[ToolRunner] WebSocket not open when stopping stream');
    }
  };

  // Cleanup streaming on unmount or tool change
  useEffect(() => {
    return () => {
      if (isStreamingRef.current) {
        stopStreamingTool();
      }
    };
  }, [selectedTool]);

  // Stop streaming when tab becomes inactive (e.g. user switches tabs)
  useEffect(() => {
    if (!isActive && isStreamingRef.current) {
      console.log('[ToolRunner] Tab became inactive — stopping streaming');
      stopStreamingTool();
    }
  }, [isActive]);
  
  const renderTool = () => {
    if (!selectedTool) {
      return (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyIcon}>🔧</Text>
          <Text style={styles.emptyText}>No Tool Selected</Text>
          <Text style={styles.emptySubtext}>
            Go to the Tools tab to select a tool to run
          </Text>
        </View>
      );
    }

    // All tools get the camera view - they're all camera-based
    // Tools loaded from GitHub PRs will process the camera images
    return (
      <View style={styles.toolContainer}>
        {/* Camera View - Takes up most of screen */}
        <View style={styles.cameraSection}>
          <CameraView ref={cameraViewRef} />
        </View>

        {/* Tool Controls - Fixed bottom section */}
        {selectedTool.path !== 'camera' && (
          <View style={styles.controlsSection}>
            {/* Fixed controls - always visible */}
            <View style={styles.fixedControls}>
              {/* Tool name */}
              <View style={styles.toolHeader}>
                <Text 
                  ref={toolNameRef}
                  style={styles.toolNameCompact} 
                  numberOfLines={1}
                  accessible={true}
                  accessibilityRole="header"
                  accessibilityLabel={`Tool: ${selectedTool.name}`}>
                  {selectedTool.name}
                </Text>
              </View>

              {/* Main action buttons - Full width, readable */}
              <View style={styles.buttonRow}>
                <TouchableOpacity
                  style={[styles.button, isStreaming ? styles.stopButton : styles.streamButton, isRunning && styles.buttonDisabled]}
                  onPress={isStreaming ? stopStreamingTool : startStreamingTool}
                  disabled={isRunning}
                  accessible={true}
                  accessibilityLabel={isStreaming ? "Stop streaming" : "Start streaming"}
                  accessibilityHint={isStreaming ? "Stops continuous tool execution" : "Starts continuous tool execution on camera frames"}
                  accessibilityRole="button"
                  accessibilityState={{ disabled: isRunning }}>
                  <Text style={styles.buttonText}>
                    {isStreaming ? 'Stop' : 'Stream'}
                  </Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={[styles.button, styles.oneShotButton, (isRunning || isStreaming) && styles.buttonDisabled]}
                  onPress={() => handleRunTool(false)}
                  disabled={isRunning || isStreaming}
                  accessible={true}
                  accessibilityLabel={isRunning ? "Processing" : "Take photo"}
                  accessibilityHint="Runs the tool once on the current camera frame"
                  accessibilityRole="button"
                  accessibilityState={{ disabled: isRunning || isStreaming }}>
                  <Text style={styles.buttonText}>
                    {isRunning ? 'Processing...' : 'Take Photo'}
                  </Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={[
                    styles.button, 
                    styles.conversationButton, 
                    (!lastOneShotResult || isRunning || isStreaming) && styles.buttonDisabled
                  ]}
                  onPress={() => {
                    console.log('[ToolRunner] Conversation button pressed');
                    if (lastOneShotResult) {
                      // Register the conversation image with the backend first
                      console.log('[ToolRunner] Starting conversation with saved one-shot result');
                      console.log('[ToolRunner]   - lastOneShotResult.image length:', lastOneShotResult.image.length);
                      console.log('[ToolRunner]   - lastOneShotResult.image preview:', lastOneShotResult.image.substring(0, 100));
                      console.log('[ToolRunner]   - Is data URI?', lastOneShotResult.image.startsWith('data:'));
                      
                      registerConversationImage(
                        lastOneShotResult.conversationId,
                        lastOneShotResult.image
                      );
                      
                      // Then save to local chat
                      saveToChat(
                        lastOneShotResult.result,
                        lastOneShotResult.conversationId,
                        lastOneShotResult.image
                      );
                      
                      // Clear the saved result after using it
                      setLastOneShotResult(null);
                      
                      // Navigate to Chat tab with the conversation ID
                      if (onNavigateToChat) {
                        onNavigateToChat(lastOneShotResult.conversationId);
                        AccessibilityInfo.announceForAccessibility('Navigating to chat. Conversation ready.');
                      } else {
                        AccessibilityInfo.announceForAccessibility('Conversation created. Go to Chat tab to continue.');
                      }
                    }
                  }}
                  disabled={!lastOneShotResult || isRunning || isStreaming}
                  accessible={true}
                  accessibilityLabel={lastOneShotResult ? "Start conversation from last result" : "Start conversation, take photo first"}
                  accessibilityHint={lastOneShotResult ? "Creates a new conversation using the last photo result" : "Button disabled until you take a photo"}
                  accessibilityRole="button"
                  accessibilityState={{ disabled: !lastOneShotResult || isRunning || isStreaming }}>
                  <Text style={styles.buttonText}>
                    💬 Conversation
                  </Text>
                </TouchableOpacity>
              </View>

              {/* Follow-up mic button - shown during custom GPT streaming */}
              {isCustomGptStreaming && isStreaming && (
                <View style={styles.followupSection}>
                  <TouchableOpacity
                    style={[
                      styles.followupButton,
                      isListeningFollowup && styles.followupButtonActive,
                      isProcessingFollowup && styles.buttonDisabled,
                    ]}
                    onPress={isListeningFollowup ? stopFollowupListening : startFollowupListening}
                    disabled={isProcessingFollowup}
                    accessible={true}
                    accessibilityLabel={
                      isProcessingFollowup
                        ? 'Processing follow-up question'
                        : isListeningFollowup
                        ? 'Stop listening, tap to send follow-up'
                        : 'Ask a follow-up question'
                    }
                    accessibilityHint="Pauses the tool and lets you ask a follow-up question by voice"
                    accessibilityRole="button"
                    accessibilityState={{ disabled: isProcessingFollowup }}>
                    <Text style={styles.followupButtonText}>
                      {isProcessingFollowup
                        ? '⏳ Processing...'
                        : isListeningFollowup
                        ? '⏹️ Stop & Send'
                        : '🎤 Follow-up'}
                    </Text>
                  </TouchableOpacity>
                  {followupTranscript ? (
                    <Text
                      style={styles.followupTranscript}
                      numberOfLines={2}
                      accessible={true}
                      accessibilityRole="text"
                      accessibilityLabel={`You said: ${followupTranscript}`}>
                      {followupTranscript}
                    </Text>
                  ) : null}
                </View>
              )}

              {/* Output section - visible but compact with accessibility live region */}
              {toolOutput && (
                <View 
                  style={styles.outputSection}
                  accessible={true}
                  accessibilityLiveRegion="polite"
                  accessibilityRole="text"
                  accessibilityLabel={`Tool output: ${toolOutput}`}
                  accessibilityHint="Long press to copy text">
                  <Text 
                    style={styles.outputText} 
                    numberOfLines={2}
                    selectable={true}
                    accessible={false}>
                    {toolOutput}
                  </Text>
                </View>
              )}
            </View>

            {/* Scrollable details - takes remaining space */}
            <ScrollView 
              style={styles.detailsScroll}
              contentContainerStyle={styles.detailsContent}
              keyboardShouldPersistTaps="handled"
              showsVerticalScrollIndicator={true}>
              <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
                <View>
                  {selectedTool.description && (
                    <View style={styles.detailSection}>
                      <Text style={styles.detailTitle}>Description</Text>
                      <Text 
                        style={styles.toolDescription}
                        selectable={true}
                        accessible={true}
                        accessibilityRole="text"
                        accessibilityLabel={`Description: ${selectedTool.description}`}
                        accessibilityHint="Long press to copy text">
                        {selectedTool.description}
                      </Text>
                    </View>
                  )}
                  
                  {(selectedTool.branch_name || selectedTool.language || selectedTool.pr_title) && (
                    <View style={styles.detailSection}>
                      <Text style={styles.detailTitle}>Details</Text>
                      {selectedTool.pr_title && (
                        <Text 
                          style={styles.toolMeta}
                          selectable={true}
                          accessible={true}
                          accessibilityRole="text">
                          PR: {selectedTool.pr_title}
                        </Text>
                      )}
                      {selectedTool.branch_name && (
                        <Text 
                          style={styles.toolMeta}
                          selectable={true}
                          accessible={true}
                          accessibilityRole="text">
                          Branch: {selectedTool.branch_name}
                        </Text>
                      )}
                      {selectedTool.language && (
                        <Text 
                          style={styles.toolMeta}
                          selectable={true}
                          accessible={true}
                          accessibilityRole="text">
                          Language: {selectedTool.language}
                        </Text>
                      )}
                    </View>
                  )}

                  {selectedTool.source_code && (
                    <View style={styles.detailSection}>
                      <Text style={styles.detailTitle}>Source Code</Text>
                      <View style={styles.codeScroll}>
                        <Text 
                          style={styles.codeText}
                          selectable={true}
                          accessible={true}
                          accessibilityRole="text"
                          accessibilityLabel="Source code"
                          accessibilityHint="Long press to copy code">
                          {selectedTool.source_code}
                        </Text>
                      </View>
                    </View>
                  )}
                </View>
              </TouchableWithoutFeedback>
            </ScrollView>
          </View>
        )}
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container} edges={[]}>
      {/* Back Button */}
      {showBackButton && onBack && (
        <View style={styles.backButtonContainer}>
          <TouchableOpacity
            style={styles.backButton}
            onPress={onBack}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Back to tool selector"
            accessibilityHint="Double tap to return to tool selection">
            <Text style={styles.backButtonText}>← Back to Tools</Text>
          </TouchableOpacity>
        </View>
      )}
      
      {selectedTool && (
        <View style={styles.header}>
          <Text style={styles.headerText}>Running: {selectedTool.name}</Text>
        </View>
      )}
      <View style={styles.content}>
        {renderTool()}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  backButtonContainer: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: '#fff',
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    alignSelf: 'flex-start',
  },
  backButtonText: {
    fontSize: 16,
    color: '#2563eb',
    fontWeight: '600',
  },
  header: {
    backgroundColor: '#2196F3',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#1976D2',
  },
  headerText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  content: {
    flex: 1,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 32,
  },
  emptyIcon: {
    fontSize: 64,
    marginBottom: 16,
  },
  emptyText: {
    fontSize: 20,
    fontWeight: '600',
    color: '#666',
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
    lineHeight: 20,
  },
  toolContainer: {
    flex: 1,
  },
  cameraSection: {
    flex: 1, // Camera takes ALL available space
    backgroundColor: '#000',
  },
  controlsSection: {
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    maxHeight: '30%', // Controls limited to 30% max
  },
  fixedControls: {
    // Controls that are always visible (buttons, output)
    backgroundColor: '#fff',
  },
  toolHeader: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#f9f9f9',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  toolNameCompact: {
    fontSize: 13,
    fontWeight: '600',
    color: '#333',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: '#fff',
  },
  button: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  outputSection: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: '#f5f5f5',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    maxHeight: 60,
  },
  outputText: {
    fontSize: 13,
    color: '#333',
    lineHeight: 18,
  },
  detailsScroll: {
    maxHeight: 200, // Constrain height to force scrolling
    backgroundColor: '#fafafa',
  },
  detailsContent: {
    paddingBottom: 20, // Ensure content can scroll fully
  },
  detailSection: {
    backgroundColor: '#fff',
    padding: 12,
    marginTop: 1,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  detailTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: '#666',
    marginBottom: 6,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  toolInfo: {
    backgroundColor: '#fff',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  toolName: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  toolDescription: {
    fontSize: 13,
    color: '#666',
    lineHeight: 18,
  },
  toolMeta: {
    fontSize: 11,
    color: '#999',
    marginTop: 2,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    backgroundColor: '#fafafa',
    minHeight: 80,
    textAlignVertical: 'top',
  },
  oneShotButton: {
    backgroundColor: '#2196F3',
  },
  streamButton: {
    backgroundColor: '#4CAF50',
  },
  stopButton: {
    backgroundColor: '#f44336',
  },
  audioButton: {
    backgroundColor: '#FF9800',
  },
  audioButtonDisabled: {
    backgroundColor: '#9E9E9E',
  },
  conversationButton: {
    backgroundColor: '#2196F3',
  },
  conversationButtonActive: {
    backgroundColor: '#4CAF50',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  runButton: {
    backgroundColor: '#4CAF50',
    margin: 16,
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  runButtonDisabled: {
    backgroundColor: '#9E9E9E',
  },
  runButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  output: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    minHeight: 100,
  },
  codeScroll: {
    backgroundColor: '#2b2b2b',
    borderRadius: 6,
    padding: 10,
    maxHeight: 200,
  },
  codeText: {
    fontSize: 11,
    color: '#d4d4d4',
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
  },
  followupSection: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#f0f0f0',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    gap: 8,
  },
  followupButton: {
    backgroundColor: '#FF9800',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 120,
  },
  followupButtonActive: {
    backgroundColor: '#f44336',
  },
  followupButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  followupTranscript: {
    flex: 1,
    fontSize: 13,
    color: '#333',
    fontStyle: 'italic',
  },
});
