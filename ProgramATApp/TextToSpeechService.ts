/**
 * TextToSpeech Service
 * Static wrapper for text-to-speech functionality
 *
 * @format
 */

import Tts from 'react-native-tts';
import { Platform } from 'react-native';

class TextToSpeechService {
  private static initialized: boolean = false;
  private static available: boolean = true;
  private static isSpeaking: boolean = false;
  private static lastSpokenText: string = '';
  private static lastSpeakTime: number = 0;
  private static pendingText: string | null = null;
  private static pendingRate: number = 1.0;
  private static speakingTimeoutId: any = null;

  /**
   * Initialize TTS
   */
  static async initialize() {
    if (this.initialized) return;

    try {
      // Try to initialize TTS - may fail if native module not linked
      if (Platform.OS === 'ios' || Platform.OS === 'android') {
        await Tts.setDefaultLanguage('en-US');
        
        // Set default speech rate to 1.5x speed
        try {
          await Tts.setDefaultRate(1.5);
          console.log('[TTS] Default rate set to 1.5x');
        } catch (rateError) {
          console.warn('[TTS] Could not set default rate:', rateError);
        }
        
        // Listen for TTS events
        Tts.addEventListener('tts-start', () => {
          console.log('[TTS] Event: tts-start');
          this.isSpeaking = true;
        });
        
        Tts.addEventListener('tts-finish', () => {
          console.log('[TTS] Event: tts-finish');
          this.isSpeaking = false;
          
          // Clear timeout if set
          if (this.speakingTimeoutId) {
            clearTimeout(this.speakingTimeoutId);
            this.speakingTimeoutId = null;
          }
          
          // After finishing, speak pending text if any
          this.speakPending();
        });
        
        Tts.addEventListener('tts-cancel', () => {
          console.log('[TTS] Event: tts-cancel');
          this.isSpeaking = false;
          
          // Clear timeout if set
          if (this.speakingTimeoutId) {
            clearTimeout(this.speakingTimeoutId);
            this.speakingTimeoutId = null;
          }
          
          // After cancel, speak pending text if any
          this.speakPending();
        });
        
        console.log('[TTS] Initialized successfully');
      }
      this.initialized = true;
    } catch (error) {
      console.warn('[TTS] Not available - native module may not be linked:', error);
      this.available = false;
      this.initialized = true; // Mark as initialized to prevent retries
    }
  }

  /**
   * Speak pending text (called after TTS finishes)
   */
  private static async speakPending(): Promise<void> {
    console.log('[TTS] speakPending called, pendingText:', this.pendingText?.substring(0, 30), 'isSpeaking:', this.isSpeaking);
    
    if (this.pendingText && !this.isSpeaking) {
      const textToSpeak = this.pendingText;
      const rate = this.pendingRate;
      this.pendingText = null;
      this.pendingRate = 1.0;
      
      console.log('[TTS] Speaking queued text:', textToSpeak.substring(0, 50));
      this.isSpeaking = true;
      this.lastSpokenText = textToSpeak;
      this.lastSpeakTime = Date.now();
      
      try {
        await Tts.speak(textToSpeak);
        console.log('[TTS] Queued speech completed successfully');
      } catch (error) {
        console.log('[TTS] Speak error:', error?.toString().substring(0, 100));
        this.isSpeaking = false;
      }
    }
  }

  /**
   * Speak text with optional rate control and interruption
   * @param text Text to speak
   * @param rate Speech rate (default 1.0)
   * @param interrupt If true, stops current speech and speaks immediately (default false)
   */
  static async speak(text: string, rate: number = 1.0, interrupt: boolean = false): Promise<void> {
    console.log('[TTS] speak() called, text:', text.substring(0, 30), 'interrupt:', interrupt, 'isSpeaking:', this.isSpeaking);
    
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.available) {
      console.log('[TTS] Would speak (TTS unavailable):', text);
      return;
    }

    // If interruption is requested and currently speaking
    if (interrupt && this.isSpeaking) {
      console.log('[TTS] Interrupting current speech for new text');
      
      // Clear any pending text since we're interrupting with new content
      this.pendingText = null;
      
      // For iOS: Don't call stop, just directly speak new text (natural interruption)
      if (Platform.OS === 'ios') {
        console.log('[TTS] iOS: Direct speech interruption');
        this.isSpeaking = true;
        this.lastSpokenText = text;
        this.lastSpeakTime = Date.now();
        
        try {
          await Tts.speak(text);
          console.log('[TTS] iOS interruption completed successfully');
        } catch (error) {
          console.log('[TTS] iOS interruption error:', error);
          this.isSpeaking = false;
        }
        return;
      } else {
        // Android: Use stop then speak
        try {
          Tts.stop();
          console.log('[TTS] Android: stopped, now speaking new text');
        } catch (e) {
          console.log('[TTS] Android stop error:', e);
        }
        // Small delay for Android stop to take effect
        await new Promise<void>(resolve => setTimeout(() => resolve(), 100));
        this.isSpeaking = false;
      }
    }

    // If already speaking and not interrupting, queue the latest text (replace any existing queue)
    if (this.isSpeaking && !interrupt) {
      console.log('[TTS] Queuing latest text (replacing any pending):', text.substring(0, 50));
      this.pendingText = text;
      this.pendingRate = rate;
      return;
    }

    // Speak immediately (either not currently speaking, or we just stopped for interruption)
    try {
      console.log('[TTS] Speaking NOW:', text.substring(0, 50));
      this.isSpeaking = true;
      this.lastSpokenText = text;
      this.lastSpeakTime = Date.now();
      
      await Tts.speak(text);
      console.log('[TTS] Speech completed successfully');
    } catch (error) {
      console.log('[TTS] Speak error:', error?.toString().substring(0, 100));
      this.isSpeaking = false;
    }
  }

  /**
   * Speak text with interruption (convenience method for new information)
   */
  static async speakWithInterrupt(text: string, rate: number = 1.0): Promise<void> {
    return this.speak(text, rate, true);
  }

  /**
   * Force immediate speech with maximum interruption
   */
  static async speakImmediately(text: string, rate: number = 1.0): Promise<void> {
    console.log('[TTS] speakImmediately called:', text.substring(0, 30));
    
    // Stop everything and speak immediately
    this.stopAll();
    
    // Wait a moment for stop to take effect
    await new Promise<void>(resolve => setTimeout(() => resolve(), 200));
    
    // Speak the text
    return this.speak(text, rate, false);
  }

  /**
   * Backward compatibility method - converts queueLatest to interrupt behavior
   * @deprecated Use speak() with interrupt parameter instead
   */
  static async speakWithQueue(text: string, rate: number = 1.0, queueLatest: boolean = false): Promise<void> {
    // Convert queueLatest to interrupt behavior
    // If queueLatest is true, we don't interrupt (old behavior)
    // If queueLatest is false, we interrupt (new behavior for fresh info)
    const interrupt = !queueLatest;
    return this.speak(text, rate, interrupt);
  }

  /**
   * Debug method to check current TTS state
   */
  static getDebugState(): any {
    return {
      initialized: this.initialized,
      available: this.available,
      isSpeaking: this.isSpeaking,
      hasPendingText: !!this.pendingText,
      pendingText: this.pendingText?.substring(0, 50),
      lastSpokenText: this.lastSpokenText?.substring(0, 50),
      lastSpeakTime: this.lastSpeakTime,
      timeSinceLastSpeak: Date.now() - this.lastSpeakTime,
      hasTimeout: !!this.speakingTimeoutId
    };
  }

  /**
   * Stop speaking and clear pending text
   */
  static stop(): void {
    console.log('[TTS] stop() called, isSpeaking:', this.isSpeaking, 'pendingText:', !!this.pendingText);
    
    if (!this.available) {
      console.log('[TTS] TTS not available, clearing state');
      this.isSpeaking = false;
      this.pendingText = null;
      return;
    }

    // Clear state immediately
    this.isSpeaking = false;
    this.pendingText = null;
    
    // Clear timeout if set
    if (this.speakingTimeoutId) {
      clearTimeout(this.speakingTimeoutId);
      this.speakingTimeoutId = null;
    }

    // Only try to stop TTS on Android (avoid iOS API issues)
    if (Platform.OS === 'android') {
      try {
        console.log('[TTS] Android: Calling Tts.stop()');
        Tts.stop();
      } catch (error) {
        console.log('[TTS] Android stop error:', error);
      }
    } else {
      console.log('[TTS] iOS: Skipping Tts.stop() call (state cleared)');
    }
  }

  /**
   * Force stop all speech and clear everything
   */
  static stopAll(): void {
    console.log('[TTS] stopAll() called');
    
    // Clear all state immediately
    this.isSpeaking = false;
    this.pendingText = null;
    
    // Clear any timeouts
    if (this.speakingTimeoutId) {
      clearTimeout(this.speakingTimeoutId);
      this.speakingTimeoutId = null;
    }
    
    // Use the normal stop method
    this.stop();
  }
}

export default TextToSpeechService;
