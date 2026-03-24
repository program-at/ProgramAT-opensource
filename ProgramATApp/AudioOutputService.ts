/**
 * Audio Output Service
 * Provides multiple audio output mechanisms for tools
 * 
 * Supported Output Types:
 * 1. Text-to-Speech (TTS) - Natural voice output
 * 2. Beep/Tone - Simple notification sounds
 * 3. Earcon - Non-speech audio icons (success, error, warning)
 * 4. Spearcon - Speed-compressed speech (faster than TTS)
 * 5. Auditory Icons - Environmental sounds (click, whoosh, etc.)
 * 
 * @format
 */

import Tts from 'react-native-tts';
import { Platform, Vibration } from 'react-native';
import BeepService from './BeepService';
// Note: For full audio support, we'll need react-native-sound
// npm install react-native-sound

/**
 * Audio output types that tools can request
 */
export enum AudioOutputType {
  // Speech output
  SPEECH = 'speech',                    // Full text-to-speech
  SPEECH_FAST = 'speech_fast',         // Faster TTS (1.5x speed)
  SPEECH_SLOW = 'speech_slow',         // Slower TTS (0.7x speed)
  
  // Non-speech sounds
  BEEP_LOW = 'beep_low',               // Low frequency beep (200Hz)
  BEEP_MID = 'beep_mid',               // Mid frequency beep (440Hz)
  BEEP_HIGH = 'beep_high',             // High frequency beep (880Hz)
  
  // Earcons (abstract audio cues)
  SUCCESS = 'success',                 // Task completed successfully
  ERROR = 'error',                     // Error occurred
  WARNING = 'warning',                 // Warning/caution
  INFO = 'info',                       // Informational notification
  PROGRESS = 'progress',               // Task in progress
  
  // Auditory icons (environmental sounds)
  CLICK = 'click',                     // Button/interaction click
  WHOOSH = 'whoosh',                   // Transition/movement
  POP = 'pop',                         // Popup/appearance
  CHIME = 'chime',                     // Gentle notification
  
  // Haptic feedback (vibration patterns)
  VIBRATE_SHORT = 'vibrate_short',     // Quick vibration
  VIBRATE_LONG = 'vibrate_long',       // Sustained vibration
  VIBRATE_PATTERN = 'vibrate_pattern', // Patterned vibration
  
  // Silent (for tools that want to suppress output)
  SILENT = 'silent',
}

/**
 * Audio output parameters
 */
export interface AudioOutputParams {
  type: AudioOutputType;
  text?: string;                       // For TTS outputs
  duration?: number;                   // For beeps/tones (ms)
  frequency?: number;                  // For beeps (Hz)
  volume?: number;                     // 0.0 to 1.0
  rate?: number;                       // TTS speech rate (0.5 to 2.0)
  pitch?: number;                      // TTS pitch (0.5 to 2.0)
  interrupt?: boolean;                 // Stop current audio before playing
  vibrationPattern?: number[];         // For custom vibration patterns
}

class AudioOutputService {
  private ttsInitialized: boolean = false;
  private ttsAvailable: boolean = true;
  private currentlySpeaking: boolean = false;
  
  /**
   * Initialize TTS engine
   */
  async initialize() {
    if (this.ttsInitialized) return;
    
    try {
      if (Platform.OS === 'ios' || Platform.OS === 'android') {
        await Tts.setDefaultLanguage('en-US');
        await Tts.setDefaultRate(0.5);
        await Tts.setDefaultPitch(1.0);
        
        // Listen for TTS events
        Tts.addEventListener('tts-start', () => {
          this.currentlySpeaking = true;
          console.log('[AudioOutput] TTS started');
        });
        
        Tts.addEventListener('tts-finish', () => {
          this.currentlySpeaking = false;
          console.log('[AudioOutput] TTS finished');
        });
        
        Tts.addEventListener('tts-cancel', () => {
          this.currentlySpeaking = false;
          console.log('[AudioOutput] TTS cancelled');
        });
        
        console.log('[AudioOutput] TTS initialized successfully');
      }
      this.ttsInitialized = true;
    } catch (error) {
      console.warn('[AudioOutput] TTS not available:', error);
      this.ttsAvailable = false;
      this.ttsInitialized = true;
    }
  }
  
  /**
   * Main method to play audio output
   */
  async play(params: AudioOutputParams): Promise<void> {
    console.log('[AudioOutput] Playing:', params.type, params.text?.substring(0, 50));
    
    // Initialize if needed
    if (!this.ttsInitialized) {
      await this.initialize();
    }
    
    // Stop current audio if interrupt requested
    if (params.interrupt) {
      await this.stopAll();
    }
    
    // Route to appropriate handler based on type
    switch (params.type) {
      // Speech outputs
      case AudioOutputType.SPEECH:
        await this.playSpeech(params.text || '', params.rate || 0.5, params.pitch || 1.0);
        break;
      case AudioOutputType.SPEECH_FAST:
        await this.playSpeech(params.text || '', 1.5, params.pitch || 1.0);
        break;
      case AudioOutputType.SPEECH_SLOW:
        await this.playSpeech(params.text || '', 0.7, params.pitch || 1.0);
        break;
      
      // Beep outputs - play beep followed by text description
      case AudioOutputType.BEEP_LOW:
        await this.playBeep(200, params.duration || 200, params.text);
        break;
      case AudioOutputType.BEEP_MID:
        await this.playBeep(440, params.duration || 200, params.text);
        break;
      case AudioOutputType.BEEP_HIGH:
        await this.playBeep(880, params.duration || 200, params.text);
        break;
      
      // Earcon outputs (abstract audio cues)
      case AudioOutputType.SUCCESS:
        await this.playEarcon('success');
        break;
      case AudioOutputType.ERROR:
        await this.playEarcon('error');
        break;
      case AudioOutputType.WARNING:
        await this.playEarcon('warning');
        break;
      case AudioOutputType.INFO:
        await this.playEarcon('info');
        break;
      case AudioOutputType.PROGRESS:
        await this.playEarcon('progress');
        break;
      
      // Auditory icons
      case AudioOutputType.CLICK:
        await this.playAuditoryIcon('click');
        break;
      case AudioOutputType.WHOOSH:
        await this.playAuditoryIcon('whoosh');
        break;
      case AudioOutputType.POP:
        await this.playAuditoryIcon('pop');
        break;
      case AudioOutputType.CHIME:
        await this.playAuditoryIcon('chime');
        break;
      
      // Haptic feedback
      case AudioOutputType.VIBRATE_SHORT:
        Vibration.vibrate(100);
        break;
      case AudioOutputType.VIBRATE_LONG:
        Vibration.vibrate(500);
        break;
      case AudioOutputType.VIBRATE_PATTERN:
        Vibration.vibrate(params.vibrationPattern || [0, 100, 50, 100]);
        break;
      
      // Silent (no output)
      case AudioOutputType.SILENT:
        console.log('[AudioOutput] Silent output requested');
        break;
      
      default:
        console.warn('[AudioOutput] Unknown output type:', params.type);
    }
  }
  
  /**
   * Play text-to-speech
   */
  private async playSpeech(text: string, rate: number = 0.5, pitch: number = 1.0): Promise<void> {
    if (!this.ttsAvailable) {
      console.log('[AudioOutput] Would speak (TTS unavailable):', text);
      return;
    }
    
    try {
      await Tts.setDefaultRate(rate);
      await Tts.setDefaultPitch(pitch);
      await Tts.speak(text);
    } catch (error) {
      console.error('[AudioOutput] TTS failed:', error);
      this.ttsAvailable = false;
    }
  }
  
  /**
   * Play a beep tone at specified frequency
   * Pure vibration - no speech, no audio files
   */
  private async playBeep(frequency: number, duration: number, text?: string): Promise<void> {
    console.log(`[AudioOutput] Beep ${frequency}Hz for ${duration}ms`);
    
    // Play vibration pattern (NO speech)
    await BeepService.playBeep(frequency, duration);
  }
  
  /**
   * Play an earcon (non-speech audio icon)
   * Earcons are abstract sounds that represent concepts
   */
  private async playEarcon(type: string): Promise<void> {
    // TODO: Load and play pre-recorded earcon sounds
    // For now, use different beep patterns
    console.log(`[AudioOutput] Playing earcon: ${type}`);
    
    switch (type) {
      case 'success':
        // Rising tone pattern
        Vibration.vibrate([0, 50, 30, 50, 30, 100]);
        break;
      case 'error':
        // Descending harsh pattern
        Vibration.vibrate([0, 100, 50, 100]);
        break;
      case 'warning':
        // Alternating pattern
        Vibration.vibrate([0, 100, 50, 100, 50, 100]);
        break;
      case 'info':
        // Single gentle pulse
        Vibration.vibrate(100);
        break;
      case 'progress':
        // Repeating short pulse
        Vibration.vibrate([0, 50]);
        break;
    }
  }
  
  /**
   * Play an auditory icon (environmental sound)
   * Auditory icons are recognizable real-world sounds
   */
  private async playAuditoryIcon(type: string): Promise<void> {
    // TODO: Load and play pre-recorded sound effects
    console.log(`[AudioOutput] Playing auditory icon: ${type}`);
    
    switch (type) {
      case 'click':
        Vibration.vibrate(10);
        break;
      case 'whoosh':
        Vibration.vibrate([0, 20, 10, 30, 20, 40]);
        break;
      case 'pop':
        Vibration.vibrate(50);
        break;
      case 'chime':
        Vibration.vibrate([0, 50, 30, 50]);
        break;
    }
  }
  
  /**
   * Stop all audio output
   */
  async stopAll(): Promise<void> {
    try {
      if (this.ttsAvailable) {
        await Tts.stop();
      }
      Vibration.cancel();
      this.currentlySpeaking = false;
      console.log('[AudioOutput] All audio stopped');
    } catch (error) {
      console.error('[AudioOutput] Stop failed:', error);
    }
  }
  
  /**
   * Check if currently speaking
   */
  isSpeaking(): boolean {
    return this.currentlySpeaking;
  }
  
  /**
   * Parse audio output from tool result
   * Tools can return special format: {audio: {...}, text: "..."}
   */
  async playFromToolResult(result: any): Promise<void> {
    console.log('[AudioOutput] playFromToolResult called with:', typeof result, result);
    
    if (!result) return;
    
    // Check if result has audio specification
    if (typeof result === 'object' && result.audio) {
      console.log('[AudioOutput] Using advanced audio format:', result.audio);
      const audioParams: AudioOutputParams = {
        type: result.audio.type || AudioOutputType.SPEECH,
        text: result.audio.text || result.text || '',
        duration: result.audio.duration,
        frequency: result.audio.frequency,
        volume: result.audio.volume,
        rate: result.audio.rate,
        pitch: result.audio.pitch,
        interrupt: result.audio.interrupt !== false, // Default to true
      };
      
      console.log('[AudioOutput] Audio params:', audioParams);
      await this.play(audioParams);
    }
    // If result is a string, default to TTS
    else if (typeof result === 'string') {
      console.log('[AudioOutput] Using simple string format, length:', result.length);
      await this.play({
        type: AudioOutputType.SPEECH,
        text: result,
        interrupt: false,
      });
    }
    // If result has text property, use TTS
    else if (result.text) {
      console.log('[AudioOutput] Using text property format');
      await this.play({
        type: AudioOutputType.SPEECH,
        text: result.text,
        interrupt: false,
      });
    }
  }
}

export default new AudioOutputService();
