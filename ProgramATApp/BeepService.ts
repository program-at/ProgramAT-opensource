/**
 * Beep Service
 * Real audio beeps using react-native-audio-api Web Audio API implementation
 * 
 * Generates pure sine wave tones in real-time - actual audio, not vibration.
 * 
 * @format
 */

import { AudioContext } from 'react-native-audio-api';

class BeepService {
  private audioContext: AudioContext | null = null;
  private isPlaying: boolean = false;
  private pendingBeep: { frequency: number; duration: number } | null = null;

  constructor() {
    this.initialize();
  }

  /**
   * Initialize audio context
   */
  private async initialize(): Promise<void> {
    console.log('[Beep] Initializing BeepService...');
    try {
      console.log('[Beep] Creating AudioContext...');
      this.audioContext = new AudioContext();
      console.log('[Beep] ✓ Audio context created successfully');
      console.log('[Beep] Audio context state:', this.audioContext.state);
      console.log('[Beep] Sample rate:', this.audioContext.sampleRate);
    } catch (error) {
      console.log('[Beep] ✗ Failed to initialize audio context:', error);
      console.log('[Beep] Error details:', JSON.stringify(error));
    }
  }

  /**
   * Play pending beep (called after current beep finishes)
   */
  private async playPending(): Promise<void> {
    if (this.pendingBeep && !this.isPlaying) {
      const { frequency, duration } = this.pendingBeep;
      this.pendingBeep = null;
      console.log('[Beep] Playing queued beep:', frequency, 'Hz,', duration, 'ms');
      await this.playBeep(frequency, duration);
    }
  }

  /**
   * Play a beep tone using Web Audio API
   * Generates a pure sine wave at the specified frequency
   * If already playing, queues only the most recent beep (skips intermediate ones)
   */
  async playBeep(frequency: number = 440, duration: number = 200): Promise<void> {
    console.log('[Beep] playBeep called with', frequency, 'Hz,', duration, 'ms');
    
    // If already playing, queue this as the latest (replacing any previous pending)
    if (this.isPlaying) {
      console.log('[Beep] Already playing, queuing latest beep');
      this.pendingBeep = { frequency, duration };
      return;
    }
    
    if (!this.audioContext) {
      console.log('[Beep] ERROR: Audio context not available - initialization may have failed');
      return;
    }

    this.isPlaying = true;

    try {
      console.log('[Beep] Audio context state:', this.audioContext.state);
      console.log('[Beep] Audio context currentTime:', this.audioContext.currentTime);
      
      const now = this.audioContext.currentTime;
      
      // Create oscillator (sine wave generator)
      console.log('[Beep] Creating oscillator...');
      const oscillator = this.audioContext.createOscillator();
      oscillator.type = 'sine';
      oscillator.frequency.value = frequency;
      console.log('[Beep] Oscillator created, type:', oscillator.type, 'frequency:', oscillator.frequency.value);
      
      // Create gain node for volume control and fade envelope
      console.log('[Beep] Creating gain node...');
      const gainNode = this.audioContext.createGain();
      gainNode.gain.value = 0;
      console.log('[Beep] Gain node created, initial gain:', gainNode.gain.value);
      
      // Connect: oscillator → gain → output
      console.log('[Beep] Connecting nodes...');
      oscillator.connect(gainNode);
      gainNode.connect(this.audioContext.destination);
      console.log('[Beep] Nodes connected');
      
      // Set up fade envelope to avoid clicks
      const fadeTime = 0.005; // 5ms fade
      const durationSec = duration / 1000;
      
      console.log('[Beep] Setting up gain envelope...');
      // Fade in
      gainNode.gain.setValueAtTime(0, now);
      gainNode.gain.linearRampToValueAtTime(0.3, now + fadeTime);
      
      // Sustain
      gainNode.gain.setValueAtTime(0.3, now + fadeTime);
      gainNode.gain.setValueAtTime(0.3, now + durationSec - fadeTime);
      
      // Fade out
      gainNode.gain.linearRampToValueAtTime(0, now + durationSec);
      console.log('[Beep] Gain envelope configured');
      
      // Start and stop oscillator
      console.log('[Beep] Starting oscillator at time', now);
      oscillator.start(now);
      console.log('[Beep] Scheduling stop at time', now + durationSec);
      oscillator.stop(now + durationSec);
      
      console.log('[Beep] ✓ Successfully playing', frequency, 'Hz for', duration, 'ms');
      
      // Wait for beep to complete
      await new Promise<void>(resolve => setTimeout(resolve, duration));
      console.log('[Beep] Beep completed');
      
    } catch (error) {
      console.log('[Beep] ERROR during playback:', error);
      console.log('[Beep] Error details:', JSON.stringify(error));
    } finally {
      this.isPlaying = false;
      // After finishing, play pending beep if any
      this.playPending();
    }
  }

  /**
   * Play a sequence of beeps
   */
  async playBeepSequence(pattern: Array<{frequency: number, duration: number, pause?: number}>): Promise<void> {
    for (const beep of pattern) {
      await this.playBeep(beep.frequency, beep.duration);
      if (beep.pause) {
        await new Promise<void>(resolve => setTimeout(resolve, beep.pause));
      }
    }
  }

  /**
   * Play iOS-style loading sound (clicking noise)
   * Creates a rapid series of short clicks
   */
  async playLoadingSound(): Promise<void> {
    console.log('[Beep] Playing iOS loading sound...');
    const clickPattern = [
      { frequency: 800, duration: 50, pause: 100 },
      { frequency: 900, duration: 50, pause: 100 },
      { frequency: 1000, duration: 50, pause: 150 },
    ];
    
    await this.playBeepSequence(clickPattern);
  }

  /**
   * Play continuous loading clicks (for ongoing loading)
   */
  private loadingInterval: ReturnType<typeof setInterval> | null = null;

  startLoadingSound(): void {
    if (this.loadingInterval) {
      return; // Already playing
    }
    
    console.log('[Beep] Starting continuous loading sound...');
    this.playLoadingSound(); // Play immediately
    
    this.loadingInterval = setInterval(() => {
      this.playLoadingSound();
    }, 1000); // Repeat every second
  }

  stopLoadingSound(): void {
    if (this.loadingInterval) {
      console.log('[Beep] Stopping loading sound...');
      clearInterval(this.loadingInterval);
      this.loadingInterval = null;
    }
  }
}

export default new BeepService();
