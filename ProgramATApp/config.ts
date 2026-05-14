/**
 * Application Configuration
 * 
 * Centralized configuration for the app.
 * Modify these values based on your environment.
 * 
 * NOTE: For production apps, use environment variables or a secure
 * configuration management system instead of hardcoding values here.
 * This configuration is suitable for development and testing.
 * 
 * @format
 */

export type AppMode = 'development' | 'production' | 'review';

// Server configuration mapping - secret codes to server URLs
// Add new servers here as needed
export const SERVER_CONFIGS: Record<string, { url: string; name: string }> = {
  // Default server (no code needed)
  'default': {
    url: 'ws://34.144.178.116:8080',
    name: 'Default Server'
  }

  // Add additional servers with secret codes
  // Example: 'mysecret123': { url: 'ws://10.0.0.1:8080', name: 'Dev Server' },
};

export const Config = {
  // Application Mode
  // 'development': Full features including Issues tab, PR selection, GitHub integration
  // 'production': Simplified mode - only pulls tools from main branch, no Issues tab
  APP_MODE: 'development' as AppMode, // Change to 'production' for production deployment
  
  // Default WebSocket server URL
  // Set at runtime from AsyncStorage (entered by user in Settings).
  // Empty string means no server configured yet.
  WEBSOCKET_SERVER_URL: '',

  // General/review server URL — used in review mode for tool execution and PR fetching.
  // This is the shared server that hosts the tools being reviewed.
  REVIEW_SERVER_URL: 'ws://34.162.96.157:8080',

  // GitHub repo for the general/review server (owner/repo format).
  // Used in review mode so the user's server can post approve/reject to the correct repo.
  REVIEW_GITHUB_REPO: 'program-at/ProgramAT-opensource',
  
  // GitHub branch configuration
  // In production mode, tools are fetched from this branch only
  PRODUCTION_BRANCH: 'main',
  
  // Frame streaming configuration
  FRAME_CAPTURE_INTERVAL_MS: 500, // 2 FPS (1000ms / 2)
  FRAME_QUALITY_PRIORITIZATION: 'speed' as const, // 'speed' or 'quality'
  
  // Image quality settings (0.0 - 1.0)
  // Lower values = smaller file size, lower quality
  // Recommended: 0.6-0.8 for good balance
  // If frames are too large and causing disconnections, try 0.5-0.6
  IMAGE_QUALITY: 0.7,
  
  // WebSocket reconnection settings
  MAX_RECONNECT_ATTEMPTS: 5,
  RECONNECT_DELAY_MS: 3000,
  
  // Feature flags based on mode
  get ENABLE_ISSUES_TAB() {
    return this.APP_MODE === 'development';
  },
  get ENABLE_PR_SELECTION() {
    return this.APP_MODE === 'development' || this.APP_MODE === 'review';
  },
  get ENABLE_BRANCH_SELECTION() {
    return this.APP_MODE === 'development';
  },
  get ENABLE_MODE_SWITCHER() {
    return true; // Allow switching between modes in app
  },
};

export default Config;

