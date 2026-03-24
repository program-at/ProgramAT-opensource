/**
 * Theme configuration for light and dark modes
 */

export type ThemeMode = 'light' | 'dark';

export interface Theme {
  // Background colors
  background: string;
  backgroundSecondary: string;
  backgroundTertiary: string;
  
  // Text colors
  text: string;
  textSecondary: string;
  textTertiary: string;
  
  // Interactive colors
  primary: string;
  primaryDark: string;
  success: string;
  warning: string;
  error: string;
  info: string;
  
  // Border and separator colors
  border: string;
  separator: string;
  
  // Card and surface colors
  card: string;
  surface: string;
  
  // Tab bar colors
  tabBarBackground: string;
  tabBarActive: string;
  tabBarInactive: string;
  
  // Status colors
  statusConnected: string;
  statusDisconnected: string;
  statusWarning: string;
  
  // Input colors
  inputBackground: string;
  inputBorder: string;
  inputPlaceholder: string;
  
  // Overlay colors
  overlay: string;
  modalBackground: string;
}

export const lightTheme: Theme = {
  background: '#ffffff',
  backgroundSecondary: '#f5f5f5',
  backgroundTertiary: '#e8e8e8',
  
  text: '#000000',
  textSecondary: '#666666',
  textTertiary: '#999999',
  
  primary: '#007AFF',
  primaryDark: '#0051D5',
  success: '#34C759',
  warning: '#FF9500',
  error: '#FF3B30',
  info: '#5856D6',
  
  border: '#d1d1d6',
  separator: '#e5e5ea',
  
  card: '#ffffff',
  surface: '#f9f9f9',
  
  tabBarBackground: '#f9f9f9',
  tabBarActive: '#007AFF',
  tabBarInactive: '#666666',
  
  statusConnected: '#34C759',
  statusDisconnected: '#FF3B30',
  statusWarning: '#FF9500',
  
  inputBackground: '#ffffff',
  inputBorder: '#d1d1d6',
  inputPlaceholder: '#999999',
  
  overlay: 'rgba(0, 0, 0, 0.5)',
  modalBackground: '#ffffff',
};

export const darkTheme: Theme = {
  background: '#1a1a1a',        // Almost black charcoal
  backgroundSecondary: '#2a2a2a', // Lighter charcoal
  backgroundTertiary: '#3a3a3a',  // Even lighter charcoal
  
  text: '#fcfcfcff',              // Light gray for primary text
  textSecondary: '#d2d1d1ff',     // Medium gray for secondary text
  textTertiary: '#b0b0b0',      // Darker gray for tertiary text
  
  primary: '#62a9faff',           // Brighter blue for dark mode
  primaryDark: '#4A9EFF',       // Darker variant
  success: '#40D566',           // Bright green
  warning: '#FFB340',           // Bright orange
  error: '#FF5249',             // Bright red
  info: '#7C7AE8',              // Bright purple
  
  border: '#444444',            // Dark gray border
  separator: '#333333',         // Darker separator
  
  card: '#242424',              // Card background
  surface: '#2e2e2e',           // Surface color
  
  tabBarBackground: '#202020',
  tabBarActive: '#4A9EFF',
  tabBarInactive: '#d2d1d1ff',
  
  statusConnected: '#40D566',
  statusDisconnected: '#FF5249',
  statusWarning: '#FFB340',
  
  inputBackground: '#2a2a2a',
  inputBorder: '#444444',
  inputPlaceholder: '#808080',
  
  overlay: 'rgba(0, 0, 0, 0.7)',
  modalBackground: '#242424',
};

export function getTheme(mode: ThemeMode): Theme {
  return mode === 'dark' ? darkTheme : lightTheme;
}
