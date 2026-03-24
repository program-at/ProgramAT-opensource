/**
 * Settings Component
 * App configuration including mode switching and server connection
 */

import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  Alert,
  Switch,
  TextInput,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Config, { AppMode, SERVER_CONFIGS } from './config';
import WebSocketService from './WebSocketService';
import { useTheme } from './ThemeContext';

const CUSTOM_SERVER_KEY = '@custom_server_code';

interface SettingsProps {
  appMode: AppMode;
  onModeChange: (mode: AppMode) => void;
}

export default function Settings({ appMode, onModeChange }: SettingsProps) {
  const { theme, themeMode, toggleTheme } = useTheme();
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [secretCode, setSecretCode] = useState('');
  const [activeServerName, setActiveServerName] = useState('Default Server');
  const [currentServerUrl, setCurrentServerUrl] = useState(Config.WEBSOCKET_SERVER_URL);

  // Helper to get server name from URL
  const getServerNameFromUrl = (url: string): string => {
    for (const [code, config] of Object.entries(SERVER_CONFIGS)) {
      if (config.url === url) {
        return config.name;
      }
    }
    return 'Default Server';
  };

  // Actual connected server name (derived from current URL)
  const actualServerName = getServerNameFromUrl(currentServerUrl);

  useEffect(() => {
    // Check connection status on mount
    setIsConnected(WebSocketService.isConnected());
    setCurrentServerUrl(WebSocketService.getServerUrl());

    // Load saved secret code
    loadSavedServerCode();

    // Listen for connection changes
    const checkConnection = setInterval(() => {
      setIsConnected(WebSocketService.isConnected());
      setCurrentServerUrl(WebSocketService.getServerUrl());
    }, 1000);

    return () => clearInterval(checkConnection);
  }, []);

  const loadSavedServerCode = async () => {
    try {
      const savedCode = await AsyncStorage.getItem(CUSTOM_SERVER_KEY);
      if (savedCode) {
        setSecretCode(savedCode);
        const serverConfig = SERVER_CONFIGS[savedCode];
        if (serverConfig) {
          setActiveServerName(serverConfig.name);
          
          // Check if current connection matches saved preference
          const currentUrl = WebSocketService.getServerUrl();
          if (currentUrl !== serverConfig.url) {
            console.log('[Settings] Server mismatch detected!');
            console.log('[Settings]   Saved server:', serverConfig.name, serverConfig.url);
            console.log('[Settings]   Current URL:', currentUrl);
            console.log('[Settings] Auto-reconnecting to saved server...');
            
            // Auto-reconnect to the saved server
            WebSocketService.setServerUrl(serverConfig.url, true);
          } else {
            console.log('[Settings] Server matches saved preference:', serverConfig.name);
          }
        } else {
          // Invalid saved code, clear it
          console.log('[Settings] Invalid saved code, clearing:', savedCode);
          await AsyncStorage.removeItem(CUSTOM_SERVER_KEY);
          setActiveServerName('Default Server');
        }
      }
    } catch (error) {
      console.error('[Settings] Error loading saved server code:', error);
    }
  };

  const handleApplySecretCode = async () => {
    const trimmedCode = secretCode.trim();
    
    // Check if code exists in config
    const serverConfig = SERVER_CONFIGS[trimmedCode] || SERVER_CONFIGS['default'];
    
    if (trimmedCode && !SERVER_CONFIGS[trimmedCode]) {
      Alert.alert(
        'Invalid Code',
        'The secret code entered is not recognized. Using default server.',
        [{ text: 'OK' }]
      );
      setSecretCode('');
      await AsyncStorage.removeItem(CUSTOM_SERVER_KEY);
      setActiveServerName('Default Server');
      
      // Switch to default server
      WebSocketService.setServerUrl(SERVER_CONFIGS['default'].url, true);
      return;
    }
    
    // Save the code
    if (trimmedCode) {
      await AsyncStorage.setItem(CUSTOM_SERVER_KEY, trimmedCode);
    } else {
      await AsyncStorage.removeItem(CUSTOM_SERVER_KEY);
    }
    
    setActiveServerName(serverConfig.name);
    
    Alert.alert(
      'Server Changed',
      `Connecting to ${serverConfig.name}...`,
      [{ text: 'OK' }]
    );
    
    // Disconnect and reconnect with new URL
    WebSocketService.setServerUrl(serverConfig.url, true);
  };

  const handleClearSecretCode = async () => {
    Alert.alert(
      'Clear Server Code?',
      'This will reset to the default server.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: async () => {
            setSecretCode('');
            await AsyncStorage.removeItem(CUSTOM_SERVER_KEY);
            setActiveServerName('Default Server');
            
            // Switch to default server
            WebSocketService.setServerUrl(SERVER_CONFIGS['default'].url, true);
          }
        }
      ]
    );
  };

  useEffect(() => {
    // Check connection status on mount
    setIsConnected(WebSocketService.isConnected());

    // Listen for connection changes
    const checkConnection = setInterval(() => {
      setIsConnected(WebSocketService.isConnected());
    }, 1000);

    return () => clearInterval(checkConnection);
  }, []);

  const handleModeSwitch = () => {
    const newMode: AppMode = appMode === 'development' ? 'production' : 'development';
    
    Alert.alert(
      'Switch App Mode?',
      `Switch to ${newMode} mode?\n\n` +
      (newMode === 'production' 
        ? '• Text Input tab will be hidden\n• Issues tab will be hidden\n• Tools will only load from main branch\n• Production tools only'
        : '• Text Input tab will be shown\n• Issues tab will be shown\n• Tools can load from any PR/branch\n• Full development features enabled'),
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Switch',
          onPress: () => {
            onModeChange(newMode);
            Alert.alert(
              'Mode Changed',
              `Now running in ${newMode} mode`,
              [{ text: 'OK' }]
            );
          }
        }
      ]
    );
  };

  const handleConnect = async () => {
    if (isConnected) {
      // Disconnect
      Alert.alert(
        'Disconnect from Server?',
        'This will stop all streaming and close the connection.',
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Disconnect',
            style: 'destructive',
            onPress: () => {
              WebSocketService.disconnect();
              setIsConnected(false);
            }
          }
        ]
      );
    } else {
      // Connect
      setIsConnecting(true);
      try {
        await WebSocketService.connect();
        setIsConnected(true);
      } catch (error) {
        Alert.alert(
          'Connection Failed',
          `Could not connect to server at ${Config.WEBSOCKET_SERVER_URL}\n\nError: ${error}`,
          [{ text: 'OK' }]
        );
      } finally {
        setIsConnecting(false);
      }
    }
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.background }]} edges={['bottom']}>
      <ScrollView style={styles.scrollView}>
        {/* Header */}
        <View style={styles.header}>
          <Text 
            style={[styles.headerText, { color: theme.text }]}
            accessible={true}
            accessibilityRole="header"
            accessibilityLabel="Settings">
            Settings
          </Text>
          <Text style={[styles.headerSubtext, { color: theme.textSecondary }]} accessible={false}>
            Configure app behavior and server connection
          </Text>
        </View>

        {/* Theme Section */}
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, { color: theme.text }]} accessibilityRole="header">Appearance</Text>
          
          <View style={[styles.settingCard, { backgroundColor: theme.card, borderColor: theme.border }]}>
            <View style={styles.settingRow}>
              <View style={styles.settingInfo}>
                <Text style={[styles.settingLabel, { color: theme.text }]}>Dark Mode</Text>
                <Text style={[styles.settingDescription, { color: theme.textSecondary }]}>
                  {themeMode === 'dark' ? 'Dark theme enabled' : 'Light theme enabled'}
                </Text>
              </View>
              <Switch
                value={themeMode === 'dark'}
                onValueChange={toggleTheme}
                trackColor={{ false: theme.border, true: theme.primary }}
                thumbColor={themeMode === 'dark' ? '#ffffff' : '#f4f3f4'}
                accessible={true}
                accessibilityRole="switch"
                accessibilityLabel="Dark mode toggle"
                accessibilityHint="Double tap to toggle between light and dark themes"
              />
            </View>
          </View>
        </View>

        {/* App Mode Section */}
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, { color: theme.text }]} accessibilityRole="header">App Mode</Text>
          
          <View style={[styles.settingCard, { backgroundColor: theme.card, borderColor: theme.border }]}>
            <View style={styles.settingInfo}>
              <Text style={[styles.settingLabel, { color: theme.text }]}>Current Mode</Text>
              <View style={[
                styles.modeBadge,
                appMode === 'production' ? styles.productionBadge : styles.developmentBadge,
                { backgroundColor: (appMode === 'production' ? theme.primary : theme.info) + '20' }
              ]}>
                <Text style={[styles.modeBadgeText, { color: appMode === 'production' ? theme.primary : theme.info }]}>
                  {appMode === 'production' ? '🚀 Production' : '🔧 Development'}
                </Text>
              </View>
            </View>
            
            <TouchableOpacity
              style={[styles.switchButton, { backgroundColor: theme.primary }]}
              onPress={handleModeSwitch}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel={`Switch to ${appMode === 'development' ? 'production' : 'development'} mode`}
              accessibilityHint="Double tap to change app mode">
              <Text style={styles.switchButtonText} accessible={false}>
                Switch to {appMode === 'development' ? 'Production' : 'Development'}
              </Text>
            </TouchableOpacity>
          </View>

          <View style={[styles.modeDescription, { backgroundColor: theme.backgroundSecondary }]}>
            <Text style={[styles.modeDescriptionTitle, { color: theme.text }]}>
              {appMode === 'production' ? 'Production Mode' : 'Development Mode'}
            </Text>
            <Text style={[styles.modeDescriptionText, { color: theme.textSecondary }]}>
              {appMode === 'production' 
                ? '• Only main branch tools available\n• Simplified interface\n• Production-ready tools only'
                : '• Full development features\n• PR and branch selection\n• Create and test new tools'}
            </Text>
          </View>
        </View>

        {/* Server Connection Section */}
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, { color: theme.text }]} accessibilityRole="header">Server Connection</Text>
          
          <View style={[styles.settingCard, { backgroundColor: theme.card, borderColor: theme.border }]}>
            <View style={styles.settingInfo}>
              <Text style={[styles.settingLabel, { color: theme.textSecondary }]}>Status</Text>
              <View style={[
                styles.statusBadge,
                { backgroundColor: (isConnected ? theme.statusConnected : theme.statusDisconnected) + '20' }
              ]}>
                <View style={[
                  styles.statusDot,
                  { backgroundColor: isConnected ? theme.statusConnected : theme.statusDisconnected }
                ]} />
                <Text style={[styles.statusText, { color: theme.text }]}>
                  {isConnected ? 'Connected' : 'Disconnected'}
                </Text>
              </View>
            </View>
            
            <TouchableOpacity
              style={[
                styles.connectButton,
                isConnected 
                  ? [styles.disconnectButton, { borderColor: theme.error }] 
                  : [styles.connectButtonActive, { backgroundColor: theme.success }]
              ]}
              onPress={handleConnect}
              disabled={isConnecting}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel={isConnected ? 'Disconnect from server' : 'Connect to server'}
              accessibilityHint={`Double tap to ${isConnected ? 'disconnect from' : 'connect to'} the server`}>
              <Text 
                style={[
                  styles.connectButtonText,
                  isConnected && [styles.disconnectButtonText, { color: theme.error }]
                ]}
                accessible={false}>
                {isConnecting ? 'Connecting...' : isConnected ? 'Disconnect' : 'Connect'}
              </Text>
            </TouchableOpacity>
            
            {activeServerName !== 'Default Server' && (
              <View 
                style={[styles.customServerBadge, { backgroundColor: theme.warning + '30' }]}
                accessible={true}
                accessibilityRole="text"
                accessibilityLabel={`Using custom server: ${actualServerName}`}>
                <Text style={[styles.customServerBadgeText, { color: theme.warning }]}>
                  Custom: {actualServerName}
                </Text>
              </View>
            )}
          </View>

          {/* Secret Code Input */}
          <View style={[styles.secretCodeSection, { backgroundColor: theme.card, borderColor: theme.border }]}>
            <Text style={[styles.secretCodeLabel, { color: theme.text }]}>Server Code</Text>
            <View style={styles.secretCodeInputRow}>
              <TextInput
                style={[styles.secretCodeInput, { 
                  backgroundColor: theme.inputBackground, 
                  borderColor: theme.inputBorder, 
                  color: theme.text 
                }]}
                value={secretCode}
                onChangeText={setSecretCode}
                placeholder="Enter server code"
                placeholderTextColor={theme.inputPlaceholder}
                autoCapitalize="none"
                autoCorrect={false}
                secureTextEntry={true}
                accessible={true}
                accessibilityLabel="Server code input"
                accessibilityHint="Enter a secret code to connect to an alternative server"
              />
              <TouchableOpacity
                style={[
                  styles.secretCodeButton,
                  { backgroundColor: theme.primary },
                  !secretCode.trim() && styles.secretCodeButtonDisabled
                ]}
                onPress={handleApplySecretCode}
                disabled={!secretCode.trim()}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Apply server code"
                accessibilityHint="Double tap to apply the secret code and switch servers">
                <Text style={styles.secretCodeButtonText}>
                  Apply
                </Text>
              </TouchableOpacity>
            </View>
            {activeServerName !== 'Default Server' && (
              <TouchableOpacity
                style={[styles.resetServerButton, { backgroundColor: theme.error }]}
                onPress={handleClearSecretCode}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Reset to default server"
                accessibilityHint="Double tap to switch back to the default server">
                <Text style={styles.resetServerButtonText}>Reset to Default Server</Text>
              </TouchableOpacity>
            )}
          </View>

          <View style={[styles.serverInfo, { backgroundColor: theme.backgroundSecondary }]}>
            <Text style={[styles.serverInfoLabel, { color: theme.textSecondary }]} accessible={false}>Server URL:</Text>
            <Text 
              style={[styles.serverInfoValue, { color: theme.text }]}
              selectable={true}
              accessible={true}
              accessibilityRole="text"
              accessibilityLabel={`Server URL: ${currentServerUrl}`}
              accessibilityHint="Long press to copy URL">
              {currentServerUrl}
            </Text>
          </View>
        </View>

        {/* App Info Section */}
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, { color: theme.text }]} accessibilityRole="header">App Information</Text>
          
          <View style={[styles.infoCard, { backgroundColor: theme.card, borderColor: theme.border }]}>
            <View style={styles.infoRow}>
              <Text style={[styles.infoLabel, { color: theme.textSecondary }]}>Version</Text>
              <Text 
                style={[styles.infoValue, { color: theme.text }]}
                selectable={true}
                accessible={true}
                accessibilityRole="text">
                1.0.0
              </Text>
            </View>
            <View style={styles.infoRow}>
              <Text style={[styles.infoLabel, { color: theme.textSecondary }]}>Build</Text>
              <Text 
                style={[styles.infoValue, { color: theme.text }]}
                selectable={true}
                accessible={true}
                accessibilityRole="text">
                Development
              </Text>
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 20,
    borderBottomWidth: 1,
  },
  headerText: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  headerSubtext: {
    fontSize: 14,
  },
  section: {
    marginTop: 24,
    paddingHorizontal: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
  },
  settingCard: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  settingInfo: {
    marginBottom: 12,
  },
  settingLabel: {
    fontSize: 14,
    marginBottom: 6,
  },
  settingDescription: {
    fontSize: 13,
    marginTop: 2,
  },
  modeBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  productionBadge: {},
  developmentBadge: {},
  modeBadgeText: {
    fontSize: 14,
    fontWeight: '600',
  },
  switchButton: {
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    alignItems: 'center',
  },
  switchButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  modeDescription: {
    padding: 12,
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#3b82f6',
  },
  modeDescriptionTitle: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 4,
  },
  modeDescriptionText: {
    fontSize: 13,
    lineHeight: 20,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    alignSelf: 'flex-start',
  },
  connectedBadge: {},
  disconnectedBadge: {},
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  connectedDot: {},
  disconnectedDot: {},
  statusText: {
    fontSize: 14,
    fontWeight: '600',
  },
  connectButton: {
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    alignItems: 'center',
  },
  connectButtonActive: {},
  disconnectButton: {
    backgroundColor: 'transparent',
    borderWidth: 2,
  },
  connectButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  disconnectButtonText: {},
  serverInfo: {
    padding: 12,
    borderRadius: 8,
    marginTop: 8,
  },
  serverInfoLabel: {
    fontSize: 12,
    marginBottom: 4,
  },
  serverInfoValue: {
    fontSize: 14,
    fontFamily: 'monospace',
  },
  infoCard: {
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
  },
  infoLabel: {
    fontSize: 14,
  },
  infoValue: {
    fontSize: 14,
    fontWeight: '500',
  },
  customServerBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    marginTop: 12,
    alignSelf: 'flex-start',
  },
  customServerBadgeText: {
    fontSize: 14,
    fontWeight: '600',
  },
  secretCodeSection: {
    borderRadius: 12,
    padding: 16,
    marginTop: 12,
    borderWidth: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  secretCodeLabel: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
  },
  secretCodeInputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  secretCodeInput: {
    flex: 1,
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
  },
  secretCodeButton: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
  },
  secretCodeButtonDisabled: {
    backgroundColor: '#9ca3af',
  },
  secretCodeButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  resetServerButton: {
    marginTop: 12,
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  resetServerButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
});
