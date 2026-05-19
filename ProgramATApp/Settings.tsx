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
import Config, { AppMode } from './config';
import WebSocketService from './WebSocketService';
import { useTheme } from './ThemeContext';

const SERVER_URL_KEY = '@server_url';

interface SettingsProps {
  appMode: AppMode;
  onModeChange: (mode: AppMode) => void;
}

export default function Settings({ appMode, onModeChange }: SettingsProps) {
  const { theme, themeMode, toggleTheme } = useTheme();
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [serverUrl, setServerUrl] = useState('');
  const [currentServerUrl, setCurrentServerUrl] = useState(Config.WEBSOCKET_SERVER_URL);

  useEffect(() => {
    setIsConnected(WebSocketService.isConnected());
    setCurrentServerUrl(WebSocketService.getServerUrl());
    loadSavedServerUrl();

    const checkConnection = setInterval(() => {
      setIsConnected(WebSocketService.isConnected());
      setCurrentServerUrl(WebSocketService.getServerUrl());
    }, 1000);

    return () => clearInterval(checkConnection);
  }, []);

  const loadSavedServerUrl = async () => {
    try {
      const saved = await AsyncStorage.getItem(SERVER_URL_KEY);
      if (saved) {
        setServerUrl(saved);
        const currentUrl = WebSocketService.getServerUrl();
        if (currentUrl !== saved) {
          WebSocketService.setServerUrl(saved, true);
        }
      }
    } catch (error) {
      console.error('[Settings] Error loading saved server URL:', error);
    }
  };

  const handleSaveServerUrl = async () => {
    const trimmed = serverUrl.trim();
    if (!trimmed) {
      Alert.alert('No URL Entered', 'Please enter your server WebSocket URL.');
      return;
    }
    if (!trimmed.startsWith('ws://') && !trimmed.startsWith('wss://')) {
      Alert.alert('Invalid URL', 'Server URL must start with ws:// or wss://');
      return;
    }
    await AsyncStorage.setItem(SERVER_URL_KEY, trimmed);
    Alert.alert('Server Saved', `Connecting to ${trimmed}...`, [{ text: 'OK' }]);
    WebSocketService.setServerUrl(trimmed, true);
  };

  const handleClearServerUrl = async () => {
    Alert.alert(
      'Clear Server URL?',
      'This will disconnect and remove the saved server address.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: async () => {
            setServerUrl('');
            await AsyncStorage.removeItem(SERVER_URL_KEY);
            WebSocketService.disconnect();
          }
        }
      ]
    );
  };

  const handleModeSwitchTo = (targetMode: AppMode) => {
    const modeDescriptions: Record<AppMode, string> = {
      production: '• Text Input tab will be hidden\n• Issues tab will be hidden\n• Tools will only load from main branch',
      development: '• Text Input tab will be shown\n• Issues tab will be shown\n• Tools can load from any PR/branch\n• Full development features enabled',
      review: '• Connect to general server to browse and test community PRs\n• Approve or reject PRs using your own GitHub identity\n• Your server handles all approvals/rejections',
    };

    Alert.alert(
      'Switch App Mode?',
      `Switch to ${targetMode} mode?\n\n${modeDescriptions[targetMode]}`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Switch',
          onPress: () => {
            onModeChange(targetMode);
            Alert.alert('Mode Changed', `Now running in ${targetMode} mode`, [{ text: 'OK' }]);
          }
        }
      ]
    );
  };

  const handleModeSwitch = () => {
    // Cycle: development → production → review → development
    const nextMode: AppMode =
      appMode === 'development' ? 'production'
      : appMode === 'production' ? 'review'
      : 'development';

    const modeDescriptions: Record<AppMode, string> = {
      production: '• Text Input tab will be hidden\n• Issues tab will be hidden\n• Tools will only load from main branch',
      development: '• Text Input tab will be shown\n• Issues tab will be shown\n• Tools can load from any PR/branch\n• Full development features enabled',
      review: '• Connect to general server to browse and test community PRs\n• Approve or reject PRs using your own GitHub identity\n• Your server handles all approvals/rejections',
    };

    Alert.alert(
      'Switch App Mode?',
      `Switch to ${nextMode} mode?\n\n${modeDescriptions[nextMode]}`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Switch',
          onPress: () => {
            onModeChange(nextMode);
            Alert.alert('Mode Changed', `Now running in ${nextMode} mode`, [{ text: 'OK' }]);
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
                { backgroundColor: (appMode === 'production' ? theme.primary : appMode === 'review' ? theme.warning : theme.info) + '20' }
              ]}>
                <Text style={[styles.modeBadgeText, { color: appMode === 'production' ? theme.primary : appMode === 'review' ? theme.warning : theme.info }]}>
                  {appMode === 'production' ? '🚀 Production' : appMode === 'review' ? '🔍 Review' : '🔧 Development'}
                </Text>
              </View>
            </View>

            {/* Explicit buttons for all three modes */}
            <View style={styles.modeButtonRow}>
              <TouchableOpacity
                style={[
                  styles.modeButton,
                  { borderColor: theme.info },
                  appMode === 'development' && { backgroundColor: theme.info + '20' }
                ]}
                onPress={() => appMode !== 'development' && handleModeSwitchTo('development')}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Switch to development mode"
                accessibilityState={{ selected: appMode === 'development' }}>
                <Text style={[styles.modeButtonText, { color: theme.info }]}>🔧 Dev</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[
                  styles.modeButton,
                  { borderColor: theme.primary },
                  appMode === 'production' && { backgroundColor: theme.primary + '20' }
                ]}
                onPress={() => appMode !== 'production' && handleModeSwitchTo('production')}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Switch to production mode"
                accessibilityState={{ selected: appMode === 'production' }}>
                <Text style={[styles.modeButtonText, { color: theme.primary }]}>🚀 Prod</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[
                  styles.modeButton,
                  { borderColor: theme.warning },
                  appMode === 'review' && { backgroundColor: theme.warning + '20' }
                ]}
                onPress={() => appMode !== 'review' && handleModeSwitchTo('review')}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Switch to review mode"
                accessibilityState={{ selected: appMode === 'review' }}>
                <Text style={[styles.modeButtonText, { color: theme.warning }]}>🔍 Review</Text>
              </TouchableOpacity>
            </View>
          </View>

          <View style={[styles.modeDescription, { backgroundColor: theme.backgroundSecondary }]}>
            <Text style={[styles.modeDescriptionTitle, { color: theme.text }]}>
              {appMode === 'production' ? 'Production Mode' : appMode === 'review' ? 'Review Mode' : 'Development Mode'}
            </Text>
            <Text style={[styles.modeDescriptionText, { color: theme.textSecondary }]}>
              {appMode === 'production'
                ? '• Only main branch tools available\n• Simplified interface\n• Production-ready tools only'
                : appMode === 'review'
                ? '• Browse and test community PRs from the general server\n• Approve or reject using your own GitHub identity\n• Tool frames run on the general server'
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
          </View>

          {/* Server URL Input */}
          <View style={[styles.secretCodeSection, { backgroundColor: theme.card, borderColor: theme.border }]}>
            <Text style={[styles.secretCodeLabel, { color: theme.text }]}>Server URL</Text>
            <Text style={[styles.settingDescription, { color: theme.textSecondary, marginBottom: 8 }]}>
              Enter the WebSocket address of your self-hosted server (e.g. wss://192.168.1.10)
            </Text>
            <View style={styles.secretCodeInputRow}>
              <TextInput
                style={[styles.secretCodeInput, { 
                  backgroundColor: theme.inputBackground, 
                  borderColor: theme.inputBorder, 
                  color: theme.text 
                }]}
                value={serverUrl}
                onChangeText={setServerUrl}
                placeholder="wss://your-server-ip:8080"
                placeholderTextColor={theme.inputPlaceholder}
                autoCapitalize="none"
                autoCorrect={false}
                keyboardType="url"
                accessible={true}
                accessibilityLabel="Server URL input"
                accessibilityHint="Enter the WebSocket URL of your self-hosted ProgramAT server"
              />
              <TouchableOpacity
                style={[
                  styles.secretCodeButton,
                  { backgroundColor: theme.primary },
                  !serverUrl.trim() && styles.secretCodeButtonDisabled
                ]}
                onPress={handleSaveServerUrl}
                disabled={!serverUrl.trim()}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Save server URL"
                accessibilityHint="Double tap to save and connect to this server">
                <Text style={styles.secretCodeButtonText}>Save</Text>
              </TouchableOpacity>
            </View>
            {serverUrl.trim() && (
              <TouchableOpacity
                style={[styles.resetServerButton, { backgroundColor: theme.error }]}
                onPress={handleClearServerUrl}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Clear server URL"
                accessibilityHint="Double tap to clear the saved server address and disconnect">
                <Text style={styles.resetServerButtonText}>Clear Server</Text>
              </TouchableOpacity>
            )}
          </View>

          <View style={[styles.serverInfo, { backgroundColor: theme.backgroundSecondary }]}>
            <Text style={[styles.serverInfoLabel, { color: theme.textSecondary }]} accessible={false}>Active URL:</Text>
            <Text 
              style={[styles.serverInfoValue, { color: theme.text }]}
              selectable={true}
              accessible={true}
              accessibilityRole="text"
              accessibilityLabel={`Active server URL: ${currentServerUrl || 'None'}`}
              accessibilityHint="Long press to copy URL">
              {currentServerUrl || 'Not configured'}
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
  modeButtonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 12,
    gap: 8,
  },
  modeButton: {
    flex: 1,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1.5,
    alignItems: 'center',
  },
  modeButtonText: {
    fontSize: 13,
    fontWeight: '600',
  },
});
