/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 */
//import packages
import React, { useEffect, useState, useCallback } from 'react';
import { StatusBar, StyleSheet, useColorScheme, View, Text, TouchableOpacity, LogBox } from 'react-native';
import {
  SafeAreaProvider,
  useSafeAreaInsets,
} from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';
import TabNavigator from './TabNavigator';
import WebSocketService from './WebSocketService';
import TextToSpeechService from './TextToSpeechService';
import BeepService from './BeepService';
import { SERVER_CONFIGS } from './config';
import { ThemeProvider, useTheme } from './ThemeContext';

// Storage key for custom server code (must match Settings.tsx)
const CUSTOM_SERVER_KEY = '@custom_server_code';

// Suppress Fast Refresh hook warnings that don't actually affect the app
LogBox.ignoreLogs([
  'Rendered fewer hooks than expected',
  'Rendered more hooks than expected',
]);
LogBox.ignoreAllLogs(false); // Set to true to hide all logs (not recommended)

function App() {
  return (
    <SafeAreaProvider>
      <ThemeProvider>
        <AppWrapper />
      </ThemeProvider>
    </SafeAreaProvider>
  );
}

function AppWrapper() {
  const { theme, themeMode } = useTheme();
  
  return (
    <>
      <StatusBar barStyle={themeMode === 'dark' ? 'light-content' : 'dark-content'} />
      <AppContent />
    </>
  );
}

interface Issue {
  number: number;
  title: string;
  labels: string[];
  created_at: string;
  updated_at: string;
}

interface Tool {
  name: string;
  path: string;
  description?: string;
  code?: string;
  language?: string;
  pr_number?: number;
  pr_title?: string;
  branch_name?: string;
}

function AppContent() {
  const { theme } = useTheme();
  const safeAreaInsets = useSafeAreaInsets();
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(true);
  const [connectionError, setConnectionError] = useState('');
  const [showIssueSelector, setShowIssueSelector] = useState(false);
  const [selectedPR, setSelectedPR] = useState<{number: number; title: string} | null>(null);
  const [prList, setPRList] = useState<any[]>([]);
  const [spokenFeedback, setSpokenFeedback] = useState<string>('');
  const [prTools, setPRTools] = useState<Tool[]>([]); // Tools for selected PR
  
  // Copilot session data
  const [copilotSessions, setCopilotSessions] = useState<any[]>([]);
  const [copilotSummaries, setCopilotSummaries] = useState<any[]>([]);
  const [copilotLogs, setCopilotLogs] = useState<any[]>([]);

  // Function to clear copilot data when switching PRs
  const clearCopilotData = useCallback(() => {
    console.log('[App] Clearing copilot data for new PR');
    setCopilotSessions([]);
    setCopilotSummaries([]);
    setCopilotLogs([]);
  }, []);

  // Load persisted selected PR on mount
  useEffect(() => {
    const loadSelectedPR = async () => {
      try {
        const savedPR = await AsyncStorage.getItem('selectedPR');
        if (savedPR) {
          const pr = JSON.parse(savedPR);
          console.log('[App] Loaded persisted PR:', pr.number, pr.title);
          setSelectedPR(pr);
          // Request tools for the persisted PR if we're connected
          if (WebSocketService.isConnected()) {
            WebSocketService.sendIssueSelection(pr.number);
          }
        }
      } catch (error) {
        console.error('[App] Error loading selected PR:', error);
      }
    };
    loadSelectedPR();
  }, []);

  useEffect(() => {
    console.log('[App] useEffect running - setting up WebSocket callbacks');
    // Set up WebSocket callbacks
    WebSocketService.onConnectionChange((connected) => {
      console.log('[App] onConnectionChange callback fired, connected:', connected);
      setIsConnected(connected);
      setIsConnecting(false); // Stop connecting state when connection status changes
      if (connected) {
        setConnectionError('');
      }
    });

    WebSocketService.onError((error) => {
      console.log('[App] onError callback fired:', error);
      setConnectionError(error);
    });

    // Listen for ALL message types from server (centralized handler)
    const handleMessage = (message: any) => {
      console.log('[App] Received message:', message);
      
      if (message.type === 'issue_list') {
        // Handle issue list from server (unused - we only use PRs now)
        console.log('[App] Ignoring issue list - using PRs only');
      } else if (message.type === 'pr_list') {
        // Handle PR list from server
        console.log('[App] Setting PR list:', message.prs?.length || 0, 'PRs');
        setPRList(message.prs || []);
      } else if (message.type === 'pr_tools') {
        // Handle tools from a specific PR
        console.log('[App] Received tools for PR #' + message.pr_number + ':', message.tools?.length || 0);
        setPRTools(message.tools || []); // Store in PR tools state
        
        // Set selected PR with title from backend
        setSelectedPR({
          number: message.pr_number,
          title: message.pr_title || `PR #${message.pr_number}`
        });
      } else if (message.type === 'production_tools') {
        // Handle tools from production mode (main branch)
        console.log('[App] Received production tools from main branch:', message.tools?.length || 0);
        setPRTools(message.tools || []); // Store production tools in same state
        
        // Clear selected PR in production mode
        setSelectedPR({
          number: 0,
          title: 'Production (main branch)'
        });
      } else if (message.type === 'issue_selected') {
        setSelectedPR({
          number: message.issue_number,
          title: message.issue_title
        });
      } else if (message.type === 'mode_switched') {
        // Handle mode switch with tools
        if (message.mode === 'update') {
          setSelectedPR({
            number: message.issue_number,
            title: message.issue_title
          });
          // Update tools if provided
          if (message.tools) {
            console.log('[App] Received tools for PR:', message.tools.length);
            setPRTools(message.tools); // Store in PR tools state
          }
        } else if (message.mode === 'create') {
          setSelectedPR(null);
          setPRTools([]); // Clear PR tools
        }
      } else if (message.type === 'issue_created') {
        // Clear selection after successful create
        setSelectedPR(null);
        setPRTools([]);
        const feedbackMsg = 'New issue created successfully';
        TextToSpeechService.speakWithInterrupt(feedbackMsg);
        setSpokenFeedback(feedbackMsg);
      } else if (message.type === 'issue_updated') {
        // Provide feedback for successful update
        const feedbackMsg = 'Update sent to issue';
        TextToSpeechService.speakWithInterrupt(feedbackMsg);
        setSpokenFeedback(feedbackMsg);
        // Note: We keep the selected issue so user can send more updates
      } else if (message.type === 'feedback') {
        // Handle feedback for missing fields in create mode
        console.log('[App] Received feedback:', message.message);
        if (message.message) {
          TextToSpeechService.speakWithInterrupt(message.message);
          setSpokenFeedback(message.message);
        }
      } else if (message.type === 'pr_sessions_list') {
        // Handle Copilot sessions for a PR
        console.log('[App] Received sessions for PR #' + message.pr_number + ':', message.sessions?.length || 0);
        setCopilotSessions(message.sessions || []);
      } else if (message.type === 'session_summaries') {
        // Handle summaries for a session (batch fetch)
        console.log('[App] Received summaries for session:', message.summaries?.length || 0);
        setCopilotSummaries(message.summaries || []);
      } else if (message.type === 'copilot_summary') {
        // Handle live summary as it arrives
        console.log('[App] Received live summary:', message.summary);
        const newSummary = {
          id: Date.now(), // Temporary ID until we refetch
          session_id: message.session_id,
          summary_text: message.summary,
          start_entry_num: message.start_entry,
          end_entry_num: message.end_entry,
          timestamp: message.timestamp
        };
        setCopilotSummaries(prev => [...prev, newSummary]);
      } else if (message.type === 'session_logs') {
        // Handle logs for a session
        console.log('[App] Received logs for session:', message.logs?.length || 0);
        setCopilotLogs(message.logs || []);
      }
    };

    WebSocketService.onMessage(handleMessage);

    // Auto-connect on mount
    console.log('[App] Calling connectToServer on mount');
    connectToServer();

    return () => {
      console.log('[App] useEffect cleanup - disconnecting WebSocket');
      // Disconnect when unmounting
      WebSocketService.disconnect();
    };
  }, []);

  // Loading sound effect for server connection
  useEffect(() => {
    let beepTimer: ReturnType<typeof setTimeout> | null = null;
    
    if (isConnecting) {
      console.log('[App] Connection starting, will beep after 3 seconds if still connecting');
      // Wait 3 seconds before starting beep
      beepTimer = setTimeout(() => {
        console.log('[App] 3 seconds elapsed, starting loading sound');
        BeepService.startLoadingSound();
      }, 3000);
    } else {
      console.log('[App] Stopping connection loading sound');
      BeepService.stopLoadingSound();
    }

    // Cleanup on unmount or when isConnecting changes
    return () => {
      if (beepTimer) {
        clearTimeout(beepTimer);
      }
      BeepService.stopLoadingSound();
    };
  }, [isConnecting]);

  const connectToServer = async () => {
    console.log('[App] connectToServer called');
    try {
      setIsConnecting(true);
      setConnectionError('');
      
      // Load saved server code BEFORE connecting
      try {
        const savedCode = await AsyncStorage.getItem(CUSTOM_SERVER_KEY);
        console.log('[App] AsyncStorage returned savedCode:', savedCode);
        console.log('[App] SERVER_CONFIGS keys:', Object.keys(SERVER_CONFIGS));
        console.log('[App] Is savedCode in SERVER_CONFIGS?', savedCode ? !!SERVER_CONFIGS[savedCode] : 'no savedCode');
        
        if (savedCode && SERVER_CONFIGS[savedCode]) {
          console.log('[App] Found saved server code:', savedCode);
          console.log('[App] Server config:', JSON.stringify(SERVER_CONFIGS[savedCode]));
          console.log('[App] Setting server URL to:', SERVER_CONFIGS[savedCode].url);
          WebSocketService.setServerUrl(SERVER_CONFIGS[savedCode].url, false);
        } else {
          console.log('[App] No valid saved server code, using default');
          console.log('[App] savedCode was:', savedCode);
        }
      } catch (storageError) {
        console.error('[App] Error loading saved server code:', storageError);
      }
      
      console.log('[App] Calling WebSocketService.connect()');
      await WebSocketService.connect();
      console.log('[App] WebSocketService.connect() completed successfully');
    } catch (error) {
      console.error('[App] Failed to connect:', error);
      // Display error to user
      if (error instanceof Error) {
        setConnectionError(error.message);
      } else {
        setConnectionError('Failed to connect to server');
      }
    } finally {
      console.log('[App] connectToServer finally block - setting isConnecting to false');
      setIsConnecting(false);
    }
  };

  const disconnectFromServer = () => {
    WebSocketService.disconnect();
  };

  const handleIssueSelect = (pr: {number: number; title: string}) => {
    setSelectedPR(pr);
    // Persist the selected PR
    AsyncStorage.setItem('selectedPR', JSON.stringify(pr)).catch(error => {
      console.error('[App] Error saving selected PR:', error);
    });
    // Note: PR selection is now sent via WebSocketService.sendIssueSelection() 
    // from IssueSelector component, not as text
  };

  const handleNewIssue = () => {
    setSelectedPR(null);
    // Clear persisted PR
    AsyncStorage.removeItem('selectedPR').catch(error => {
      console.error('[App] Error removing selected PR:', error);
    });
    // Send mode switch to backend
    WebSocketService.sendIssueSelection('create');
  };

  return (
    <View 
      style={[styles.container, { paddingTop: safeAreaInsets.top, backgroundColor: theme.background }]}
      accessible={false}>
      
      {/* PR Mode Indicator */}
      {selectedPR && (
        <View 
          style={[styles.issueModeBar, { backgroundColor: theme.primaryDark + '20', borderBottomColor: theme.primary }]}
          accessible={false}>
          <Text 
            style={[styles.issueModeText, { color: theme.primary }]}
            accessible={true}
            accessibilityRole="text"
            accessibilityLabel={selectedPR.number === 0 
              ? 'Current mode: Running from main branch' 
              : `Current mode: Updating PR ${selectedPR.number}: ${selectedPR.title}`}
            accessibilityLiveRegion="polite">
            {selectedPR.number === 0 
              ? 'Mode: Running from main' 
              : `Mode: Updating #${selectedPR.number} - ${selectedPR.title}`}
          </Text>
          <TouchableOpacity
            style={[styles.newIssueButton, { backgroundColor: theme.success }]}
            onPress={handleNewIssue}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Switch to create new issue mode"
            accessibilityHint="Double tap to stop updating the current issue and switch to creating a new issue">
            <Text 
              style={styles.newIssueButtonText}
              accessible={false}
              importantForAccessibility="no-hide-descendants">
              New Issue
            </Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Tab Navigator */}
      <TabNavigator 
        serverFeedback={spokenFeedback}
        selectedIssue={selectedPR}
        onIssueSelect={handleIssueSelect}
        onNewIssue={handleNewIssue}
        prList={prList}
        issueTools={prTools}
        copilotSessions={copilotSessions}
        copilotSummaries={copilotSummaries}
        copilotLogs={copilotLogs}
        onClearCopilotData={clearCopilotData}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  issueModeBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderBottomWidth: 1,
  },
  issueModeText: {
    fontSize: 13,
    fontWeight: '500',
    flex: 1,
    marginRight: 8,
  },
  newIssueButton: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 4,
  },
  newIssueButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
});

export default App;
