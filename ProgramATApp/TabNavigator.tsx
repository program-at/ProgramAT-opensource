/**
 * Tab Navigator Component
 * Bottom tab navigation for PRs/Text, Tools/Runner, Chat, and Settings
 */

import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Config, { AppMode } from './config';
import PRsAndText from './PRsAndText';
import ToolsAndRunner from './ToolsAndRunner';
import Settings from './Settings';
import Chat from './Chat';
import WebSocketService from './WebSocketService';
import { useTheme } from './ThemeContext';

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

interface TabNavigatorProps {
  serverFeedback?: string;
  selectedIssue: {number: number; title: string} | null;
  onIssueSelect: (issue: {number: number; title: string}) => void;
  onNewIssue: () => void;
  prList?: any[]; // PR list
  issueTools?: Tool[];
  copilotSessions?: any[];
  copilotSummaries?: any[];
  copilotLogs?: any[];
  onClearCopilotData?: () => void;
}

type TabName = 'prs' | 'tools' | 'settings' | 'chat';

export default function TabNavigator({ 
  serverFeedback, 
  selectedIssue, 
  onIssueSelect, 
  onNewIssue,
  prList = [], // PR list with default
  issueTools = [],
  copilotSessions = [],
  copilotSummaries = [],
  copilotLogs = [],
  onClearCopilotData
}: TabNavigatorProps) {
  const { theme } = useTheme();
  const [appMode, setAppMode] = useState<AppMode>(Config.APP_MODE);
  // Start on tools tab in production mode, prs tab in development
  const [activeTab, setActiveTab] = useState<TabName>(appMode === 'production' ? 'tools' : 'prs');
  const [pendingConversationId, setPendingConversationId] = useState<string | null>(null);
  const insets = useSafeAreaInsets();

  const handleModeChange = (newMode: AppMode) => {
    setAppMode(newMode);
    Config.APP_MODE = newMode;
    
    // If switching to production and on prs tab, go to tools
    if (newMode === 'production' && activeTab === 'prs') {
      setActiveTab('tools');
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'prs':
        // Only show in dev mode, fallback to tools in production
        if (appMode !== 'development') {
          return (
            <ToolsAndRunner 
              issueTools={issueTools}
              productionMode={appMode === 'production'}
              isActive={true} // In production, we're in the fallback view
              selectedIssue={selectedIssue}
            />
          );
        }
        return (
          <PRsAndText
            serverFeedback={serverFeedback}
            selectedIssue={selectedIssue}
            onIssueSelect={onIssueSelect}
            onNewIssue={onNewIssue}
            prList={prList}
            isActive={activeTab === 'prs'}
            onNavigateToTools={() => setActiveTab('tools')}
            copilotSessions={copilotSessions}
            copilotSummaries={copilotSummaries}
            copilotLogs={copilotLogs}
            onClearCopilotData={onClearCopilotData}
          />
        );
      case 'tools':
        return (
          <ToolsAndRunner 
            issueTools={issueTools}
            productionMode={appMode === 'production'}
            isActive={activeTab === 'tools'}
            selectedIssue={selectedIssue}
            onNavigateToChat={(conversationId) => {
              setPendingConversationId(conversationId || null);
              setActiveTab('chat');
            }}
          />
        );
      case 'settings':
        return <Settings appMode={appMode} onModeChange={handleModeChange} />;
      case 'chat':
        return <Chat webSocketService={WebSocketService} initialConversationId={pendingConversationId || undefined} />;
      default:
        return appMode === 'development' 
          ? <PRsAndText 
              serverFeedback={serverFeedback} 
              selectedIssue={selectedIssue} 
              onIssueSelect={onIssueSelect} 
              onNewIssue={onNewIssue} 
              prList={prList} 
              isActive={true}
              copilotSessions={copilotSessions}
              copilotSummaries={copilotSummaries}
              copilotLogs={copilotLogs}
              onClearCopilotData={onClearCopilotData}
            />
          : <ToolsAndRunner issueTools={issueTools} productionMode={true} isActive={true} selectedIssue={selectedIssue} onNavigateToChat={(conversationId) => {
              setPendingConversationId(conversationId || null);
              setActiveTab('chat');
            }} />;
    }
  };

  return (
    <View style={styles.container}>
      {/* Content Area */}
      <View style={styles.content}>
        {renderContent()}
      </View>

      {/* Bottom Tab Bar */}
      <View 
        style={[
          styles.tabBar, 
          { 
            backgroundColor: theme.tabBarBackground, 
            borderTopColor: theme.border 
          }
        ]} 
        accessibilityRole="tablist">
        {/* PRs tab - only show in development mode */}
        {appMode === 'development' && (
          <TouchableOpacity
            style={[styles.tab, activeTab === 'prs' && styles.activeTab]}
            onPress={() => setActiveTab('prs')}
            accessible={true}
            accessibilityRole="tab"
            accessibilityLabel="Pull requests and text input tab"
            accessibilityState={{ selected: activeTab === 'prs' }}>
            <Text style={[styles.tabIcon, { color: activeTab === 'prs' ? theme.tabBarActive : theme.tabBarInactive }, activeTab === 'prs' && styles.activeTabIcon]} importantForAccessibility="no">
              📋
            </Text>
            <Text style={[styles.tabLabel, { color: activeTab === 'prs' ? theme.tabBarActive : theme.tabBarInactive }, activeTab === 'prs' && styles.activeTabLabel]} importantForAccessibility="no">
              PRs
            </Text>
          </TouchableOpacity>
        )}

        <TouchableOpacity
          style={[styles.tab, activeTab === 'tools' && styles.activeTab]}
          onPress={() => setActiveTab('tools')}
          accessible={true}
          accessibilityRole="tab"
          accessibilityLabel="Tools tab"
          accessibilityState={{ selected: activeTab === 'tools' }}>
          <Text style={[styles.tabIcon, { color: activeTab === 'tools' ? theme.tabBarActive : theme.tabBarInactive }, activeTab === 'tools' && styles.activeTabIcon]} importantForAccessibility="no">
            🛠️
          </Text>
          <Text style={[styles.tabLabel, { color: activeTab === 'tools' ? theme.tabBarActive : theme.tabBarInactive }, activeTab === 'tools' && styles.activeTabLabel]} importantForAccessibility="no">
            Tools
          </Text>
        </TouchableOpacity>

        {/* Chat Tab */}
        <TouchableOpacity
          style={[styles.tab, activeTab === 'chat' && styles.activeTab]}
          onPress={() => setActiveTab('chat')}
          accessible={true}
          accessibilityRole="tab"
          accessibilityLabel="Chat tab"
          accessibilityState={{ selected: activeTab === 'chat' }}>
          <Text style={[styles.tabIcon, { color: activeTab === 'chat' ? theme.tabBarActive : theme.tabBarInactive }, activeTab === 'chat' && styles.activeTabIcon]} importantForAccessibility="no">
            💬
          </Text>
          <Text style={[styles.tabLabel, { color: activeTab === 'chat' ? theme.tabBarActive : theme.tabBarInactive }, activeTab === 'chat' && styles.activeTabLabel]} importantForAccessibility="no">
            Chat
          </Text>
        </TouchableOpacity>

        {/* Settings Tab */}
        <TouchableOpacity
          style={[styles.tab, activeTab === 'settings' && styles.activeTab]}
          onPress={() => setActiveTab('settings')}
          accessible={true}
          accessibilityRole="tab"
          accessibilityLabel="Settings tab"
          accessibilityState={{ selected: activeTab === 'settings' }}>
          <Text style={[styles.tabIcon, { color: activeTab === 'settings' ? theme.tabBarActive : theme.tabBarInactive }, activeTab === 'settings' && styles.activeTabIcon]} importantForAccessibility="no">
            ⚙️
          </Text>
         <Text style={[styles.tabLabel, { color: activeTab === 'settings' ? theme.tabBarActive : theme.tabBarInactive }, activeTab === 'settings' && styles.activeTabLabel]} importantForAccessibility="no">
            Settings
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    //backgroundColor: '#fff',
  },
  content: {
    flex: 1,
  },
  tabBar: {
    flexDirection: 'row',
    //backgroundColor: '#fff',
    borderTopWidth: 1,
    //borderTopColor: '#e0e0e0',
    paddingBottom: 8,
    paddingTop: 8,
    elevation: 8,
    //shadowColor: '#000',
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    alignItems: 'center',
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    position: 'relative',
  },
  activeTab: {
    borderTopWidth: 2,
  },
  tabIcon: {
    fontSize: 24,
    marginBottom: 4,
    opacity: 0.8,
  },
  activeTabIcon: {
    opacity: 1,
  },
  tabLabel: {
    fontSize: 12,
    //color: '#666',
    fontWeight: '500',
  },
  activeTabLabel: {
    //color: '#2196F3',
    fontWeight: '600',
  },
});
