/**
 * PRs and Text Combined Component
 * Shows PR list by default, with navigation to text input for selected PRs
 */

import React, { useState, useEffect, useRef } from 'react';
import { View, StyleSheet } from 'react-native';
import { useTheme } from './ThemeContext';
import IssueSelector from './IssueSelector';
import TextInput from './TextInput';
import ViewLogsScreen from './ViewLogsScreen';
import BeepService from './BeepService';

interface PRsAndTextProps {
  serverFeedback?: string;
  selectedIssue: {number: number; title: string} | null;
  onIssueSelect: (issue: {number: number; title: string}) => void;
  onNewIssue: () => void;
  prList?: any[];
  isActive?: boolean; // Whether this tab is currently active
  onNavigateToTools?: () => void; // Callback to navigate to Tools tab
  copilotSessions?: any[];
  copilotSummaries?: any[];
  copilotLogs?: any[];
  onClearCopilotData?: () => void; // Callback to clear copilot data when switching PRs
}

type ViewMode = 'text-input' | 'pr-list' | 'view-logs';

export default function PRsAndText({ 
  serverFeedback, 
  selectedIssue, 
  onIssueSelect, 
  onNewIssue,
  prList = [],
  isActive = true,
  onNavigateToTools,
  copilotSessions = [],
  copilotSummaries = [],
  copilotLogs = [],
  onClearCopilotData = () => {}
}: PRsAndTextProps) {
  const { theme } = useTheme();
  // Start with PR list view
  const [viewMode, setViewMode] = useState<ViewMode>('pr-list');
  const [viewLogsPR, setViewLogsPR] = useState<{number: number; title: string} | null>(null);
  
  // Track the last PR we viewed logs for
  const lastViewedPRRef = useRef<number | null>(null);

  // Reset to PR list view whenever the tab becomes active
  useEffect(() => {
    if (isActive) {
      setViewMode('pr-list');
    }
  }, [isActive]);

  const handleIssueSelect = (issue: {number: number; title: string}) => {
    onIssueSelect(issue);
    setViewMode('text-input'); // Navigate to text input after selecting PR
  };

  const handleCreateNew = () => {
    onNewIssue();
    setViewMode('text-input'); // Navigate to text input for new issue
  };

  const handleViewPRs = () => {
    setViewMode('pr-list'); // Navigate to PR list
  };

  const handleBackToPRs = () => {
    setViewMode('pr-list'); // Go back to PR list
  };

  const handleViewLogs = (pr: {number: number; title: string}) => {
    // If viewing a different PR than last time, clear old data and play loading beep
    if (lastViewedPRRef.current !== pr.number) {
      console.log(`[PRsAndText] Viewing logs for new PR #${pr.number}, clearing old data`);
      onClearCopilotData();
      BeepService.playLoadingSound();
      lastViewedPRRef.current = pr.number;
    }
    setViewLogsPR(pr);
    setViewMode('view-logs');
  };

  return (
    <View style={[styles.container, { backgroundColor: theme.background }]}>
      {viewMode === 'pr-list' ? (
        <IssueSelector 
          visible={true}
          onClose={() => {}} // Empty function - we're embedded, don't need to close
          onIssueSelect={handleIssueSelect}
          onCreateNew={handleCreateNew}
          onNavigateToTools={onNavigateToTools}
          onViewLogs={handleViewLogs}
          prs={prList}
          embedded={true}
          selectedIssue={selectedIssue}
        />
      ) : viewMode === 'view-logs' && viewLogsPR ? (
        <ViewLogsScreen
          prNumber={viewLogsPR.number}
          prTitle={viewLogsPR.title}
          onBack={handleBackToPRs}
          sessions={copilotSessions}
          summaries={copilotSummaries}
          logs={copilotLogs}
          onClearData={onClearCopilotData}
        />
      ) : (
        <TextInput 
          serverFeedback={serverFeedback}
          selectedIssue={selectedIssue}
          onNewIssue={onNewIssue}
          onBack={handleBackToPRs}
          showBackButton={true}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});
