/**
 * PRs and Text Combined Component
 * Shows PR list by default, with navigation to text input for selected PRs
 */

import React, { useState, useEffect } from 'react';
import { View, StyleSheet, Linking, Alert } from 'react-native';
import { useTheme } from './ThemeContext';
import IssueSelector from './IssueSelector';
import TextInput from './TextInput';
import Config, { AppMode } from './config';
import ReviewPane from './ReviewPane';

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
  appMode?: AppMode;
}

type ViewMode = 'text-input' | 'pr-list' | 'view-logs' | 'review';

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
  onClearCopilotData = () => {},
  appMode = 'development'
}: PRsAndTextProps) {
  const { theme } = useTheme();
  // Start with PR list view
  const [viewMode, setViewMode] = useState<ViewMode>('pr-list');
  const [viewLogsPR, setViewLogsPR] = useState<{number: number; title: string; body?: string} | null>(null);
  

  // Reset to PR list view whenever the tab becomes active
  useEffect(() => {
    if (isActive) {
      setViewMode('pr-list');
    }
  }, [isActive]);

  const handleIssueSelect = (issue: {number: number; title: string}) => {
    onIssueSelect(issue);
    if (appMode !== 'review') {
      setViewMode('text-input'); // Navigate to text input after selecting PR
    }
  };

  const handleCreateNew = () => {
    onNewIssue();
    setViewMode('text-input'); // Navigate to text input for new issue
  };

  const handleReviewPR = (pr: {number: number; title: string; body?: string}) => {
    setViewLogsPR(pr); // reuse this state to track the target PR
    setViewMode('review');
  };

  const handleViewPRs = () => {
    setViewMode('pr-list'); // Navigate to PR list
  };

  const handleBackToPRs = () => {
    setViewMode('pr-list'); // Go back to PR list
  };

  const handleViewLogs = (pr: {number: number; title: string}) => {
    const url = `https://github.com/${Config.REVIEW_GITHUB_REPO}/pull/${pr.number}`;
    Linking.openURL(url).catch(() =>
      Alert.alert('Could not open GitHub', `Open this URL manually:\n${url}`)
    );
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
          onReviewPR={handleReviewPR}
          prs={prList}
          embedded={true}
          selectedIssue={selectedIssue}
          reviewMode={appMode === 'review'}
        />
      ) : viewMode === 'review' && viewLogsPR ? (
        <ReviewPane
          prNumber={viewLogsPR.number}
          prTitle={viewLogsPR.title}
          prBody={viewLogsPR.body}
          onBack={handleBackToPRs}
        />
      ) : appMode !== 'review' ? (
        <TextInput 
          serverFeedback={serverFeedback}
          selectedIssue={selectedIssue}
          onNewIssue={onNewIssue}
          onBack={handleBackToPRs}
          showBackButton={true}
        />
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});
