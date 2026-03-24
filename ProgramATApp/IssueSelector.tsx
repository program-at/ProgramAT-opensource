/**
 * IssueSelector Component
 * Displays a list of GitHub issues or PRs and allows selection for iteration
 *
 * @format
 */

import React, { useEffect, useState } from 'react';
import {
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ScrollView,
  Modal,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTheme } from './ThemeContext';
import WebSocketService from './WebSocketService';
import BeepService from './BeepService';

interface Issue {
  number: number;
  title: string;
  labels: string[];
  created_at: string;
  updated_at: string;
}

interface PR {
  number: number;
  title: string;
  branch: string;
  state: string;
  mentioned_issues: string[];
  created_at: string;
  updated_at: string;
}

interface IssueSelectorProps {
  visible: boolean;
  onClose: () => void;
  onIssueSelect: (issue: Issue) => void;
  onPRSelect?: (pr: PR) => void;
  onCreateNew?: () => void; // New prop for switching to create mode
  onNavigateToTools?: () => void; // Callback to navigate to Tools tab
  onViewLogs?: (pr: {number: number; title: string}) => void; // Callback to view logs for a PR
  issues?: Issue[]; // Optional - not used in PR-only mode
  prs?: PR[];
  embedded?: boolean;
  selectedIssue?: {number: number; title: string} | null; // Currently selected PR
}

export default function IssueSelector({ 
  visible, 
  onClose, 
  onIssueSelect, 
  onPRSelect,
  onCreateNew,
  onNavigateToTools,
  onViewLogs,
  issues = [], // Default to empty array
  prs = [],
  embedded = false,
  selectedIssue = null
}: IssueSelectorProps) {
  const { theme } = useTheme();
  const [loading, setLoading] = useState(false);
  const [expectingNewPRs, setExpectingNewPRs] = useState(false); // Track when we're expecting fresh PRs
  const [error, setError] = useState<string>('');

  useEffect(() => {
    if (visible) {
      // Small delay to allow WebSocket to connect on app startup
      const timer = setTimeout(() => {
        requestPRList();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [visible]);

  const requestIssueList = () => {
    console.log('[IssueSelector] Requesting issue list...');
    setLoading(true);
    setExpectingNewPRs(true);
    setError('');
    
    const success = WebSocketService.requestIssueList();
    if (!success) {
      setLoading(false);
      setExpectingNewPRs(false);
      setError('Not connected to server. Please connect first.');
    }
    
    // Safety timeout in case issues never arrive
    setTimeout(() => {
      if (expectingNewPRs) {
        console.warn('[IssueSelector] Timeout - no issues received');
        setLoading(false);
        setExpectingNewPRs(false);
      }
    }, 10000);
  };

  const requestPRList = () => {
    console.log('[IssueSelector] Requesting PR list...');
    setLoading(true);
    setExpectingNewPRs(true);
    setError('');
    
    // Clear existing PRs so we know when fresh ones arrive  
    // Note: This will be handled by the parent component that manages the prs prop
    // But we mark that we're expecting new ones
    
    const success = WebSocketService.requestPRList();
    if (!success) {
      setLoading(false);
      setExpectingNewPRs(false);
      // Auto-retry after a short delay (connection might still be establishing)
      console.log('[IssueSelector] Not connected, will retry in 1 second...');
      setTimeout(() => {
        if (WebSocketService.isConnected()) {
          requestPRList();
        } else {
          setError('Not connected to server. Please connect first.');
        }
      }, 1000);
      return;
    }
    
    // Safety timeout in case PRs never arrive
    setTimeout(() => {
      if (expectingNewPRs) {
        console.warn('[IssueSelector] Timeout - no PRs received');
        setLoading(false);
        setExpectingNewPRs(false);
      }
    }, 10000);
  };

  // Stop loading when PRs arrive (only after we've requested them)
  useEffect(() => {
    if (prs.length > 0 && expectingNewPRs) {
      console.log('[IssueSelector] Fresh PRs arrived (', prs.length, 'PRs), stopping loading');
      setLoading(false);
      setExpectingNewPRs(false);
      setError('');
      
      // Play beep when PRs arrive
      console.log('[IssueSelector] Playing beep for PR arrival...');
      BeepService.playLoadingSound().catch((error: any) => {
        console.error('[IssueSelector] Beep failed:', error);
      });
    }
  }, [prs, expectingNewPRs]);

  const handleIssuePress = (issue: Issue) => {
    // Send mode selection to backend
    WebSocketService.sendIssueSelection('update', issue.number, issue.title);
    
    // Call the parent callback
    onIssueSelect(issue);
    onClose();
  };

  const handlePRPress = (pr: PR) => {
    console.log('[IssueSelector] PR selected:', pr.number, pr.title);
    
    // Show native alert dialog for better accessibility
    Alert.alert(
      `PR #${pr.number}: ${pr.title}\n\nWhat would you like to do?`,
      '',
      [
        {
          text: 'Open Tools',
          onPress: () => {
            console.log('[IssueSelector] Opening tools for PR:', pr.number);
            
            // Send mode selection to backend and request tools
            WebSocketService.sendIssueSelection('update', pr.number, pr.title);
            WebSocketService.requestPRTools(pr.number);
            
            // Convert PR to Issue format
            const prAsIssue: Issue = {
              number: pr.number,
              title: pr.title,
              labels: [],
              created_at: pr.created_at,
              updated_at: pr.updated_at
            };
            
            // Select the PR
            onIssueSelect(prAsIssue);
            
            // Close selector
            onClose();
            
            // Navigate to Tools tab
            if (onNavigateToTools) {
              onNavigateToTools();
            }
          }
        },
        {
          text: 'Update Issue',
          onPress: () => {
            console.log('[IssueSelector] Opening update mode for PR:', pr.number);
            
            // Send mode selection to backend
            WebSocketService.sendIssueSelection('update', pr.number, pr.title);
            WebSocketService.requestPRTools(pr.number);
            
            // Convert PR to Issue format
            const prAsIssue: Issue = {
              number: pr.number,
              title: pr.title,
              labels: [],
              created_at: pr.created_at,
              updated_at: pr.updated_at
            };
            
            // Call the parent callback
            onIssueSelect(prAsIssue);
            
            // Close selector
            onClose();
          }
        },
        {
          text: 'View Logs',
          onPress: () => {
            console.log('[IssueSelector] Viewing logs for PR:', pr.number);
            
            // Call parent callback to show ViewLogsScreen
            if (onViewLogs) {
              onViewLogs({ number: pr.number, title: pr.title });
            }
          }
        },
        {
          text: 'Cancel',
          style: 'cancel'
        }
      ]
    );
  };

  const renderContent = () => (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.background }]} edges={embedded ? [] : ['top', 'bottom']}>
      <View style={[styles.header, { backgroundColor: theme.background, borderBottomColor: theme.border }]}>
        <Text 
          style={[styles.headerText, { color: theme.text }]}
          accessible={true}
          accessibilityRole="header"
          accessibilityLabel="Select a pull request to update">
          Select Pull Request
        </Text>
        {!embedded && (
          <TouchableOpacity
            style={styles.closeButton}
            onPress={onClose}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Close selector"
            accessibilityHint="Double tap to close and return to the main screen"
            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}>
            <Text 
              style={[styles.closeButtonText, { color: theme.text }]}
              accessible={false}
              importantForAccessibility="no-hide-descendants">
              ✕
            </Text>
          </TouchableOpacity>
        )}
      </View>

        {loading ? (
          <View 
            style={styles.loadingContainer}
            accessible={true}
            accessibilityRole="progressbar"
            accessibilityLabel="Loading issues"
            accessibilityLiveRegion="polite">
            <Text 
              style={[styles.loadingText, { color: theme.textSecondary }]}
              accessible={false}
              importantForAccessibility="no-hide-descendants">
              Loading issues...
            </Text>
          </View>
        ) : error !== '' ? (
          <View 
            style={styles.emptyContainer}
            accessible={false}>
            <Text 
              style={[styles.errorText, { color: theme.error }]}
              accessible={true}
              accessibilityRole="alert"
              accessibilityLabel={`Error loading: ${error}`}>
              {error}
            </Text>
            <TouchableOpacity
              style={[styles.retryButton, { backgroundColor: theme.primary }]}
              onPress={requestPRList}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Retry loading"
              accessibilityHint="Double tap to try loading again">
              <Text 
                style={styles.retryButtonText}
                accessible={false}
                importantForAccessibility="no-hide-descendants">
                Retry
              </Text>
            </TouchableOpacity>
          </View>
        ) : prs.length === 0 ? (
          <View 
            style={styles.emptyContainer}
            accessible={false}>
            <Text 
              style={[styles.emptyText, { color: theme.textTertiary }]}
              accessible={true}
              accessibilityRole="text"
              accessibilityLabel="No open pull requests found">
              No open pull requests found
            </Text>
            <TouchableOpacity
              style={[styles.retryButton, { backgroundColor: theme.primary }]}
              onPress={requestPRList}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Retry loading pull requests"
              accessibilityHint="Double tap to try loading again">
              <Text 
                style={styles.retryButtonText}
                accessible={false}
                importantForAccessibility="no-hide-descendants">
                Retry
              </Text>
            </TouchableOpacity>
          </View>
        ) : (
          <ScrollView 
            style={styles.issueList}
            accessible={false}
            accessibilityLabel={`List of ${prs.length} open pull requests`}>
            {prs.map((pr) => {
              const isSelected = selectedIssue?.number === pr.number;
              return (
              <TouchableOpacity
                key={pr.number}
                style={[
                  styles.issueCard, 
                  { backgroundColor: theme.card, borderColor: theme.border },
                  isSelected && { backgroundColor: theme.backgroundSecondary, borderColor: theme.primary }
                ]}
                onPress={() => handlePRPress(pr)}
                accessible={true}
                accessibilityRole="button"
                accessibilityState={{ selected: isSelected }}
                accessibilityLabel={`Pull request number ${pr.number}: ${pr.title}. Branch: ${pr.branch}${pr.mentioned_issues && pr.mentioned_issues.length > 0 ? `. Addresses issues: ${pr.mentioned_issues.join(', ')}` : ''}. Last updated ${new Date(pr.updated_at).toLocaleDateString()}`}
                accessibilityHint="Double tap to select this pull request">
                <View 
                  style={styles.issueHeader}
                  accessible={false}>
                  <Text 
                    style={[
                      styles.issueNumber, 
                      { color: theme.primary },
                      isSelected && { color: theme.primary }
                    ]}
                    accessible={false}
                    importantForAccessibility="no-hide-descendants">
                    PR #{pr.number}
                  </Text>
                </View>
                <Text 
                  style={[
                    styles.issueTitle, 
                    { color: theme.text },
                    isSelected && { color: theme.text }
                  ]} 
                  numberOfLines={2}
                  accessible={false}
                  importantForAccessibility="no-hide-descendants">
                  {pr.title}
                </Text>
                <Text 
                  style={[styles.branchName, { color: theme.textTertiary }]}
                  numberOfLines={1}
                  accessible={false}
                  importantForAccessibility="no-hide-descendants">
                  Branch: {pr.branch}
                </Text>
                {pr.mentioned_issues && pr.mentioned_issues.length > 0 && (
                  <Text 
                    style={[styles.mentionedIssues, { color: theme.textTertiary }]}
                    accessible={false}
                    importantForAccessibility="no-hide-descendants">
                    Addresses: {pr.mentioned_issues.map(i => `#${i}`).join(', ')}
                  </Text>
                )}
                <Text 
                  style={[styles.issueDate, { color: theme.textTertiary }]}
                  accessible={false}
                  importantForAccessibility="no-hide-descendants">
                  Updated: {new Date(pr.updated_at).toLocaleDateString()}
                </Text>
              </TouchableOpacity>
              );
            })}
          </ScrollView>
        )}

        <View 
          style={[styles.footer, { backgroundColor: theme.background, borderTopColor: theme.border }]}
          accessible={false}>
          <TouchableOpacity
            style={[styles.createNewButton, { backgroundColor: theme.success }]}
            onPress={() => {
              if (onCreateNew) {
                onCreateNew(); // Switch to create mode
              }
              onClose(); // Close the selector
            }}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Create new issue instead"
            accessibilityHint="Double tap to close this screen and return to creating a new issue">
            <Text 
              style={styles.createNewButtonText}
              accessible={false}
              importantForAccessibility="no-hide-descendants">
              Create New Issue Instead
            </Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );

  // If embedded mode, render directly without Modal
  if (embedded) {
    return renderContent();
  }

  // Otherwise, wrap in Modal
  return (
    <Modal
      visible={visible}
      animationType="slide"
      transparent={false}
      onRequestClose={onClose}>
      {renderContent()}
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    minHeight: 60,
  },
  headerText: {
    fontSize: 18,
    fontWeight: 'bold',
    flex: 1,
  },
  closeButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 12,
  },
  closeButtonText: {
    fontSize: 24,
    fontWeight: 'bold',
    lineHeight: 24,
  },
  modeToggle: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 8,
    gap: 8,
    borderBottomWidth: 1,
  },
  modeButton: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  modeButtonActive: {
  },
  modeButtonText: {
    fontSize: 14,
    fontWeight: '600',
  },
  modeButtonTextActive: {
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 16,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  emptyText: {
    fontSize: 16,
    marginBottom: 20,
    textAlign: 'center',
  },
  errorText: {
    fontSize: 16,
    marginBottom: 20,
    textAlign: 'center',
  },
  retryButton: {
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  retryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  issueList: {
    flex: 1,
    padding: 12,
  },
  issueCard: {
    borderRadius: 8,
    padding: 16,
    marginBottom: 12,
    borderWidth: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  issueHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  issueNumber: {
    fontSize: 14,
    fontWeight: 'bold',
    marginRight: 8,
  },
  labelsContainer: {
    flexDirection: 'row',
    gap: 6,
  },
  label: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  bugLabel: {
  },
  enhancementLabel: {
  },
  labelText: {
    fontSize: 11,
    fontWeight: '600',
  },
  issueTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
  },
  branchName: {
    fontSize: 13,
    fontFamily: 'monospace',
    marginBottom: 4,
  },
  mentionedIssues: {
    fontSize: 12,
    marginBottom: 4,
  },
  issueDate: {
    fontSize: 12,
  },
  footer: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderTopWidth: 1,
  },
  createNewButton: {
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  createNewButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  issueCardSelected: {
  },
  issueNumberSelected: {
  },
  issueTitleSelected: {
  },
});
