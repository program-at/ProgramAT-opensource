/**
 * ViewLogsScreen - View Copilot session logs and summaries for a PR
 * @format
 */

import React, {useState, useEffect, useCallback, useRef} from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  ActivityIndicator,
  ScrollView,
  AccessibilityInfo,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import WebSocketService from './WebSocketService';
import BeepService from './BeepService';
import { useTheme } from './ThemeContext';

interface CopilotSession {
  session_id: string;
  pr_number: number;
  started_at: string;
  ended_at: string | null;
  exit_code: number | null;
  status: string; // 'active', 'completed', 'failed'
}

interface CopilotSummary {
  id: number;
  session_id: string;
  summary_text: string;
  start_entry_num: number;
  end_entry_num: number;
  timestamp: string;
}

interface CopilotLog {
  id: number;
  session_id: string;
  log_index: number;
  log_line: string;
  timestamp: string;
  is_code: boolean;
  entry_num: number;
}

interface ViewLogsScreenProps {
  prNumber: number;
  prTitle: string;
  onBack: () => void;
  sessions: CopilotSession[];
  summaries: CopilotSummary[];
  logs: CopilotLog[];
  onClearData: () => void; // Callback to clear old data when switching PRs
}

export default function ViewLogsScreen({
  prNumber,
  prTitle,
  onBack,
  sessions: sessionsProp,
  summaries: summariesProp,
  logs: logsProp,
  onClearData,
}: ViewLogsScreenProps) {
  const { theme } = useTheme();
  // Use props directly for data, only store UI state locally
  const [selectedSession, setSelectedSession] = useState<CopilotSession | null>(null);
  const [expandedSummaryId, setExpandedSummaryId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingSummaries, setLoadingSummaries] = useState(false);
  const [loadingLogs, setLoadingLogs] = useState(false);
  const [noDataFound, setNoDataFound] = useState(false);
  
  // Track the current PR number to detect changes
  const previousPrNumber = useRef<number | null>(null);
  const loadingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const previousSummaryCount = useRef<number>(0);

  // Detect when PR changes and clear old data
  useEffect(() => {
    if (previousPrNumber.current !== null && previousPrNumber.current !== prNumber) {
      console.log(`PR changed from #${previousPrNumber.current} to #${prNumber}, clearing old data`);
      // Clear old data and play loading sound
      onClearData();
      setSelectedSession(null);
      setExpandedSummaryId(null);
      setLoading(true);
      setLoadingSummaries(false);
      setLoadingLogs(false);
      setNoDataFound(false);
      previousSummaryCount.current = 0;
      BeepService.playLoadingSound();
      
      // Clear any existing timeout
      if (loadingTimeoutRef.current) {
        clearTimeout(loadingTimeoutRef.current);
        loadingTimeoutRef.current = null;
      }
    }
    previousPrNumber.current = prNumber;
  }, [prNumber, onClearData]);

  // Auto-select first session when sessions arrive
  useEffect(() => {
    if (sessionsProp.length > 0) {
      setLoading(false);
      setNoDataFound(false);
      
      // Clear the loading timeout since we got data
      if (loadingTimeoutRef.current) {
        clearTimeout(loadingTimeoutRef.current);
        loadingTimeoutRef.current = null;
      }
      
      // Auto-select first session if we don't have one selected
      if (!selectedSession && sessionsProp.length === 1) {
        console.log('Auto-selecting first session:', sessionsProp[0].session_id);
        setSelectedSession(sessionsProp[0]);
        setLoadingSummaries(true);
        WebSocketService.requestSessionSummaries(sessionsProp[0].session_id);
      }
    }
  }, [sessionsProp]);

  // Update loading states when data arrives
  useEffect(() => {
    if (summariesProp.length > 0) {
      console.log('Summaries updated, count:', summariesProp.length);
      setLoadingSummaries(false);

      // Beep when a new live summary arrives (not on initial batch load)
      if (previousSummaryCount.current > 0 && summariesProp.length > previousSummaryCount.current) {
        console.log('New summary arrived, playing beep');
        BeepService.playBeep(880, 150);
      }
      previousSummaryCount.current = summariesProp.length;
    }
  }, [summariesProp]);

  useEffect(() => {
    if (logsProp.length > 0) {
      console.log('Logs updated, count:', logsProp.length);
      setLoadingLogs(false);
    }
  }, [logsProp]);

  // Request sessions when component mounts or PR changes
  useEffect(() => {
    console.log(`ViewLogsScreen requesting sessions for PR #${prNumber}`);
    setLoading(true);
    setNoDataFound(false);
    WebSocketService.requestPRSessions(prNumber);
    
    // Set timeout to show "no data found" if nothing arrives within 5 seconds
    if (loadingTimeoutRef.current) {
      clearTimeout(loadingTimeoutRef.current);
    }
    
    loadingTimeoutRef.current = setTimeout(() => {
      if (sessionsProp.length === 0) {
        console.log(`No sessions found for PR #${prNumber} after timeout`);
        setLoading(false);
        setNoDataFound(true);
      }
    }, 5000);
    
    // Cleanup timeout on unmount
    return () => {
      if (loadingTimeoutRef.current) {
        clearTimeout(loadingTimeoutRef.current);
        loadingTimeoutRef.current = null;
      }
    };
  }, [prNumber]);

  const handleSelectSession = useCallback((session: CopilotSession) => {
    console.log(`Selected session: ${session.session_id}`);
    setSelectedSession(session);
    setExpandedSummaryId(null);
    setLoadingSummaries(true);
    WebSocketService.requestSessionSummaries(session.session_id);
  }, []);

  const handleExpandSummary = useCallback((summaryId: number, sessionId: string, startEntry: number, endEntry: number) => {
    if (expandedSummaryId === summaryId) {
      // Collapse
      setExpandedSummaryId(null);
    } else {
      // Expand and fetch logs for this entry range
      setExpandedSummaryId(summaryId);
      setLoadingLogs(true);
      WebSocketService.requestSessionLogs(sessionId, startEntry, endEntry);
    }
  }, [expandedSummaryId]);

  const formatDate = (dateString: string | null): string => {
    if (!dateString) return 'N/A';
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch {
      return dateString;
    }
  };

  const getStatusIcon = (status: string): string => {
    switch (status) {
      case 'completed':
        return '✓';
      case 'failed':
        return '✗';
      case 'active':
        return '⚡';
      default:
        return '•';
    }
  };

  const renderSessionItem = ({item}: {item: CopilotSession}) => (
    <TouchableOpacity
      style={[
        styles.sessionItem,
        { backgroundColor: theme.card, borderColor: theme.border },
        selectedSession?.session_id === item.session_id && { 
          borderColor: theme.primary, 
          backgroundColor: theme.backgroundSecondary 
        },
      ]}
      onPress={() => handleSelectSession(item)}
      accessible={true}
      accessibilityLabel={`Session started ${formatDate(item.started_at)}, status ${item.status}`}
      accessibilityHint="Double tap to view session summaries">
      <View style={styles.sessionHeader}>
        <Text style={[styles.sessionStatus, { color: theme.primary }]}>
          {getStatusIcon(item.status)} {item.status.toUpperCase()}
        </Text>
        <Text style={[styles.sessionDate, { color: theme.textSecondary }]}>{formatDate(item.started_at)}</Text>
      </View>
      {item.ended_at && (
        <Text style={[styles.sessionEndDate, { color: theme.textTertiary }]}>Ended: {formatDate(item.ended_at)}</Text>
      )}
    </TouchableOpacity>
  );

  const renderSummaryItem = ({item}: {item: CopilotSummary}) => {
    const isExpanded = expandedSummaryId === item.id;
    const relevantLogs = isExpanded
      ? logsProp.filter(
          (log: CopilotLog) =>
            log.entry_num >= item.start_entry_num &&
            log.entry_num <= item.end_entry_num,
        )
      : [];

    return (
      <View
        style={[styles.summaryItem, { backgroundColor: theme.card, borderLeftColor: theme.primary }]}>
        <TouchableOpacity
          onPress={() =>
            handleExpandSummary(
              item.id,
              item.session_id,
              item.start_entry_num,
              item.end_entry_num,
            )
          }
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel={`Summary: ${item.summary_text}. Entries ${item.start_entry_num} to ${item.end_entry_num}`}
          accessibilityHint={
            isExpanded
              ? 'Double tap to collapse full logs'
              : 'Double tap to show full logs'
          }>
          <Text style={[styles.summaryText, { color: theme.text }]}>{item.summary_text}</Text>
          <Text style={[styles.summaryMeta, { color: theme.textSecondary }]}>
            Entries {item.start_entry_num}-{item.end_entry_num} •{' '}
            {formatDate(item.timestamp)}
          </Text>
          <Text style={[styles.expandButton, { color: theme.primary }]}>
            {isExpanded ? '▼ Hide Full Logs' : '▶ Show Full Logs'}
          </Text>
        </TouchableOpacity>

        {isExpanded && (
          <View style={[styles.fullLogsContainer, { borderTopColor: theme.border }]}>
            {loadingLogs ? (
              <ActivityIndicator size="small" color={theme.primary} />
            ) : (
              relevantLogs.map((log: CopilotLog) => (
                <Text
                  key={log.id}
                  style={[
                    styles.logLine,
                    { color: theme.text },
                    log.is_code && styles.logLineCode,
                  ]}>
                  {log.log_line}
                </Text>
              ))
            )}
          </View>
        )}
      </View>
    );
  };

  if (loading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: theme.background }]} edges={['bottom']}>
        <View style={[styles.header, { backgroundColor: theme.card, borderBottomColor: theme.border }]}>
          <TouchableOpacity
            onPress={onBack}
            style={styles.backButton}
            accessible={true}
            accessibilityLabel="Back to PRs"
            accessibilityHint="Double tap to go back">
            <Text style={[styles.backButtonText, { color: theme.primary }]}>← Back</Text>
          </TouchableOpacity>
          <Text style={[styles.title, { color: theme.text }]}>PR #{prNumber} Logs</Text>
        </View>
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" color={theme.primary} />
          <Text style={[styles.loadingText, { color: theme.textSecondary }]}>Loading sessions...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (noDataFound || sessionsProp.length === 0) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: theme.background }]} edges={['bottom']}>
        <View style={[styles.header, { backgroundColor: theme.card, borderBottomColor: theme.border }]}>
          <TouchableOpacity
            onPress={onBack}
            style={styles.backButton}
            accessible={true}
            accessibilityLabel="Back to PRs">
            <Text style={[styles.backButtonText, { color: theme.primary }]}>← Back</Text>
          </TouchableOpacity>
          <Text style={[styles.title, { color: theme.text }]}>PR #{prNumber} Logs</Text>
        </View>
        <View style={styles.centerContainer}>
          <Text style={[styles.emptyText, { color: theme.textSecondary }]}>No Copilot session logs found for this PR</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.background }]} edges={['bottom']}>
      <View style={[styles.header, { backgroundColor: theme.card, borderBottomColor: theme.border }]}>
        <TouchableOpacity
          onPress={onBack}
          style={styles.backButton}
          accessible={true}
          accessibilityLabel="Back to PRs">
          <Text style={[styles.backButtonText, { color: theme.primary }]}>← Back</Text>
        </TouchableOpacity>
        <Text style={[styles.title, { color: theme.text }]}>PR #{prNumber} Logs</Text>
        <Text style={[styles.subtitle, { color: theme.textSecondary }]}>{prTitle}</Text>
      </View>

      {/* Sessions List - horizontal FlatList is OK */}
      {sessionsProp.length > 1 && (
        <View style={[styles.sessionsSection, { borderBottomColor: theme.border }]}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>
            {sessionsProp.length} Session{sessionsProp.length > 1 ? 's' : ''}
          </Text>
          <FlatList
            data={sessionsProp}
            keyExtractor={item => item.session_id}
            renderItem={renderSessionItem}
            horizontal
            showsHorizontalScrollIndicator={false}
            style={styles.sessionsList}
          />
        </View>
      )}

      {/* Summaries List - main vertical scrolling area */}
      {selectedSession && (
        <View 
          style={styles.summariesSection}
          accessibilityLiveRegion="polite">
          <Text 
            style={[styles.sectionTitle, { color: theme.text }]}
            accessible={true}
            accessibilityRole="header"
            accessibilityLabel={`Session summaries, ${summariesProp.length} available`}>
            Session Summary
            {selectedSession.status === 'active' && ' (Live)'}
          </Text>
          {loadingSummaries ? (
            <View style={styles.centerContainer}>
              <ActivityIndicator size="small" color={theme.primary} />
            </View>
          ) : summariesProp.length === 0 ? (
            <Text style={[styles.emptyText, { color: theme.textSecondary }]}>No summaries available</Text>
          ) : (
            <FlatList
              data={summariesProp}
              keyExtractor={item => item.id.toString()}
              renderItem={renderSummaryItem}
              style={styles.summariesList}
            />
          )}
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    padding: 16,
    paddingTop: 8,
    borderBottomWidth: 1,
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    alignSelf: 'flex-start',
    marginBottom: 12,
  },
  backButtonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  subtitle: {
    fontSize: 14,
    marginTop: 4,
  },
  content: {
    flex: 1,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
  },
  emptyText: {
    fontSize: 16,
    textAlign: 'center',
  },
  sessionsSection: {
    borderBottomWidth: 1,
    paddingVertical: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    paddingHorizontal: 16,
    marginBottom: 8,
  },
  sessionsList: {
    paddingHorizontal: 16,
  },
  sessionItem: {
    borderRadius: 8,
    padding: 12,
    marginRight: 12,
    minWidth: 200,
    borderWidth: 2,
  },
  sessionItemSelected: {
  },
  sessionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  sessionStatus: {
    fontSize: 12,
    fontWeight: '700',
  },
  sessionDate: {
    fontSize: 12,
  },
  sessionEndDate: {
    fontSize: 11,
    marginTop: 4,
  },
  summariesSection: {
    flex: 1,
    paddingTop: 12,
  },
  summariesList: {
    flex: 1,
    paddingHorizontal: 16,
  },
  summaryItem: {
    borderRadius: 8,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
  },
  summaryText: {
    fontSize: 16,
    marginBottom: 8,
    lineHeight: 22,
  },
  summaryMeta: {
    fontSize: 12,
    marginBottom: 8,
  },
  expandButton: {
    fontSize: 14,
    fontWeight: '600',
  },
  fullLogsContainer: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
  },
  logLine: {
    fontSize: 12,
    fontFamily: 'Courier',
    marginBottom: 4,
    lineHeight: 16,
  },
  logLineCode: {
    backgroundColor: '#2D2D2D',
    color: '#FFFFFF',
    padding: 4,
    borderRadius: 4,
  },
});
