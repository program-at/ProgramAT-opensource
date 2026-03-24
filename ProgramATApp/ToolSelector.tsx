/**
 * ToolSelector Component
 * Displays available tools from the tools folder and allows selection
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ScrollView,
  AccessibilityInfo,
  findNodeHandle,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Config from './config';
import WebSocketService from './WebSocketService';
import BeepService from './BeepService';
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
  custom_gpt?: boolean;
  gpt_query?: string;
  system_instruction?: string;
  query_interval?: number;
}

interface ToolSelectorProps {
  onToolSelect: (tool: Tool) => void;
  selectedTool: Tool | null;
  issueTools?: Tool[];
  productionMode?: boolean;
  selectedIssue?: {number: number; title: string} | null;
}

export default function ToolSelector({ onToolSelect, selectedTool, issueTools = [], productionMode = false, selectedIssue = null }: ToolSelectorProps) {
  const { theme } = useTheme();
  const [tools, setTools] = useState<Tool[]>([]);
  const [loading, setLoading] = useState(true);
  const [expectingNewTools, setExpectingNewTools] = useState(false);
  const [loadingStartTime, setLoadingStartTime] = useState<number>(0); // Track when loading started
  const headerRef = useRef<Text>(null);

  console.log('[ToolSelector] Rendered - productionMode:', productionMode, 'issueTools:', issueTools.length);

  // Set accessibility focus to header when component mounts
  useEffect(() => {
    const timeout = setTimeout(() => {
      if (headerRef.current) {
        const reactTag = findNodeHandle(headerRef.current);
        if (reactTag) {
          AccessibilityInfo.setAccessibilityFocus(reactTag);
        }
      }
    }, 100); // Small delay to ensure component is rendered
    
    return () => clearTimeout(timeout);
  }, []);

  useEffect(() => {
    console.log('[ToolSelector] useEffect triggered - calling loadTools(), productionMode:', productionMode, 'selectedIssue:', selectedIssue?.number);
    loadTools();
  }, [productionMode, selectedIssue]);

  // Update tools when issueTools changes (both development and production modes)
  useEffect(() => {
    if (issueTools.length > 0) {
      console.log('[ToolSelector] Received tools:', issueTools.length);
      console.log('[ToolSelector] Tool names:', issueTools.map(t => t.name));
      
      // Update the tools
      setTools(issueTools);
      
      // If we were expecting new tools (i.e., we cleared tools and requested fresh ones)
      // OR if we had no tools before, then stop loading
      if (expectingNewTools || tools.length === 0) {
        console.log('[ToolSelector] Fresh tools arrived, stopping loading');
        setLoading(false);
        setExpectingNewTools(false);
      } else {
        console.log('[ToolSelector] Tools updated but not expecting new ones');
      }
    }
  }, [issueTools, expectingNewTools, tools.length]);

  // Loading sound effect for tool fetching
  useEffect(() => {
    let beepTimer: ReturnType<typeof setTimeout> | null = null;
    
    if (loading) {
      console.log('[ToolSelector] Loading tools, will beep after 3 seconds if still loading');
      // Wait 3 seconds before starting beep
      beepTimer = setTimeout(() => {
        console.log('[ToolSelector] 3 seconds elapsed, starting loading sound');
        BeepService.startLoadingSound();
      }, 3000);
    } else {
      console.log('[ToolSelector] Stopping loading sound');
      BeepService.stopLoadingSound();
    }

    // Cleanup on unmount or when loading state changes
    return () => {
      if (beepTimer) {
        clearTimeout(beepTimer);
      }
      BeepService.stopLoadingSound();
    };
  }, [loading]);

  const loadTools = async () => {
    setLoading(true);
    setExpectingNewTools(true);
    setLoadingStartTime(Date.now());
    
    // Clear existing tools so we know when fresh ones actually arrive
    setTools([]);
    
    // Production mode: request tools from main branch only
    if (productionMode) {
      console.log('[ToolSelector] Production mode - requesting tools from main branch');
      const success = WebSocketService.requestProductionTools();
      
      if (!success) {
        console.error('[ToolSelector] Failed to request production tools - WebSocket not connected');
        setLoading(false);
        setExpectingNewTools(false);
        return;
      }
      
      // Safety timeout in case tools never arrive
      setTimeout(() => {
        if (tools.length === 0) {
          console.warn('[ToolSelector] Timeout - no tools received');
          setLoading(false);
          setExpectingNewTools(false);
        }
      }, 10000); // 10 second safety timeout
      
      return;
    }
    
    // Development mode: Check if a PR is selected
    console.log('[ToolSelector] Development mode - checking if PR is selected');
    
    // If no PR is selected, stop loading immediately
    if (!selectedIssue) {
      console.log('[ToolSelector] No PR selected in development mode');
      setLoading(false);
      setExpectingNewTools(false);
      return;
    }
    
    // PR is selected - request tools and wait for them to arrive
    console.log('[ToolSelector] PR selected - requesting tools via WebSocket');
    
    // Safety timeout in case tools never arrive
    setTimeout(() => {
      if (tools.length === 0) {
        console.warn('[ToolSelector] Timeout - no tools received for selected PR');
        setLoading(false);
        setExpectingNewTools(false);
      }
    }, 10000);
  };

  const handleToolPress = (tool: Tool) => {
    onToolSelect(tool);
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.backgroundSecondary }]} edges={[]} accessible={false}>
      <View style={[styles.header, { backgroundColor: theme.background, borderBottomColor: theme.border }]} accessible={false}>
        <Text 
          ref={headerRef}
          style={[styles.headerText, { color: theme.text }]}
          accessible={true}
          accessibilityRole="header"
          accessibilityLabel="Select a Tool">
          Select a Tool
        </Text>
        <Text style={[styles.headerSubtext, { color: theme.textSecondary }]} accessible={false}>
          {productionMode 
            ? `Production tools from ${Config.PRODUCTION_BRANCH} branch`
            : 'Choose a tool to run or create a new one'}
        </Text>
        {productionMode && (
          <View style={[styles.productionBadge, { backgroundColor: theme.success }]} accessible={false}>
            <Text style={[styles.productionBadgeText, { color: '#fff' }]} accessible={false}>🚀 Production Mode</Text>
          </View>
        )}
      </View>

      {loading ? (
        <View style={styles.loadingContainer}>
          <Text style={[styles.loadingText, { color: theme.textSecondary }]}>Loading tools...</Text>
        </View>
      ) : tools.length === 0 ? (
        <View style={styles.emptyContainer}>
          {!productionMode && !selectedIssue ? (
            <>
              <Text style={[styles.emptyText, { color: theme.textTertiary }]}>No Branch Selected</Text>
              <Text style={[styles.emptySubtext, { color: theme.textTertiary }]}>
                Go to the PRs tab to select a pull request
              </Text>
            </>
          ) : (
            <>
              <Text style={[styles.emptyText, { color: theme.textTertiary }]}>No tools found</Text>
              <Text style={[styles.emptySubtext, { color: theme.textTertiary }]}>
                Request a tool to be created in your issue description
              </Text>
            </>
          )}
        </View>
      ) : (
        <ScrollView style={styles.toolList}>
          {tools.map((tool, index) => {
            // Build comprehensive accessibility label including all metadata
            const accessibilityParts = [tool.name];
            if (tool.description) accessibilityParts.push(tool.description);
            if (tool.pr_title) accessibilityParts.push(`Pull request: ${tool.pr_title}`);
            if (tool.branch_name) accessibilityParts.push(`Branch: ${tool.branch_name}`);
            if (tool.language) accessibilityParts.push(`Language: ${tool.language}`);
            if (selectedTool?.name === tool.name) accessibilityParts.push('Selected');
            
            return (
              <TouchableOpacity
                key={index}
                style={[
                  styles.toolCard,
                  { 
                    backgroundColor: theme.card, 
                    borderColor: theme.border 
                  },
                  selectedTool?.name === tool.name && { 
                    backgroundColor: theme.backgroundSecondary, 
                    borderColor: theme.primary 
                  }
                ]}
                onPress={() => handleToolPress(tool)}
                accessibilityRole="button"
                accessibilityLabel={accessibilityParts.join('. ')}
                accessibilityHint="Double tap to select this tool"
                accessibilityState={{ selected: selectedTool?.name === tool.name }}>
                <View style={styles.toolHeader}>
                  <Text 
                    style={[
                      styles.toolName,
                      { color: theme.text },
                      selectedTool?.name === tool.name && { color: theme.primary }
                    ]}
                    accessible={false}>
                    {tool.name}
                  </Text>
                  {selectedTool?.name === tool.name && (
                    <View style={[styles.selectedBadge, { backgroundColor: theme.success }]} accessible={false}>
                      <Text style={styles.selectedBadgeText} accessible={false}>✓</Text>
                    </View>
                  )}
                </View>
                {tool.description && (
                  <Text style={[styles.toolDescription, { color: theme.textSecondary }]} accessible={false}>{tool.description}</Text>
                )}
                {tool.pr_title && (
                  <Text style={[styles.toolMeta, { color: theme.textTertiary }]} accessible={false}>PR: {tool.pr_title}</Text>
                )}
                {tool.branch_name && (
                  <Text style={[styles.toolMeta, { color: theme.textTertiary }]} accessible={false}>Branch: {tool.branch_name}</Text>
                )}
                {tool.language && (
                  <Text style={[styles.toolMeta, { color: theme.textTertiary }]} accessible={false}>Language: {tool.language}</Text>
                )}
              </TouchableOpacity>
            );
          })}
        </ScrollView>
      )}

      <View style={[styles.footer, { backgroundColor: theme.background, borderTopColor: theme.border }]}>
        <Text style={[styles.footerText, { color: theme.textTertiary }]}>
          Request new tools in your issue description
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    paddingHorizontal: 16,
    paddingVertical: 16,
    borderBottomWidth: 1,
  },
  headerText: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  headerSubtext: {
    fontSize: 14,
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
    paddingHorizontal: 32,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 14,
    textAlign: 'center',
  },
  toolList: {
    flex: 1,
    paddingHorizontal: 16,
    paddingTop: 16,
  },
  toolCard: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  toolHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  toolName: {
    fontSize: 18,
    fontWeight: '600',
    flex: 1,
  },
  selectedBadge: {
    borderRadius: 12,
    width: 24,
    height: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  selectedBadgeText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  toolDescription: {
    fontSize: 14,
    lineHeight: 20,
  },
  toolMeta: {
    fontSize: 11,
    marginTop: 4,
    fontStyle: 'italic',
  },
  productionBadge: {
    marginTop: 8,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    alignSelf: 'flex-start',
  },
  productionBadgeText: {
    fontSize: 12,
    fontWeight: '600',
  },
  footer: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderTopWidth: 1,
  },
  footerText: {
    fontSize: 12,
    textAlign: 'center',
    fontStyle: 'italic',
  },
});
