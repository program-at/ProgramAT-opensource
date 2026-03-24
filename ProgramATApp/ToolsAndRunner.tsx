/**
 * Tools and Runner Combined Component
 * Shows tool selector by default, with navigation to runner for selected tool
 */

import React, { useState, useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useTheme } from './ThemeContext';
import ToolSelector from './ToolSelector';
import ToolRunner from './ToolRunner';

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

interface ToolsAndRunnerProps {
  issueTools?: Tool[];
  productionMode?: boolean;
  isActive?: boolean; // Whether this tab is currently active
  selectedIssue?: {number: number; title: string} | null;
  onNavigateToChat?: (conversationId?: string) => void; // Callback to navigate to chat tab with optional conversation ID
}

type ViewMode = 'tool-selector' | 'tool-runner';

export default function ToolsAndRunner({ 
  issueTools = [],
  productionMode = false,
  isActive = true,
  selectedIssue = null,
  onNavigateToChat
}: ToolsAndRunnerProps) {
  const { theme } = useTheme();
  const [viewMode, setViewMode] = useState<ViewMode>('tool-selector');
  const [selectedTool, setSelectedTool] = useState<Tool | null>(null);

  // Load persisted selected tool on mount
  useEffect(() => {
    const loadSelectedTool = async () => {
      try {
        const savedTool = await AsyncStorage.getItem('selectedTool');
        if (savedTool) {
          const tool = JSON.parse(savedTool);
          console.log('[ToolsAndRunner] Loaded persisted tool:', tool.name);
          setSelectedTool(tool);
        }
      } catch (error) {
        console.error('[ToolsAndRunner] Error loading selected tool:', error);
      }
    };
    loadSelectedTool();
  }, []);

  // Reset to tool selector view whenever the tab becomes active
  useEffect(() => {
    if (isActive) {
      setViewMode('tool-selector');
    }
  }, [isActive]);

  // Validate that selected tool still exists when issueTools changes
  // and update its fields (e.g. gpt_query) from the fresh data
  useEffect(() => {
    if (selectedTool && issueTools.length > 0) {
      // Check if the currently selected tool is still in the list
      const freshTool = issueTools.find(
        tool => tool.name === selectedTool.name && tool.path === selectedTool.path
      );
      if (!freshTool) {
        console.log('[ToolsAndRunner] Selected tool no longer available, clearing selection');
        setSelectedTool(null);
        setViewMode('tool-selector');
        AsyncStorage.removeItem('selectedTool');
      } else if (
        freshTool.gpt_query !== selectedTool.gpt_query ||
        freshTool.code !== selectedTool.code ||
        freshTool.system_instruction !== selectedTool.system_instruction ||
        freshTool.query_interval !== selectedTool.query_interval
      ) {
        // Update the selected tool with fresh data (e.g. updated gpt_query from code)
        console.log('[ToolsAndRunner] Updating selected tool with fresh data (gpt_query or code changed)');
        const updated = { ...selectedTool, ...freshTool };
        setSelectedTool(updated);
        AsyncStorage.setItem('selectedTool', JSON.stringify(updated)).catch(error => {
          console.error('[ToolsAndRunner] Error persisting updated tool:', error);
        });
      }
    }
  }, [issueTools]);

  const handleToolSelect = (tool: Tool) => {
    setSelectedTool(tool);
    setViewMode('tool-runner'); // Navigate to runner after selecting tool
    // Persist the selected tool
    AsyncStorage.setItem('selectedTool', JSON.stringify(tool)).catch(error => {
      console.error('[ToolsAndRunner] Error saving selected tool:', error);
    });
  };

  const handleBackToTools = () => {
    setViewMode('tool-selector'); // Go back to tool selector
  };

  return (
    <View style={[styles.container, { backgroundColor: theme.background }]}>
      {viewMode === 'tool-selector' ? (
        <ToolSelector 
          onToolSelect={handleToolSelect}
          selectedTool={selectedTool}
          issueTools={issueTools}
          productionMode={productionMode}
          selectedIssue={selectedIssue}
        />
      ) : (
        <ToolRunner 
          selectedTool={selectedTool}
          onBack={handleBackToTools}
          showBackButton={true}
          onNavigateToChat={onNavigateToChat}
          isActive={isActive}
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
