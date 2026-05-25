/**
 * ModelPickerScreen
 *
 * Standalone screen for selecting which LLM model the server should use.
 * Opened from Settings; back arrow returns the user to Settings.
 */

import React from 'react';
import {
  StyleSheet,
  Text,
  View,
  Pressable,
  ScrollView,
} from 'react-native';
import { useTheme } from './ThemeContext';

interface ModelPickerScreenProps {
  onBack: () => void;
  selectedModel: string | null;
  serverDefaultModel: string;
  availableModels: string[];
  onSelect: (model: string | null) => void;
}

interface ModelRowProps {
  model: string | null;
  selected: boolean;
  serverDefaultModel: string;
  onPress: (model: string | null) => void;
}

function ModelRow({ model, selected, serverDefaultModel, onPress }: ModelRowProps) {
  const { theme } = useTheme();
  const isDefault = model === null;
  const primaryLabel = isDefault ? 'Server default' : model!;
  const subtitle = isDefault && serverDefaultModel ? serverDefaultModel : null;
  const accentColor = isDefault ? theme.info : theme.primary;

  return (
    <Pressable
      style={({ pressed }) => [
        styles.row,
        { borderColor: theme.border, backgroundColor: theme.card },
        selected && { backgroundColor: accentColor + '20', borderColor: accentColor },
        pressed && { opacity: 0.6 },
      ]}
      onPress={() => onPress(model)}
      accessible={true}
      accessibilityRole="radio"
      accessibilityLabel={isDefault ? 'Server default model' : model!}
      accessibilityState={{ selected }}>
      <View style={styles.rowText}>
        <Text style={[styles.rowLabel, { color: theme.text }]}>{primaryLabel}</Text>
        {subtitle && (
          <Text style={[styles.rowSubtitle, { color: theme.textSecondary }]}>{subtitle}</Text>
        )}
      </View>
      {selected && (
        <Text style={[styles.checkmark, { color: accentColor }]}>✓</Text>
      )}
    </Pressable>
  );
}

export default function ModelPickerScreen({
  onBack,
  selectedModel,
  serverDefaultModel,
  availableModels,
  onSelect,
}: ModelPickerScreenProps) {
  const { theme } = useTheme();

  return (
    <View style={[styles.container, { backgroundColor: theme.background }]}>
      <View style={[styles.header, { backgroundColor: theme.card, borderBottomColor: theme.border }]}>
        <Pressable
          onPress={onBack}
          style={({ pressed }) => [styles.backButton, pressed && { opacity: 0.6 }]}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel="Back to settings"
          accessibilityHint="Double tap to return to settings">
          <Text style={[styles.backButtonText, { color: theme.primary }]}>← Back</Text>
        </Pressable>
        <Text style={[styles.title, { color: theme.text }]}>AI Model</Text>
        <Text style={[styles.subtitle, { color: theme.textSecondary }]}>
          Choose which model the server should use.
        </Text>
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        contentInsetAdjustmentBehavior="automatic">
        <ModelRow
          model={null}
          selected={selectedModel === null}
          serverDefaultModel={serverDefaultModel}
          onPress={onSelect}
        />
        {availableModels.length === 0 ? (
          <Text style={[styles.emptyText, { color: theme.textSecondary }]}>
            Connect to a server to see available models.
          </Text>
        ) : (
          availableModels.map((model) => (
            <ModelRow
              key={model}
              model={model}
              selected={selectedModel === model}
              serverDefaultModel={serverDefaultModel}
              onPress={onSelect}
            />
          ))
        )}
      </ScrollView>
    </View>
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
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    borderRadius: 12,
    borderWidth: 1,
    paddingVertical: 14,
    paddingHorizontal: 16,
    marginBottom: 10,
  },
  rowText: {
    flex: 1,
  },
  rowLabel: {
    fontSize: 16,
    fontWeight: '500',
  },
  rowSubtitle: {
    fontSize: 13,
    marginTop: 2,
  },
  checkmark: {
    fontSize: 20,
    fontWeight: '700',
    marginLeft: 12,
  },
  emptyText: {
    fontSize: 14,
    textAlign: 'center',
    marginTop: 16,
  },
});
