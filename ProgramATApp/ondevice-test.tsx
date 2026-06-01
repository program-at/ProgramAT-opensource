// 
/**
 * OnDeviceTest - On-device Python execution test screen
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Button,
  TextInput,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTheme } from './ThemeContext';
import { useModel, GEMMA_4_E2B_IT } from 'react-native-litert-lm';

interface OnDeviceTestProps {
  onBack: () => void;
}

export default function OnDeviceTest({ onBack }: OnDeviceTestProps) {
  const { theme } = useTheme();

  const {
    model,
    isReady,
    downloadProgress,
    error,
  } = useModel(GEMMA_4_E2B_IT, {
    backend: 'cpu',
    autoLoad: true,
    systemPrompt: 'You are a helpful assistant.',
    enableMemoryTracking: true,
  });

  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

//   const generate = async () => {
//     if (!model) return;

//     try {
//       const response = await model.sendMessage('Hello!');
//       console.log(response);
//     } catch (err) {
//       console.error(err);
//     }
//   };
    const generate = async () => {
    if (!model || !input.trim()) return;

    try {
        setIsGenerating(true);

        const result = await model.sendMessage(input);

        console.log(result);

        if (typeof result === 'string') {
        setResponse(result);
        } else {
        setResponse(JSON.stringify(result, null, 2));
        }
    } catch (err) {
        console.error(err);
        setResponse(`Error: ${String(err)}`);
    } finally {
        setIsGenerating(false);
    }
    };

  return (
    <SafeAreaView
      style={[
        styles.container,
        { backgroundColor: theme.background },
      ]}
    >
      <View style={styles.content}>
        <Text style={[styles.title, { color: theme.text }]}>
          On-Device Test
        </Text>

        <Text
          style={[
            styles.subtitle,
            { color: theme.textSecondary },
          ]}
        >
          On-device Python execution test harness
        </Text>

        {error && (
          <Text style={{ color: 'red', marginTop: 20 }}>
            {String(error)}
          </Text>
        )}

        {!isReady ? (
          <View style={{ marginTop: 20 }}>
            <ActivityIndicator />
            <Text style={{ color: theme.text }}>
              Loading model... {Math.round(downloadProgress * 100)}%
            </Text>
          </View>
        ) : (
          <View style={{ marginTop: 20 }}>
            {/* <Button
              title="Generate"
              onPress={generate}
            /> */}
            <TextInput
                value={input}
                onChangeText={setInput}
                placeholder="Type a message..."
                multiline
                style={[
                styles.input,
                {
                    color: theme.text,
                    borderColor: theme.textSecondary,
                },
                ]}
                placeholderTextColor={theme.textSecondary}
            />

            <Button
                title={isGenerating ? 'Generating...' : 'Generate'}
                onPress={generate}
                disabled={isGenerating}
            />

            {response.length > 0 && (
                <ScrollView style={styles.responseContainer}>
                <Text style={{ color: theme.text }}>
                    {response}
                </Text>
                </ScrollView>
            )}
            </View>

        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 15,
  },
});