/**
 * ReviewPane - Submit a yes/no review on a community PR tool
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTheme } from './ThemeContext';
import WebSocketService from './WebSocketService';
import Config from './config';

interface ReviewPaneProps {
  prNumber: number;
  prTitle: string;
  prBody?: string;
  onBack: () => void;
}

type ReviewVerdict = 'approve' | 'reject' | null;

export default function ReviewPane({ prNumber, prTitle, prBody, onBack }: ReviewPaneProps) {
  const { theme } = useTheme();
  const [verdict, setVerdict] = useState<ReviewVerdict>(null);
  const [comment, setComment] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = () => {
    if (verdict === null) {
      Alert.alert('No verdict selected', 'Please choose Approve or Request Changes before submitting.');
      return;
    }

    // In review mode, approval goes through the user's primary server (for GitHub identity).
    // Fail fast if it's not connected rather than waiting for the 15s timeout.
    if (!WebSocketService.isConnected()) {
      Alert.alert(
        'Not Connected',
        'Your personal server is not connected. Please go to Settings and connect to your server — it\'s needed to submit reviews with your GitHub identity.'
      );
      return;
    }

    Alert.alert(
      `Confirm Review`,
      `Submit a ${verdict === 'approve' ? '✅ Approval' : '❌ Request Changes'} for PR #${prNumber}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Submit',
          onPress: async () => {
            setSubmitting(true);
            // In review mode, the user's server must post to the general repo's PR.
            // Pass REVIEW_GITHUB_REPO so stream_server uses it instead of its own GITHUB_REPO.
            const result = await WebSocketService.submitToolReview(
              prNumber,
              verdict === 'approve',
              comment.trim(),
              Config.APP_MODE === 'review' ? Config.REVIEW_GITHUB_REPO : undefined
            );
            setSubmitting(false);
            if (result.success) {
              setSubmitted(true);
            } else {
              Alert.alert(
                'Submission Failed',
                result.error || 'Could not submit review. Please check your connection and try again.'
              );
            }
          }
        }
      ]
    );
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.background }]} edges={['bottom']}>
      {/* Header */}
      <View style={[styles.header, { backgroundColor: theme.background, borderBottomColor: theme.border }]}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={onBack}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel="Back to pull request list"
          accessibilityHint="Double tap to go back">
          <Text style={[styles.backButtonText, { color: theme.primary }]}>← Back</Text>
        </TouchableOpacity>
        <Text
          style={[styles.headerTitle, { color: theme.text }]}
          accessible={true}
          accessibilityRole="header">
          Review PR #{prNumber}
        </Text>
      </View>

      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        {/* PR title */}
        <View style={[styles.prTitleCard, { backgroundColor: theme.card, borderColor: theme.border }]}>
          <Text style={[styles.prTitleLabel, { color: theme.textSecondary }]}>Pull Request</Text>
          <Text style={[styles.prTitleText, { color: theme.text }]}>{prTitle}</Text>
        </View>

        {/* PR description */}
        {!!prBody && (
          <View style={[styles.prTitleCard, { backgroundColor: theme.card, borderColor: theme.border }]}>
            <Text style={[styles.prTitleLabel, { color: theme.textSecondary }]}>Description</Text>
            <Text style={[styles.prBodyText, { color: theme.text }]}>{prBody}</Text>
          </View>
        )}

        {submitted ? (
          /* Success state */
          <View style={styles.successContainer}>
            <Text style={[styles.successIcon]} accessibilityLabel="Success">✅</Text>
            <Text style={[styles.successTitle, { color: theme.text }]}>Review Submitted</Text>
            <Text style={[styles.successSubtitle, { color: theme.textSecondary }]}>
              Your {verdict === 'approve' ? 'approval' : 'change request'} for PR #{prNumber} has been sent.
            </Text>
            <TouchableOpacity
              style={[styles.backToListButton, { backgroundColor: theme.primary }]}
              onPress={onBack}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Back to pull request list">
              <Text style={styles.backToListButtonText}>Back to PR List</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <>
            {/* Verdict buttons */}
            <Text style={[styles.sectionLabel, { color: theme.textSecondary }]}>Your verdict</Text>
            <View style={styles.verdictRow}>
              <TouchableOpacity
                style={[
                  styles.verdictButton,
                  { borderColor: theme.success },
                  verdict === 'approve' && { backgroundColor: theme.success },
                ]}
                onPress={() => setVerdict('approve')}
                accessible={true}
                accessibilityRole="button"
                accessibilityState={{ selected: verdict === 'approve' }}
                accessibilityLabel="Approve this pull request"
                accessibilityHint="Double tap to select Approve">
                <Text style={[
                  styles.verdictButtonText,
                  { color: verdict === 'approve' ? '#fff' : theme.success },
                ]}>
                  ✅ Approve
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[
                  styles.verdictButton,
                  { borderColor: theme.error },
                  verdict === 'reject' && { backgroundColor: theme.error },
                ]}
                onPress={() => setVerdict('reject')}
                accessible={true}
                accessibilityRole="button"
                accessibilityState={{ selected: verdict === 'reject' }}
                accessibilityLabel="Request changes on this pull request"
                accessibilityHint="Double tap to select Request Changes">
                <Text style={[
                  styles.verdictButtonText,
                  { color: verdict === 'reject' ? '#fff' : theme.error },
                ]}>
                  ❌ Request Changes
                </Text>
              </TouchableOpacity>
            </View>

            {/* Comment */}
            <Text style={[styles.sectionLabel, { color: theme.textSecondary }]}>
              Comment <Text style={{ color: theme.textTertiary }}>(optional)</Text>
            </Text>
            <TextInput
              style={[styles.commentInput, {
                backgroundColor: theme.inputBackground,
                borderColor: theme.inputBorder,
                color: theme.text,
              }]}
              value={comment}
              onChangeText={setComment}
              placeholder="Describe what works well or what needs to change..."
              placeholderTextColor={theme.inputPlaceholder}
              multiline
              numberOfLines={5}
              textAlignVertical="top"
              accessible={true}
              accessibilityLabel="Review comment"
              accessibilityHint="Optional comment to accompany your verdict"
            />

            {/* Submit */}
            <TouchableOpacity
              style={[
                styles.submitButton,
                { backgroundColor: verdict ? theme.primary : theme.border },
              ]}
              onPress={handleSubmit}
              disabled={submitting || verdict === null}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Submit review"
              accessibilityHint="Double tap to submit your review">
              {submitting ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.submitButtonText}>Submit Review</Text>
              )}
            </TouchableOpacity>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    gap: 12,
  },
  backButton: { padding: 4 },
  backButtonText: { fontSize: 16, fontWeight: '500' },
  headerTitle: { fontSize: 17, fontWeight: '600', flex: 1 },
  content: { padding: 16, gap: 12 },
  prTitleCard: {
    borderRadius: 10,
    borderWidth: 1,
    padding: 14,
    marginBottom: 8,
  },
  prTitleLabel: { fontSize: 12, fontWeight: '500', marginBottom: 4 },
  prTitleText: { fontSize: 15, fontWeight: '600' },
  prBodyText: { fontSize: 14, lineHeight: 20 },
  sectionLabel: { fontSize: 13, fontWeight: '500', marginTop: 8, marginBottom: 6 },
  verdictRow: { flexDirection: 'row', gap: 12, marginBottom: 8 },
  verdictButton: {
    flex: 1,
    borderWidth: 2,
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: 'center',
  },
  verdictButtonText: { fontSize: 15, fontWeight: '600' },
  commentInput: {
    borderWidth: 1,
    borderRadius: 10,
    padding: 12,
    fontSize: 15,
    minHeight: 110,
    marginBottom: 8,
  },
  submitButton: {
    borderRadius: 10,
    paddingVertical: 16,
    alignItems: 'center',
    marginTop: 8,
  },
  submitButtonText: { color: '#fff', fontSize: 16, fontWeight: '700' },
  successContainer: { alignItems: 'center', paddingTop: 40, gap: 12 },
  successIcon: { fontSize: 56 },
  successTitle: { fontSize: 22, fontWeight: '700' },
  successSubtitle: { fontSize: 15, textAlign: 'center' },
  backToListButton: { marginTop: 16, borderRadius: 10, paddingVertical: 14, paddingHorizontal: 32 },
  backToListButtonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
