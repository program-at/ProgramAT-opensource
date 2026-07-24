export interface ProgressiveResultEvent {
  invocation_id?: string;
  result_index?: number;
  mode?: string;
  model?: string | null;
  text?: string;
  final?: boolean;
  metadata?: {
    model?: string;
    [key: string]: unknown;
  };
}

export function acceptsStreamingProgressEvent(
  streamingActive: boolean,
  event: {mode?: string},
): boolean {
  return event.mode !== 'streaming' || streamingActive;
}

export function acceptsProgressiveResult(
  activeInvocationId: string | null,
  lastResultIndex: number,
  event: ProgressiveResultEvent,
): boolean {
  return (
    typeof event.invocation_id === 'string' &&
    event.invocation_id === activeInvocationId &&
    typeof event.result_index === 'number' &&
    event.result_index > lastResultIndex
  );
}

export function progressiveInvocationIsRunning(
  event: {final?: boolean},
): boolean {
  return !event.final;
}

export function progressiveResultModel(
  event: ProgressiveResultEvent,
): string | null {
  const model = event.model || event.metadata?.model;
  return typeof model === 'string' && model.trim() ? model.trim() : null;
}

export function progressiveResultModelLabel(
  event: ProgressiveResultEvent,
): string | null {
  const model = progressiveResultModel(event)?.toLowerCase();
  if (!model) {
    return null;
  }
  if (model.includes('moondream')) {
    return 'Moondream';
  }
  if (model.includes('gemini')) {
    return 'Gemini';
  }
  if (model.includes('gpt') || model.includes('openai')) {
    return 'GPT';
  }
  return null;
}

export function formatProgressiveResult(
  event: ProgressiveResultEvent,
): string | null {
  if (event.final || typeof event.text !== 'string' || !event.text.trim()) {
    return null;
  }
  let text = event.text.trim();
  const model = progressiveResultModel(event);
  const label = progressiveResultModelLabel(event);
  const prefixes = [model, 'Moondream', 'Gemini', 'GPT'].filter(Boolean) as string[];
  for (const prefix of prefixes) {
    const escaped = prefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    text = text.replace(new RegExp(`^${escaped}\\s*:\\s*`, 'i'), '').trim();
  }
  return label ? `${label}: ${text}` : text;
}
