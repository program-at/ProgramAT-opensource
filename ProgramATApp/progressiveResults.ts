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

export function formatProgressiveResult(
  event: ProgressiveResultEvent,
): string | null {
  if (event.final || typeof event.text !== 'string' || !event.text.trim()) {
    return null;
  }
  const text = event.text.trim();
  const model = progressiveResultModel(event);
  return model ? `${model}: ${text}` : text;
}

export function appendProgressiveResult(
  existing: string[],
  event: ProgressiveResultEvent,
): string[] {
  const formatted = formatProgressiveResult(event);
  return formatted ? [...existing, formatted] : existing;
}
