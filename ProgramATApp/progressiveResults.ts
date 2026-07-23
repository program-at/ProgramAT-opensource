export interface ProgressiveResultEvent {
  invocation_id?: string;
  result_index?: number;
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
