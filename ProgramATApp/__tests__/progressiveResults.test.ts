import {
  appendProgressiveResult,
  acceptsProgressiveResult,
  acceptsStreamingProgressEvent,
  formatProgressiveResult,
  progressiveInvocationIsRunning,
  progressiveResultModel,
} from '../progressiveResults';

describe('progressive result lifecycle', () => {
  it('accepts repeated increasing results for the active invocation', () => {
    expect(acceptsProgressiveResult('active', 0, {
      invocation_id: 'active', result_index: 1,
    })).toBe(true);
    expect(acceptsProgressiveResult('active', 1, {
      invocation_id: 'active', result_index: 2,
    })).toBe(true);
  });

  it('rejects obsolete invocations and duplicate result indexes', () => {
    expect(acceptsProgressiveResult('new', 0, {
      invocation_id: 'old', result_index: 1,
    })).toBe(false);
    expect(acceptsProgressiveResult('active', 2, {
      invocation_id: 'active', result_index: 2,
    })).toBe(false);
  });

  it('rejects late streaming progress after the stream has stopped', () => {
    expect(acceptsStreamingProgressEvent(false, {mode: 'streaming'})).toBe(false);
    expect(acceptsStreamingProgressEvent(true, {mode: 'streaming'})).toBe(true);
    expect(acceptsStreamingProgressEvent(false, {mode: 'one_shot'})).toBe(true);
  });

  it('remains running for partials and stops on the final event', () => {
    expect(progressiveInvocationIsRunning({final: false})).toBe(true);
    expect(progressiveInvocationIsRunning({final: true})).toBe(false);
  });

  it('keeps model metadata from either transport shape', () => {
    expect(progressiveResultModel({model: 'moondream'})).toBe('moondream');
    expect(progressiveResultModel({
      metadata: {model: 'gemini/gemini-3.1-flash-lite-preview'},
    })).toBe('gemini/gemini-3.1-flash-lite-preview');
  });

  it('accumulates results and leaves them intact on the final marker', () => {
    let displayed: string[] = [];
    displayed = appendProgressiveResult(displayed, {
      model: 'moondream',
      text: 'First answer',
      final: false,
    });
    displayed = appendProgressiveResult(displayed, {
      metadata: {model: 'gemini'},
      text: 'Second answer',
      final: false,
    });
    const beforeFinal = displayed;
    displayed = appendProgressiveResult(displayed, {final: true, text: ''});

    expect(displayed).toEqual([
      'moondream: First answer',
      'gemini: Second answer',
    ]);
    expect(displayed).toBe(beforeFinal);
    expect(formatProgressiveResult({final: true, text: ''})).toBeNull();
  });
});
