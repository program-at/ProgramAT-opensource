import {acceptsProgressiveResult} from '../progressiveResults';

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
});
