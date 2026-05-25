/**
 * ModelPreference
 *
 * Single source of truth for the user's chosen LLM model.
 * Backed by AsyncStorage under '@llm_model'; loaded once on module import so
 * any code path (Settings UI, WebSocket sends) can read the current value
 * synchronously without juggling Promises.
 *
 * null = use the server's default model.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

const KEY = '@llm_model';
let cached: string | null = null;

// Fire-and-forget eager load. The race window (LLM send within the first
// few ms of app boot before AsyncStorage resolves) is invisible for users
// who haven't pinned a model — they get server default either way.
AsyncStorage.getItem(KEY)
  .then((v) => { cached = v; })
  .catch((e) => console.error('[ModelPreference] load failed', e));

export function getModelPreference(): string | null {
  return cached;
}

export async function setModelPreference(model: string | null): Promise<void> {
  const trimmed = model?.trim() || null;
  cached = trimmed;
  try {
    if (trimmed) {
      await AsyncStorage.setItem(KEY, trimmed);
    } else {
      await AsyncStorage.removeItem(KEY);
    }
  } catch (e) {
    console.error('[ModelPreference] save failed', e);
  }
}
