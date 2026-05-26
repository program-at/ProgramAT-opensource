# ProgramATApp/ — ProgramAT mobile app

React Native 0.82 (bare, non-Expo), TypeScript. The app streams phone-camera frames
to the backend over a WebSocket and speaks tool results back. It is built for blind
and low-vision users. All source files live in this directory — there is no `src/`.

## Commands

Run from this directory:

```bash
npm install          # Node >= 20
npm start            # Metro bundler — keep running in its own terminal
npm run ios          # build + launch on iOS
npm run android      # build + launch on Android
npm test             # Jest
npm run lint         # ESLint
npx tsc --noEmit     # typecheck (there is no npm script for this)
```

For iOS, run `cd ios && bundle exec pod install` before the first build and after
adding any native dependency. ESLint and Prettier are configured — run `npm run
lint` instead of formatting by hand. Verify changes with `npm test`, `npm run lint`,
and `npx tsc --noEmit`.

## Accessibility is the product

Users are blind and navigate with VoiceOver. Accessibility is a correctness
requirement. Any UI change MUST preserve:

- `accessibilityLabel` / `accessibilityRole` / `accessibilityHint` on interactive
  elements, and `accessible={false}` on decorative ones.
- `accessibilityLiveRegion="polite"` for dynamically updating content.
- Programmatic focus moves (`AccessibilityInfo.setAccessibilityFocus`) on screen
  changes.

Read `ACCESSIBILITY.md` before touching any UI.

## Where things live

- `App.tsx` — root component; owns the WebSocket lifecycle and message routing.
- `WebSocketService.ts` — singleton socket manager (a primary and a review socket).
- `config.ts` — app modes, server URLs, feature flags.
- `ToolRunner.tsx` / `CameraView.tsx` — run a tool against the live camera.
- `AudioOutputService.ts` / `TextToSpeechService.ts` / `BeepService.ts` — audio out.

`README.md` has the full file-by-file map.

## Gotchas

- **Native module changes need a full rebuild** (`npm run ios`/`android`), not just
  a Metro reload — this includes vision-camera, voice, tts, audio-api, fs.
- **App modes** (`development` / `production` / `review`) switch at runtime by
  mutating the `Config.APP_MODE` singleton; feature flags are getters off it. See
  `DEV_PRODUCTION_MODES.md`.
- **WebSocket:** there are two sockets. Use `getActiveSocket()`,
  `isActiveConnected()`, and `addMessageListener()` — don't touch `.ws` directly.
- The `LogBox.ignoreLogs` calls suppress known Fast-Refresh false positives — leave
  them.

Components are all functional with hooks; the only Context is `ThemeContext`;
styling is `StyleSheet.create`. Follow the existing patterns.
