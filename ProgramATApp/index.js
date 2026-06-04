/**
 * @format
 */

import { AppRegistry, LogBox, Linking } from 'react-native';
import App from './App';
import { name as appName } from './app.json';

// Suppress Fast Refresh false-positive hook errors that block the screen
LogBox.ignoreLogs([
  'Rendered fewer hooks than expected',
  'Rendered more hooks than expected',
  'Warning: React has detected a change in the order of Hooks',
]);

console.log('RN LINKING DEBUG INSTALLED');

Linking.getInitialURL().then(url => {
  console.log('🚨 RN INITIAL URL =', url);
});

Linking.addEventListener('url', event => {
  console.log('🚨 RN URL EVENT =', event.url);
});

AppRegistry.registerComponent(appName, () => App);
