/**
 * @format
 */

import { AppRegistry, LogBox } from 'react-native';
import App from './App';
import { name as appName } from './app.json';

// Suppress Fast Refresh false-positive hook errors that block the screen
LogBox.ignoreLogs([
  'Rendered fewer hooks than expected',
  'Rendered more hooks than expected',
  'Warning: React has detected a change in the order of Hooks',
]);

AppRegistry.registerComponent(appName, () => App);
