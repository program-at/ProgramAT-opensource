# Development and Production Modes

## Overview

ProgramAT now supports two operating modes to provide different feature sets for development and production environments:

- **Development Mode**: Full feature set for testing and development
- **Production Mode**: Streamlined interface for end users

## Mode Configuration

### config.ts

The app mode is configured in `ProgramATApp/config.ts`:

```typescript
export type AppMode = 'development' | 'production';

export const Config = {
  // Application Mode
  APP_MODE: 'development' as AppMode,  // Change to 'production' for deployment
  
  // GitHub branch for production mode
  PRODUCTION_BRANCH: 'main',
  
  // Feature flags based on mode
  get ENABLE_ISSUES_TAB() {
    return this.APP_MODE === 'development';
  },
  get ENABLE_PR_SELECTION() {
    return this.APP_MODE === 'development';
  },
  get ENABLE_BRANCH_SELECTION() {
    return this.APP_MODE === 'development';
  },
  get ENABLE_MODE_SWITCHER() {
    return true;  // Allow switching between modes in app
  },
};
```

## Features by Mode

### Development Mode (default)

**UI Features:**
- ✅ Text Input tab
- ✅ Issues tab (for GitHub issue management)
- ✅ Tools tab
- ✅ Tool Runner tab  
- ✅ PR selection dropdown
- ✅ Branch selection
- ✅ Mode switcher button (🔧 Dev)

**Tool Loading:**
- Can load tools from any PR or branch
- Supports issue-specific tools
- Full GitHub integration

**Use Cases:**
- Testing new tools
- Debugging tool functionality
- Working on specific GitHub issues
- Iterating on tool development

### Production Mode

**UI Features:**
- ✅ Text Input tab
- ❌ Issues tab (hidden)
- ✅ Tools tab
- ✅ Tool Runner tab
- ❌ PR selection (disabled)
- ❌ Branch selection (disabled)
- ✅ Mode switcher button (🚀 Prod)

**Tool Loading:**
- Only loads tools from `main` branch
- No issue-specific tools
- Production-ready tools only

**Use Cases:**
- End-user deployment
- Stable production environment
- Simplified interface for non-developers

## Mode Switcher

The mode switcher appears as a button in the top-right corner of the app:

- **🔧 Dev** (green) = Development mode active
- **🚀 Prod** (orange) = Production mode active

### Switching Modes

1. Tap the mode button in the top-right
2. Confirm the mode switch in the alert dialog
3. The app will:
   - Update the UI to show/hide features
   - Redirect from Issues tab if switching to production
   - Update tool loading behavior

### Mode Switch Dialog

The dialog shows what will change:

**Switching to Production:**
```
• Issues tab will be hidden
• Tools will only load from main branch
• PR selection disabled
```

**Switching to Development:**
```
• Issues tab will be shown
• Tools can load from any PR/branch
• Full development features enabled
```

## Implementation Details

### TabNavigator.tsx

**State Management:**
```typescript
const [appMode, setAppMode] = useState<AppMode>(Config.APP_MODE);
```

**Conditional Tab Rendering:**
```typescript
{/* Issues tab - only show in development mode */}
{appMode === 'development' && (
  <TouchableOpacity ...>
    <Text>📋</Text>
    <Text>Issues</Text>
  </TouchableOpacity>
)}
```

**Mode Switch Handler:**
```typescript
const handleModeSwitch = () => {
  const newMode: AppMode = appMode === 'development' ? 'production' : 'development';
  
  Alert.alert('Switch App Mode?', ...);
  
  setAppMode(newMode);
  Config.APP_MODE = newMode;
  
  // Redirect if on hidden tab
  if (newMode === 'production' && activeTab === 'issues') {
    setActiveTab('tools');
  }
};
```

**Fallback Content:**
```typescript
case 'issues':
  // Only show in dev mode, fallback to text input in production
  if (appMode !== 'development') {
    return <TextInput serverFeedback={serverFeedback} />;
  }
  return <IssueSelector ... />;
```

### ToolSelector.tsx (Future Enhancement)

Will be updated to respect production mode:

```typescript
// Pseudo-code for future implementation
if (Config.APP_MODE === 'production') {
  // Fetch tools from main branch only
  fetchTools({ branch: Config.PRODUCTION_BRANCH });
} else {
  // Allow PR/branch selection
  fetchTools({ pr: selectedPR, branch: selectedBranch });
}
```

## Styling

### Mode Button Styles

```typescript
modeContainer: {
  position: 'absolute',
  top: 10,
  right: 10,
  zIndex: 1000,
},
modeButton: {
  backgroundColor: '#4CAF50',  // Green for dev
  paddingHorizontal: 12,
  paddingVertical: 6,
  borderRadius: 16,
  elevation: 4,
},
productionMode: {
  backgroundColor: '#FF9800',  // Orange for production
},
```

## Testing

### Test Scenarios

1. **Mode Switch - Dev to Prod:**
   - Start in development mode
   - Tap mode switcher → confirm
   - Verify: Issues tab hidden
   - Verify: Mode button shows "🚀 Prod" (orange)

2. **Mode Switch - Prod to Dev:**
   - Start in production mode
   - Tap mode switcher → confirm
   - Verify: Issues tab visible
   - Verify: Mode button shows "🔧 Dev" (green)

3. **Tab Redirect on Mode Switch:**
   - Navigate to Issues tab (dev mode)
   - Switch to production mode
   - Verify: Automatically redirected to Tools tab

4. **Content Fallback:**
   - Navigate to Issues tab URL while in production mode
   - Verify: Text Input component shown instead

5. **Tool Loading (Future):**
   - Production mode: Only main branch tools load
   - Development mode: All PRs/branches available

## Deployment

### Setting Production Mode as Default

To deploy in production mode by default:

1. Edit `ProgramATApp/config.ts`:
   ```typescript
   APP_MODE: 'production' as AppMode,
   ```

2. Optionally disable the mode switcher:
   ```typescript
   get ENABLE_MODE_SWITCHER() {
     return false;  // No switching in production
   },
   ```

3. Rebuild the app:
   ```bash
   cd ProgramATApp
   npm run build
   ```

## Future Enhancements

### Planned Features

1. **Backend Mode Detection:**
   - Send mode info in WebSocket messages
   - Server filters tools by branch based on mode

2. **Persistent Mode Storage:**
   - Save mode preference to AsyncStorage
   - Remember user's choice across app restarts

3. **Mode-Specific Analytics:**
   - Track which tools are used in each mode
   - Monitor mode switch frequency

4. **Additional Production Restrictions:**
   - Disable experimental tools in production
   - Hide debug/verbose logging options
   - Simplified error messages

5. **Admin Mode:**
   - Third mode for admin users
   - Additional debugging tools
   - Access to all features plus admin-specific ones

## Troubleshooting

### Issues Tab Still Visible in Production Mode

1. Check `config.ts`:
   ```typescript
   console.log('Current mode:', Config.APP_MODE);
   console.log('Issues tab enabled:', Config.ENABLE_ISSUES_TAB);
   ```

2. Verify state update:
   ```typescript
   // In TabNavigator
   console.log('appMode state:', appMode);
   ```

3. Force refresh/reload the app

### Mode Switcher Not Appearing

1. Check config:
   ```typescript
   get ENABLE_MODE_SWITCHER() {
     return true;  // Must be true
   },
   ```

2. Check z-index of modeContainer (should be 1000)

3. Verify no other components overlapping

### Tools Not Loading from Main Branch

This feature is not yet implemented. Currently only the UI changes between modes. Tool loading logic needs to be added to ToolSelector.

## Related Files

- `ProgramATApp/config.ts` - Mode configuration and feature flags
- `ProgramATApp/TabNavigator.tsx` - Mode switcher UI and conditional tabs
- `ProgramATApp/ToolSelector.tsx` - Will handle mode-specific tool loading
- This file: `DEV_PRODUCTION_MODES.md` - Documentation

## Summary

The dev/production mode system provides:

✅ **Implemented:**
- Mode configuration with feature flags
- Conditional UI rendering (Issues tab)
- Mode switcher button
- Automatic tab redirection
- Visual indicators (dev = green, prod = orange)

🔄 **In Progress:**
- Production mode tool filtering (main branch only)
- Mode persistence across app restarts

❌ **Planned:**
- Backend mode detection
- Admin mode
- Mode-specific analytics
