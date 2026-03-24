# WebSocket Message Handler Fix

## Problem
The server was successfully sending 48 issues, but the UI displayed "No open issues found".

## Root Cause
**Multiple components were registering message handlers with `WebSocketService.onMessage()`:**

1. **App.tsx** - Registered a handler for `issue_selected`, `issue_created`, `issue_updated`
2. **IssueSelector.tsx** - Registered a handler for `issue_list`

Since `WebSocketService.onMessage()` overwrites the previous handler (it doesn't support multiple listeners), whichever component registered last would be the only one to receive messages. When App.tsx's useEffect ran after IssueSelector's, it overwrote the handler, so `issue_list` messages were never processed by IssueSelector.

## Solution
**Centralized message handling in App.tsx:**

1. **App.tsx** now handles ALL WebSocket message types in a single handler
2. Issue list data is stored in App.tsx state: `issueList`
3. The issue list is passed down to IssueSelector as a prop: `issues={issueList}`
4. IssueSelector no longer registers its own message handler
5. IssueSelector still requests the issue list but receives data via props

## Changes Made

### App.tsx
```typescript
// Added Issue interface and issueList state
interface Issue {
  number: number;
  title: string;
  labels: string[];
  created_at: string;
  updated_at: string;
}

const [issueList, setIssueList] = useState<Issue[]>([]);

// Updated message handler to handle ALL message types
const handleMessage = (message: any) => {
  console.log('[App] Received message:', message);
  
  if (message.type === 'issue_list') {
    // Handle issue list from server
    console.log('[App] Setting issue list:', message.issues?.length || 0, 'issues');
    setIssueList(message.issues || []);
  } else if (message.type === 'issue_selected') {
    // ... existing code
  } else if (message.type === 'issue_created' || message.type === 'issue_updated') {
    // ... existing code
  }
};

// Pass issues to IssueSelector
<IssueSelector 
  visible={showIssueSelector}
  onClose={() => setShowIssueSelector(false)}
  onIssueSelect={handleIssueSelect}
  issues={issueList}  // <-- New prop
/>
```

### IssueSelector.tsx
```typescript
// Updated interface to accept issues prop
interface IssueSelectorProps {
  visible: boolean;
  onClose: () => void;
  onIssueSelect: (issue: Issue) => void;
  issues: Issue[];  // <-- New prop
}

// Removed local issues state
// Removed WebSocketService.onMessage() handler
// Added useEffect to stop loading when issues arrive

useEffect(() => {
  if (issues.length > 0) {
    console.log('[IssueSelector] Received', issues.length, 'issues');
    setLoading(false);
    setError('');
  }
}, [issues]);
```

## Architecture Pattern
This follows the **Single Source of Truth** and **Props Down, Events Up** patterns:
- **App.tsx** = Single source of truth for WebSocket messages and issue data
- **IssueSelector.tsx** = Presentational component that receives data via props and sends events up via callbacks

## Testing
After this fix:
1. Open the Issue Selector modal
2. App.tsx receives the `issue_list` message from the server
3. App.tsx updates `issueList` state with 48 issues
4. IssueSelector receives the 48 issues via props
5. IssueSelector displays all 48 issues in the list

The "No open issues found" message should no longer appear.
