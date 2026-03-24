# GitHub Issue Iteration Feature

## Overview
The ProgramAT app now supports iterating on existing GitHub issues through voice commands. Users can select an open issue and add comments/updates to it using voice, in addition to creating new issues.

## Features

### 1. **Issue Selection**
- Browse open issues from the GitHub repository
- Select an issue to update via:
  - Manual selection from issue list UI
  - Voice commands (e.g., "select issue 42", "work on the login bug")

### 2. **Issue Modes**
The app operates in two modes:

#### Create Mode (Default)
- New issues are created from voice transcripts
- Follows the existing multi-turn conversation flow
- Prompts for missing fields one at a time

#### Update Mode
- Selected issue is displayed in the issue mode bar
- Voice transcripts are added as comments to the selected issue
- No field validation - comments are added directly

### 3. **Voice Commands**

#### List Issues
- "list issues"
- "show issues"
- "what issues are open"

**Result**: Displays a list of up to 10 most recently updated open issues

#### Select Issue
- "select issue [number]" (e.g., "select issue 42")
- "work on issue [number]"
- "update the [description] issue" (e.g., "update the camera bug")

**Result**: Switches to update mode for the specified issue

#### Create New Issue
- "create new issue"
- "new issue"

**Result**: Switches back to create mode

### 4. **UI Components**

#### Issue Mode Bar
Located below the connection status bar:
- **Create Mode**: Shows "✨ Create New Issue"
- **Update Mode**: Shows "📝 Updating: #[number] - [title]"
- Buttons:
  - **Select**: Opens issue selector modal
  - **New**: (only in update mode) Switches back to create mode

#### Issue Selector Modal
- Displays up to 50 open issues
- Shows for each issue:
  - Issue number
  - Title (truncated to 2 lines)
  - Labels (bug/enhancement)
  - Last updated date
- Tap an issue to select it
- "Create New Issue Instead" button at bottom
- "✕" close button in header

## Backend Implementation

### New Functions

#### `fetch_open_issues()`
- Fetches open issues from GitHub
- Caches results for 5 minutes to reduce API calls
- Returns list of issue dictionaries with:
  - number
  - title
  - labels
  - created_at
  - updated_at

#### `parse_issue_selection(transcript, available_issues)`
- Uses AI to parse voice commands for issue selection
- Determines mode: 'create', 'update', or 'list'
- Matches issue descriptions to actual issues
- Returns confidence score

#### `update_github_issue(issue_number, comment_text)`
- Adds a comment to an existing issue
- Sends success notification to client
- Logs activity

### Modified Functions

#### `create_github_issue(text)` - Enhanced
1. First checks if transcript is an issue selection command
2. Handles 'list' mode by broadcasting issue list
3. Handles 'update' mode selection by storing selected issue
4. If in update mode, adds comment to selected issue
5. Otherwise, follows original create issue flow

### Global State

```python
# Issue iteration tracking
selected_issue = {
    'number': None,
    'title': None,
    'mode': 'create'  # 'create' or 'update'
}

issue_cache = {
    'issues': [],
    'last_fetch': None,
    'cache_duration': 300  # 5 minutes
}
```

### WebSocket Messages

#### Server → Client

**issue_list**
```json
{
  "type": "issue_list",
  "message": "Open issues:\nIssue 42: Login bug\n...",
  "issues": [
    {
      "number": 42,
      "title": "Login bug",
      "labels": ["bug"],
      "created_at": "2025-11-25T10:00:00",
      "updated_at": "2025-11-25T15:30:00"
    }
  ]
}
```

**issue_selected**
```json
{
  "type": "issue_selected",
  "message": "Selected issue #42: Login bug",
  "issue_number": 42,
  "issue_title": "Login bug"
}
```

**issue_updated**
```json
{
  "type": "issue_updated",
  "message": "Comment added to issue #42",
  "issue_number": 42,
  "issue_url": "https://github.com/owner/repo/issues/42"
}
```

#### Client → Server

**Text messages** are parsed for:
- Issue selection commands
- Issue list requests
- Regular issue creation
- Comments (when in update mode)

## Frontend Implementation

### New Components

#### `IssueSelector.tsx`
- Modal component for browsing and selecting issues
- Features:
  - Scrollable list of issues
  - Visual labels for bug/enhancement
  - Tap to select
  - Loading state
  - Empty state with retry
  - Full VoiceOver accessibility

### Modified Components

#### `App.tsx` - Enhanced
- Added issue mode state management
- Issue mode bar UI
- Select/New buttons
- WebSocket message handlers for:
  - `issue_selected`
  - `issue_created`
  - `issue_updated`
- Integration with IssueSelector modal

## User Workflows

### Workflow 1: Select and Update Issue
1. User says "list issues" or taps "Select" button
2. Issue list appears
3. User says "select issue 42" or taps issue in list
4. Issue mode bar updates to show selected issue
5. User speaks updates
6. After pause, comment is added to the issue
7. Success notification appears

### Workflow 2: Browse Issues Manually
1. User taps "Select" button
2. Issue selector modal opens
3. User browses list
4. User taps an issue
5. Modal closes
6. Issue mode bar updates
7. User can now speak updates

### Workflow 3: Switch Back to Create Mode
1. User is in update mode
2. User says "create new issue" or taps "New" button
3. Issue mode switches to create
4. User can now create new issues

## Accessibility

All new components support VoiceOver:
- Issue mode bar announces current mode
- Select/New buttons have labels and hints
- Issue selector modal is fully navigable
- Each issue card announces number, title, and labels
- Close and retry buttons are accessible

## Configuration

No new environment variables required. Uses existing:
- `GITHUB_TOKEN`: For GitHub API access
- `GITHUB_REPO`: Repository to fetch issues from
- `GEMINI_API_KEY`: For AI-powered issue selection parsing

## Limitations

1. **Cache Duration**: Issue list is cached for 5 minutes
   - May not show very recent issues
   - Manually close and reopen selector to refresh

2. **Issue Limit**: Only shows 50 most recent open issues
   - Very old issues may not appear
   - Consider closing old issues regularly

3. **Voice Selection**: AI parsing may not always match correctly
   - Use manual selection from UI for precision
   - Issue numbers are most reliable

4. **Update Comments**: No validation when adding comments
   - All text after pause is added as-is
   - No field prompting in update mode

## Future Enhancements

- Add ability to close/reopen issues via voice
- Support for editing issue title/description
- Add labels via voice
- Assign issues to users
- Filter issues by label or status
- Search issues by keyword
- Manual cache refresh button
- Support for pull requests

---

**Last Updated**: December 2, 2025
