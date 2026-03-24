# Update Mode Testing Guide

## What We Discovered

The text "Update the spelling of the last name it should be spelled SEEHORN" was being interpreted as an **issue SELECTION command** instead of a **comment to add**, because:

1. The AI in `parse_issue_selection()` is trained to match phrases like "update the camera issue" to existing issues
2. Your text "Update the spelling of the last name on the website" matched issue #80 (personal website issue)
3. So it switched TO update mode but didn't add the comment

## The Core Issue

When the server restarts, the `selected_issue` global state is reset to:
```python
selected_issue = {
    'mode': 'create',
    'number': None,
    'title': None
}
```

So even though we added logic to skip parsing when already in update mode, the restart cleared that state.

## Current Flow (After Restart)

1. **Select issue** → sends "select issue 80" → switches to update mode ✅
2. **Speak comment** → text is received ✅
3. **After 5 seconds** → monitor triggers
4. **Problem**: We check if `mode == 'update'` FIRST, which should work!

But in your case, the old text from BEFORE the restart was processed, and at that time mode was 'create', so it parsed it as a selection command.

## Correct Testing Procedure

**IMPORTANT: You must select the issue AFTER each server restart!**

### Step 1: Connect and Select
1. Connect to server (restart clears state)
2. Tap "Browse Issues"
3. Select issue #80
4. Server logs should show:
   ```
   INFO - Issue selection parsed: mode=update, issue=80
   INFO - Switched to update mode for issue #80
   ```

### Step 2: Speak Your Comment
1. Tap record button
2. Say: "This is a test comment to verify the update mode is working"
3. Stop recording
4. Text log should show: `2025-12-02T04:XX:XX - This is a test comment to verify the update mode is working`

### Step 3: Wait for Processing
1. **Wait at least 5 seconds** without speaking
2. Server logs should show:
   ```
   INFO - Pause detected (5.Xs), calling create_github_issue with content: This is a test...
   INFO - create_github_issue called with text: This is a test...
   INFO - Current mode: update, issue: 80  <-- KEY: Should be 'update' not 'create'
   INFO - Already in update mode for issue #80, adding comment
   INFO - Assigned GitHub Copilot to issue #80
   INFO - Added comment to issue #80
   ```

### Step 4: Verify on GitHub
1. Go to https://github.com/idea11y/ProgramAT/issues/80
2. Should see your comment
3. Should see Copilot assigned to the issue

## If It Still Doesn't Work

Check the server log for "Current mode:" - if it says 'create', then the state was lost somehow.

Possible causes:
1. Server restarted between selection and speaking
2. Client disconnected and reconnected (WebSocket reconnection)
3. Some code path is resetting the `selected_issue` dict

## Server Status

✅ Server running at 04:22:34
✅ Latest code with enhanced logging
✅ Ready to test - try selecting an issue NOW and speaking a comment!

## Pro Tip

To avoid confusion, use simple, clear comments when testing:
- ✅ "This is a test comment"
- ✅ "Adding a note about the bug"
- ❌ "Update the spelling..." (sounds like a selection command)
