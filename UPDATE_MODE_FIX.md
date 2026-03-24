# Update Mode Fix

## Problem
When in update mode (after selecting an issue), speaking didn't create a comment on the issue. Nothing happened.

## Root Causes

### 1. Logic Order Issue
The `create_github_issue()` function was parsing every text input with `parse_issue_selection()` to check if it's a selection command. If you said something like "the camera isn't working", the AI would return `mode: 'create'`, which would reset the update mode and try to create a new issue instead of adding a comment.

**Old flow:**
1. Receive text
2. Parse text with AI to detect if it's a selection command
3. If AI returns `mode: 'create'`, reset to create mode
4. Check if we're in update mode (but it was just reset!)

### 2. Missing Features
The `update_github_issue()` function didn't:
- Assign GitHub Copilot to the issue
- Add comments to associated pull requests

## Solutions

### 1. Fixed Logic Order
**New flow:**
1. Receive text
2. **FIRST**: Check if we're already in update mode - if yes, immediately add comment and return
3. **SECOND**: Only if not in update mode, parse with AI to check for selection commands
4. Handle selection/creation as before

This prevents the AI from accidentally resetting update mode when you're speaking about the issue.

```python
# FIRST: Check if we're already in update mode
if selected_issue['mode'] == 'update' and selected_issue['number']:
    logger.info(f"Already in update mode for issue #{selected_issue['number']}, adding comment")
    await update_github_issue(selected_issue['number'], text.strip())
    return

# SECOND: Only parse for selection if not already in update mode
available_issues = fetch_open_issues()
selection = parse_issue_selection(text.strip(), available_issues)
# ... rest of logic
```

### 2. Enhanced update_github_issue()

Added three key features:

**A. Assign GitHub Copilot**
```python
try:
    copilot_user = g.get_user("copilot")
    current_assignees = [a.login for a in issue.assignees]
    if "copilot" not in current_assignees:
        issue.add_to_assignees(copilot_user)
        logger.info(f"Assigned GitHub Copilot to issue #{issue_number}")
except Exception as e:
    logger.warning(f"Could not assign Copilot: {e}")
```

**B. Find and Comment on Associated PR**
```python
# Search for PR that references this issue
pulls = repo.get_pulls(state='open')
for pr in pulls:
    if pr.body and f"#{issue_number}" in pr.body:
        pr_number = pr.number
        pr.create_issue_comment(comment_text)
        logger.info(f"Added comment to associated PR #{pr_number}")
        break
```

**C. Enhanced Response**
```python
success_data = {
    'type': 'issue_updated',
    'message': f"Comment added to issue #{issue_number}" + 
               (f" and PR #{pr_number}" if pr_number else ""),
    'issue_number': issue_number,
    'issue_url': issue.html_url,
    'pr_number': pr_number,
    'pr_url': pr_url
}
```

## Testing

### How to test update mode:

1. **Connect to server** from the app
2. **Select an issue:**
   - Tap "Browse Issues" button
   - Tap any issue in the list
   - Should see banner: "📝 Updating: #42 - Issue Title"
3. **Speak a comment:**
   - Tap record button
   - Say something like: "I think the camera frame rate should be 30 FPS instead of 60"
   - Stop recording
4. **Expected results:**
   - Server logs: "Already in update mode for issue #42, adding comment"
   - Comment appears on GitHub issue
   - GitHub Copilot is assigned to the issue
   - If there's a PR that references the issue, comment is added there too
   - App receives `issue_updated` message

### How to switch back to create mode:

- Tap "New Issue" button in the banner, OR
- Say "create new issue", OR  
- Tap "Create New Issue Instead" at bottom of issue selector

## Changed Files

1. **`/backend/stream_server.py`**
   - Modified `create_github_issue()` function (lines ~640-665)
   - Modified `update_github_issue()` function (lines ~252-315)

## Server Status

✅ Server restarted with new code
✅ Running on 34.144.178.116:8080
✅ Ready to test

## Known Limitations

1. **PR Detection**: Only finds PRs that explicitly reference the issue number in the PR body (e.g., "Fixes #42"). PRs without issue references won't receive comments.

2. **Copilot Assignment**: If the GitHub user "copilot" doesn't exist or can't be accessed, assignment will fail silently (logged as warning).

3. **Update Mode Persistence**: The `selected_issue` state is global in the server. If the server restarts, update mode is lost. Future enhancement could persist this in a database.
