I want you to fully set up and run this repository locally for me.

Important constraints:
- DO NOT try to acquire API keys or create accounts for me.
- The repository already contains backend/.env.example.
- You should create backend/.env from that template automatically.
- You should fill backend/.env using the values provided below.
- Do NOT ask me again for these values unless one is invalid or missing.

================ PROVIDED ENV VALUES ================

GEMINI_API_KEY=PASTE_GEMINI_KEY_HERE
OPENAI_API_KEY=PASTE_OPENAI_KEY_HERE (optional)
ANTHROPIC_API_KEY=PASTE_ANTHROPIC_KEY_HERE (optional)

GITHUB_TOKEN=PASTE_GITHUB_TOKEN_HERE
GITHUB_REPO=your-username/your-fork-name

NGROK_AUTHTOKEN=PASTE_NGROK_TOKEN_HERE

GOOGLE_APPLICATION_CREDENTIALS=backend/credentials.json

LLM_MODEL=gemini-3-flash-preview

=====================================================

Additional notes:
- If GOOGLE_APPLICATION_CREDENTIALS points to a missing file, tell me exactly where to place the credentials JSON.
- If some provider keys are unused, leave them in the .env file anyway.
- Never print secrets back into logs after writing them.

Your goals:
1. Read the README carefully.
2. Create backend/.env from backend/.env.example if needed.
3. Insert the provided values into backend/.env.
4. Preserve existing configuration when possible.
5. Create backend/.venv if it does not exist.
6. Install all backend dependencies.
7. Install ngrok if missing.
8. Configure ngrok using the provided NGROK_AUTHTOKEN.
9. Start the backend server.
10. Start ngrok forwarding on port 8080.
11. Automatically retrieve the ngrok forwarding URL.
12. Convert the forwarding URL from:
   https://...
   to:
   wss://...
13. Print the final websocket URL clearly for me to paste into the app.

Implementation requirements:
- Use Python 3.11 if required.
- Use repository-documented commands whenever possible.
- Prefer autonomous execution over asking me to run commands manually.
- If something fails:
  - inspect logs
  - diagnose the issue
  - attempt at least two fixes automatically before asking me for help

Important:
- Never overwrite secrets unnecessarily.
- Never expose secrets in output logs.
- Keep going until:
  - the backend server is running
  - ngrok forwarding works
  - the websocket URL is usable

Assume I prefer autonomous behavior unless external login/authentication is strictly required.