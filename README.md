# ProgramAT

This repository contains code for an AI-powered assistive technology platform, ProgramAT. ProgramAT equips blind or low-vision (BLV) users to create custom camera-based assistive technologies via natural language instructions. BLV users can test their camera-based ATs with input from their iPhone's camera. The system has 3 major components:
- a mobile app, [installable via TestFlight](https://testflight.apple.com/join/ahpWstQr).
- a server that runs the computation for the camera-based AT you build using the app. This interfaces with a GitHub repository, and runs the AI models necessary for your task. 
- ProgramAT GitHub repository. This is where your tools will live. It will be a fork of this repository for each individual user. We recommend forking closer to the event date, or keeping an eye on when your fork is behind the parent repository in order to sync.


## What is the ProgramAT Mobile App?

This is a React Native app that facilitates AT creation, iteration, and testing of the created AT using your camera feed. You can also monitor the status of their creation. The app streams frames to a Python backend on your server, which executes pluggable tools — object detection, OCR, scene description, and more and returns spoken feedback.

## Getting Started

### Prerequisites

- A [GitHub account](https://github.com/). Refer to [screen reader friendly instructions by Jeff Bishop](https://community-access.org/git-going-with-github/docs/00-pre-workshop-setup.html).
- A [Copilot subscription](https://github.com/features/copilot/plans).
- Github mobile, installed and logged in
- Testflight, installed from Apple's app store
- A computer with decent compute capacity, or ability to host a VM with decent compute capacity. For best performance, we recommend around 20G of available disk space wherever you plan to host. A GPU is not required.
- A screen reader and an accessible web browser.
- iPhone 12 or higher running iOS 26. Apple Intelligence or AI-specific features for processors are not required.
- Python 3.11+ (for the backend server)
- Node.js >= 20
- Optional: React Native CLI development environment ([setup guide](https://reactnative.dev/docs/set-up-your-environment)). This is required if you want to run and install the app. You can skip this requirement if you are installing the app using our TestFlight link.
- Optional: For iOS: Xcode and CocoaPods. You can skip if you are not building and running the app itself locally.

### Installation

1. **Fork this repository**
   This is easiest to do from the Github website.
   First, select the button that says Fork.
   Once you have done so, it will take you to an interface where you can select the owner and name of the fork. By default, this is your username for owner, and `ProgramAT-opensource` for the repository name. We recommend leaving these defaults intact, but you can change them if you would like.
   Then, click the button that says Create Fork.
   After a few seconds, a copy of the repository will be made in your Github account.

2. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/ProgramAT-opensource.git
   cd ProgramAT-opensource
   ```

3. **Get the API keys**
   See the API keys section for a detailed description.

### Running the Mobile App Locally (Optional)

You only need this section if you want to make changes to the frontend and run the React Native app yourself. If you are using the TestFlight version of the app, you can skip this section.

This section requires an iPhone and XCode on Mac system.

1. **Install React Native dependencies**

   ```bash
   cd ProgramATApp
   npm install
   ```

2. **Install iOS dependencies (iOS only)**

   ```bash
   cd ios
   pod install
   cd ..
   ```

3. **Start the React Native development server**

   ```bash
   cd ProgramATApp
   npx react-native start
   ```

   This starts the React Native Metro development server.

4. **Connect your phone**
   Plug your iPhone into your computer while running the development server.

   Depending on your setup, you may also need to:
   - Trust the computer on your iPhone.
   - Turn on Developer Mode on your iPhone.
   - Open the app through Xcode or the React Native development workflow.

Notes:

- If you are using the TestFlight version of the app, you do not need to run the React Native development server.
- The backend server and the React Native development server are separate processes and must be started in different terminals or windows.
- The backend handles tool execution and AI processing. The React Native development server only serves the mobile app frontend during local development.

### API Keys

The backend reads configuration from `backend/.env` automatically when it starts, or from environment variables if you set them directly.

Use the table below as a quick reference for what each value does.

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_MODEL` | Optional | Model name used by LiteLLM. Update this in `backend/.env` to change which provider/model is active (falls back to `GEMINI_MODEL` if present). |
| `GEMINI_API_KEY` | Optional | Provider API key for Google Gemini. Keep provider keys in `.env` as needed: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. |
| `GOOGLE_APPLICATION_CREDENTIALS` | For OCR tools | Google Cloud Vision API credentials used by Live OCR |
| `GITHUB_TOKEN` | For GitHub features | GitHub personal access token with `repo` scope |
| `GITHUB_REPO` | Yes (to access your own tools) | Target repo in `owner/repo` format |
| `HOST` / `PORT` | Optional | Server bind address (default `0.0.0.0:8080`) |

### How to Get the Keys

The following keys are required for the application to run, but you can add the key of other providers and use the corresponding model for text parsing and image analysis.

#### GitHub Token

1. Go to any GitHub page: <https://github.com/settings/tokens>.
2. Click **Tokens(classic)** and click **Generate new token**.
3. Add a note, choose an expiration, and select scopes to be **repo** and **workflow**.
4. Click **Generate token**, copy it and save it properly. You won't be able to see it again.
5. Paste the token into `backend/.env` as `GITHUB_TOKEN`.

#### Gemini API Key

1. Go to Google AI Studio: <https://aistudio.google.com/app/apikey>
2. Sign in with your personal Google account.
3. Click **Create API key**.
4. Select an existing Google Cloud project or create a new one, then click **Create API key**.
5. Copy the generated key.
6. Paste the key into `backend/.env` as `GEMINI_API_KEY`.

Billing:

- If you want to increase the rate limit, click **Set up billing** in Google AI Studio and choose other plans for your API key.

#### Google Application Credentials

This project uses Google Cloud Vision for higher-quality OCR in `tools/live_ocr.py`. If you want Cloud Vision for better OCR, create a service account and download a JSON key, then set `GOOGLE_APPLICATION_CREDENTIALS` to that file path. If you prefer not to use Cloud Vision, the tool will fall back to local Tesseract if installed (no Google credentials required).

Steps:

1. Go to Google Cloud Console: <https://console.cloud.google.com>
2. Create or select a project and enable the **Cloud Vision API** in **Menu** -> **API & Services** -> **Enabled APIs & services**.
3. Create a service account: **Menu** -> **IAM & Admin** -> **Service Accounts** -> **Create service account**. Fill name/ID and click **Continue**.
4. Create a key: for this service account, select **Actions** -> **Manage Keys** -> **Add key** -> **Create new key** -> choose **JSON** -> **Create**. 
5. A JSON file will be downloaded. Copy the JSON file from the Downloads folder to the backend folder in the codespace. Rename the file to be `credentials.json`.
6. Set `GOOGLE_APPLICATION_CREDENTIALS` to the JSON file path.

Security notes:

- Do not commit the JSON key to version control. Add `credentials.json` to `.gitignore`.
- Restrict the service account to only the Vision API and grant the minimal needed permissions.

### Running the Application

You can host the server from your personal machine for free, or use a hosting service like a GCP virtual machine or AWS virtual machine, which may have associated costs. 

The difference between these two options, aside from cost, is that when running from your personal machine, your server will shut off whenever your machine does. This is potentially avoidable by using a paid hosting service.

We provide instructions here for hosting from your personal machine. If you would prefer to use a paid hosting service, follow the instructions there for setup.

   > Notice: You can skip steps 1-2 if ngrok is already installed and configured in your terminal, or if you are using a paid hosting service.

1. **Install ngrok**
   Download ngrok for your system from <https://ngrok.com/download> and make sure `ngrok version` works in your terminal.

2. **Connect ngrok to your account**
   Follow the instructions on the downloading page to sign up for an account and get your authtoken in the ngrok dashboard. Configure it in your terminal.

   ```bash
   ngrok config add-authtoken YOUR_NGROK_AUTHTOKEN
   ```

   > Notice: You can skip steps 3-8 by filling the API keys into the prompt in `COPILOT_SETUP_PROMPT.md` and giving it to Copilot (or your coding agent) and let it set up the server and use ngrok for you. After it gives you the forwarding address, you can go to step 9 and type it in the mobile application.

3. **Create the backend `.env` file**

   ```bash
   cp .env.example .env
   ```

4. **Fill in the values in `backend/.env`**
   - `LLM_MODEL`: model name used by LiteLLM (for example `gemini-3-flash-preview` or `openai/gpt-4o`). Change this to switch provider/model without modifying code.
   - Provider API keys: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
   - `GITHUB_TOKEN`
   - `GITHUB_REPO`
   - `GOOGLE_APPLICATION_CREDENTIALS`

5. **Set up the backend**

   ```bash
   cd ../backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

6. **Activate the virtual environment and start the backend server**

   ```bash
   cd backend
   source .venv/bin/activate
   python stream_server.py
   ```

   The server listens on `0.0.0.0:8080` by default.

7. **Open up another terminal or window and start the ngrok tunnel**

   ```bash
   ngrok http 8080
   ```

8. **Copy your forwarding address**
   If you are using ngrok, keep the terminal created in step 4 open. ngrok will print a forwarding address, which is the public address your app should connect to. Copy this address.

   If you are using a paid hosting service, your VM should list a public IP address, copy this IP address.

9. **Paste the forwarding address into the app**
   Change the prefix of the forwarding address from **https** to **wss** because we are using a websocket. Open ProgramAT app on your mobile device, go to the server address field in Settings, paste the ngrok forwarding address, and tap **Connect**.

## Troubleshooting

### Installing Dependencies

1. If you are using Windows Powershell and encounter the problem "This error might have occurred since this system does not have Windows Long Path support enabled" when installing LiteLLM, you need to bypass the path length limit. Open PowerShell as Administrator and run:

```bash
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Then install the dependencies again.

### Forked Repository Settings

Since you are working from a fork of this repository, GitHub-related features may need a few repo-level settings that are not always copied over from the upstream project.

1. Make sure **Issues** are enabled in **Settings** -> **General** -> **Features**.
2. Make sure the workflow **Auto-assign to Copilot** is visible in **Actions**. If it is missing, delete and re-upload [`.github/workflows/copilot-assignment.yml`](.github/workflows/copilot-assignment.yml).
3. Go to **Settings** -> **Secrets and variables** -> **Actions**, click **New repository secret**, create a secret named `COPILOT_PAT`, and set its value to your GitHub personal access token.
4. If you use `gh auth login`, make sure that token has the required scopes, including `repo`, `workflow`, and `admin:org`.

## Usage

1.  **Select a Tool** — Navigate to the **Tools** tab, browse the available tools, and tap one to select it.

2. **Run** — The Tool Runner opens with a live camera preview. Tap **Run** to execute the tool on single frames, or **Stream** to process frames continuously. Results are spoken aloud via TTS.

3. **Chat** — After a tool run, tap **Chat** to ask follow-up questions about the result (powered by LiteLLM; change the active model with `LLM_MODEL` in `backend/.env`).

4. **Development mode** — Use the **PRs** tab to browse open pull requests, select one to load its tools, and send text updates to GitHub issues.

5. **Tool creation** — To instead create a new tool, from development mode, select the "Create New Issue Instead" button in the PRs tab. Then, type or dictate the tool you would like to make into the text box, then submit. If more information is needed, the app will ask for it: in this case, dictate or type an answer to the request and resubmit, it will be appended to your initial request. Once the request is complete, the app will tell you it has made a new issue successfully. Copilot will automatically be assigned and create a relevant pull request. From there, wait for it to generate, and then run it as described in the earlier usage steps!
   * If you prefer to do this from desktop, you can also fill out the issue template titled "Visual Assistive Technology" on the Github website
  
6. **Tool iteration** — To update or modify a tool, in development mode, click a relevant PR, and instead of choosing open tools as you would to run a tool, select update issue. Then, type or dictate the change you would like to make and submit. Copilot will automatically be assigned and make the desired changes to the relevant tool.
   * If you would prefer to do this from desktop, you can also simply leave a comment on the relevant pull request from the Github website. In this case, you will be responsible for appending `@copilot` to your comment to assign Copilot.

 
## Supported Input Modes

- **Streaming mode** — Tools process frames continuously and return real-time audio feedback
- **Single-frame mode** — Capture one frame and get a detailed result
- **Real-time camera streaming** at configurable FPS via `react-native-vision-camera`
- **Conversation mode** — Ask follow-up questions about tool results via a Chat tab. This becomes available after a single-frame mode interaction.
- **Custom GPT-like tools** — Tools flagged as Custom GPT use Gemini Live for streaming multimodal conversations instead of executing code per frame

## Supported Feedback Modes

- **Text-to-Speech feedback** — all tool results are spoken aloud automatically
- **Rich audio output** — tools can return speech, beeps, haptic vibration, earcons, and more via the AudioOutputService
- **Speech-to-Text input** — voice input for follow-up questions using `@react-native-voice/voice` and OS-level dictation

## Usage Modes

### Development Mode (GitHub Integration)

- **PR browser** — List open pull requests, select one, and load its tools
- **Text input for issues** — Create or update GitHub issues with AI-powered parsing (LiteLLM)
- **Multi-turn conversations** — The server asks for missing fields until the issue is complete
- **Copilot session logs** — View AI coding session summaries per PR

### Production Mode

- Tools are pulled from the `main` branch only
- The PR browser tab is hidden; users go straight to the tool list

### Review mode
- Tools are pulled from the main repository server (as opposed to your self-hosted server), specifically PRs people have put up for review
- You can test tools as you would in development mode, but cannot directly create or edit tools.
   - If you would like a tool of yours to be visible in review mode, create it locally, then submit a PR!
- You can review tools (approve if they work as intended, request changes otherwise)
   - Note that for this mode to work properly, you should first [become a contributor](https://github.com/program-at/ProgramAT-opensource/issues/44). These permissions are needed to leave reviews.

## Built-In Tools

| Tool | Description | Model |
|------|-------------|-------|
| **Object Recognition** | Detects and announces objects using YOLO11 + COCO | YOLOv11 |
| **Live OCR** | Reads visible text aloud in real time | Google Cloud Vision API |
| **Scene Description** | Generates a spoken description of the scene | LiteLLM Vision |
| **Camera Aiming** | Guides users to center an object for a well-framed photo | YOLOv11 |
| **Door Detection** | Detects doors/doorways with clock-face navigation cues | YOLOWorld |
| **Empty Seat Detection** | Finds unoccupied chairs and gives directional guidance | YOLOv11 |
| **Clothing Recognition** | Identifies the most prominent clothing item and its features | LiteLLM Vision |

New tools can be added by placing a Python file in the `tools/` directory. Each tool exposes a `main(image, input_data)` function and returns an audio-friendly string or dict.

## Example tool ideas!
You can make a wide variety of tools using ProgramAT! For best performance, we recommend thinking about tools that are camera based, and, for the time being, are stateless (do not need to remember their previous responses/backreference previous frames) and do not require bringing in an outside dataset.

> Want to bring in stateless capabilities? Or ability to use outside datasets? Implement these features and [become a contributor](https://github.com/program-at/ProgramAT-opensource/issues/44) to share them with the community!

To get your imagination started, here are some ideas for tools people have tried before!

- Plug point detector [(see it run!)](https://youtube.com/shorts/AXy-D1c47Bk?feature=share): Help find plug points/outlets in unfamiliar rooms.
- Playing card reader [(see it run!)](https://youtube.com/shorts/okevY0bmcr8?feature=share): Describe a hand of playing cards, and how they change over the course of a card game
- Uber finder [(see it run!)](https://youtube.com/shorts/wspprXIsd3Y?feature=share): Help identify an Uber by describing the colors, makes, and models of car in frame.
- Clothing description: Get a simple description of an article or articles of clothing
- Mail sorter: Describe what important vs. junk mail means to you and get guidance on whether you should open a particular piece of mail.
- Makeup checker: Take a photo and analyze if there are any issues with your makeup application.
- Sock matcher: Determine if two socks out of the laundry match


## Project Structure

```
ProgramAT-opensource/
├── ProgramATApp/              # React Native mobile app
│   ├── App.tsx               # Root component, WebSocket setup, state management
│   ├── TabNavigator.tsx      # Bottom tabs: PRs, Tools, Chat, Settings
│   ├── ToolSelector.tsx      # Lists available tools
│   ├── ToolRunner.tsx        # Runs selected tool against camera feed
│   ├── CameraView.tsx        # Camera capture & frame streaming
│   ├── Chat.tsx              # Follow-up conversation interface
│   ├── PRsAndText.tsx        # PR browser & text input (dev mode)
│   ├── Settings.tsx          # Mode switching, server selection, theme
│   ├── WebSocketService.ts   # WebSocket connection manager
│   ├── AudioOutputService.ts # Multi-modal audio output (TTS, beeps, haptics)
│   ├── TextToSpeechService.ts
│   ├── BeepService.ts
│   ├── config.ts             # Server URLs, feature flags, mode config
│   └── __tests__/            # Jest test suite
├── backend/                   # Python WebSocket server
│   ├── stream_server.py      # Main server — message routing, tool execution, GitHub integration
│   ├── gemini_live.py        # Gemini Live API session manager for Custom GPT tools
│   ├── gemini_summarizer.py  # Summarizes Copilot session logs
│   ├── copilot_db.py         # SQLite storage for Copilot session data
│   ├── module_manager.py     # Auto-installs missing Python packages at runtime
│   └── requirements.txt      # Python dependencies
└── tools/                     # Pluggable vision/AI tools (Python)
    ├── object_recognition.py
    ├── live_ocr.py
    ├── scene_description.py
    ├── camera_aiming.py
    ├── door_detection.py
    ├── empty_seat_detection.py
    └── clothing_recognition.py
```

## Development

### Running Tests

```bash
cd ProgramATApp
npm test
```

### Adding a New Tool

1. Create a Python file in `tools/` (e.g., `tools/my_tool.py`).
2. Implement a `main(image, input_data)` function that returns an audio-friendly string or dict.
3. The tool is automatically discovered and available in the app when loaded from the server.

See [tools/MODEL_SETUP.md](tools/MODEL_SETUP.md) and the existing tools for examples.

### Building for Production

For iOS:

1. Open `ProgramATApp/ios/ProgramATApp.xcworkspace` in Xcode
2. Select your signing team and provisioning profile
3. Build for Release

## Technology Stack

### Mobile App

- **React Native 0.82.1** — Bare (non-Expo) for direct native module access
- **TypeScript**
- **react-native-vision-camera** — Camera capture and frame streaming
- **@react-native-voice/voice** — Speech-to-text for follow-up questions
- **react-native-tts** — Text-to-speech for tool results
- **react-native-audio-api** — Programmatic audio generation (beeps, tones)
- **AsyncStorage** — Local persistence for settings and sessions

### Backend

- **Python 3.11** with async `websockets`
- **LiteLLM (configurable providers)** — Model runtime used for AI parsing, scene description, and clothing recognition. LiteLLM can route requests to provider backends such as Google Gemini, OpenAI, or Anthropic; switch the active model via `LLM_MODEL` in `backend/.env`.
- **Google Cloud Vision API** — OCR
- **Ultralytics (YOLOv11 / YOLOWorld)** — Object detection
- **OpenCV / NumPy / Pillow** — Image processing
- **PyGithub** — GitHub issue and PR integration

## Get involved!!

As an open source project, we welcome community feedback and contributions.

If you are interested in going beyond your fork and contributing to ProgramAT, please start by reading our [Code of Conduct](https://github.com/program-at/ProgramAT-opensource/blob/main/CODE_OF_CONDUCT.md) and [Contribution Guidelines](https://github.com/program-at/ProgramAT-opensource/blob/main/CONTRIBUTING.md).

Once you have done so, you can request the permissions needed to become a contributor by following [these instructions](https://github.com/program-at/ProgramAT-opensource/issues/44)!

## License

See [LICENSE](LICENSE).

## Contributions

- [Ellie Seehorn](https://seehorne.github.io/) (PhD student at University of Michigan)
- [Yushan Wei](https://github.com/yushanwe) (Undergraduate Student at University of Michigan)
- [Venkatesh Potluri](https://venkateshpotluri.me/) (Assistant Professor, University of Michigan. Principal Investigator of the [Intelligent Developer Experiences for Accessibility Lab](https://idea11y.dev/))
- [Anhong Guo](https://guoanhong.com/) (Assistant Professor, University of Michigan and principal investigator of the [Human AI Lab](https://guoanhong.com/))

Found a problem? Please file an issue!
