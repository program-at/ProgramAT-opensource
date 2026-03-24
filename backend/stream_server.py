"""
WebSocket Server for Camera Frame Streaming
Runs on GCP VM instance to receive frames from mobile app

Dependencies:
pip install websockets opencv-python numpy pillow PyGithub google-generativeai python-dotenv (once in venv)

Usage:
python stream_server.py
"""

import asyncio
import websockets
import json
import base64
import logging
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from google.cloud import secretmanager
from github import Github
import os
from dotenv import load_dotenv
import google.generativeai as genai
from module_manager import get_module_manager
import copilot_db
from gemini_summarizer import summarize_entries_sync
from gemini_live import GeminiLiveManager
import re
import traceback
from concurrent.futures import ThreadPoolExecutor

# Load .env from the backend directory (if present)
load_dotenv(dotenv_path=str(Path(__file__).parent / '.env'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 8080 #port for listening
SAVE_FRAMES = True  # Set to True to save frames to disk
FRAMES_DIR = Path(__file__).parent / 'received_frames'

# Directory setup
FRAMES_DIR = Path("backend/received_frames")
FRAMES_DIR.mkdir(exist_ok=True, parents=True)

# User session logs directory
USER_LOGS_DIR = Path(__file__).parent / 'user_logs'
USER_LOGS_DIR.mkdir(exist_ok=True, parents=True)

# Get server identity for logging
import socket
SERVER_HOSTNAME = socket.gethostname()
SERVER_IP = socket.gethostbyname(SERVER_HOSTNAME) if SERVER_HOSTNAME else 'unknown'


class SessionLogger:
    """
    Handles per-user session logging.
    Creates a new log file for each user session with timestamp.
    """
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.start_time = datetime.now()
        self.filename = self._generate_filename()
        self.filepath = USER_LOGS_DIR / self.filename
        self.file = None
        self._open_log()
    
    def _generate_filename(self) -> str:
        """Generate filename with date and timestamp"""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        # Sanitize client_id for filename (replace : with _)
        safe_client_id = self.client_id.replace(':', '_').replace('.', '-')
        return f"session_{timestamp}_{safe_client_id}.log"
    
    def _open_log(self):
        """Open the log file for writing"""
        try:
            self.file = open(self.filepath, 'w', encoding='utf-8')
            self._write_header()
        except Exception as e:
            logger.error(f"Failed to open session log {self.filepath}: {e}")
    
    def _write_header(self):
        """Write session header information"""
        header = [
            "=" * 60,
            f"Session Log for {self.client_id}",
            f"Server: {SERVER_HOSTNAME} ({SERVER_IP})",
            f"Started: {self.start_time.isoformat()}",
            "=" * 60,
            ""
        ]
        for line in header:
            self._write(line)
    
    def _write(self, message: str):
        """Write a line to the log file"""
        if self.file:
            try:
                self.file.write(message + "\n")
                self.file.flush()  # Ensure immediate write
            except Exception as e:
                logger.error(f"Failed to write to session log: {e}")
    
    def log(self, level: str, message: str):
        """Log a message with timestamp (no truncation)."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] [{level.upper()}] {message}"
        self._write(log_line)
        # Also mirror to main logger at appropriate level
        if level.upper() == 'ERROR':
            logger.error(f"[Session {self.client_id}] {message}")
        else:
            logger.info(f"[Session {self.client_id}] {message}")
    
    def log_message(self, direction: str, msg_type: str, details: str = ""):
        """Log a WebSocket message; do not truncate details.

        For very large binary fields (like base64 images), include a concise
        summary with the field size to avoid creating unusably large files,
        but still record that the field existed.
        """
        arrow = ">>>" if direction == "recv" else "<<<"
        # If details looks like JSON, try to parse and summarize large fields
        try:
            import json as _json
            obj = _json.loads(details) if isinstance(details, str) and details.strip().startswith('{') else None
        except Exception:
            obj = None

        if obj and isinstance(obj, dict):
            # Summarize any large binary fields while keeping other data intact
            summarized = {}
            for k, v in obj.items():
                if k == 'data' and isinstance(v, dict):
                    # summarize inner data
                    inner = {}
                    for ik, iv in v.items():
                        if isinstance(iv, str) and len(iv) > 1000:
                            inner[ik] = f"<str length={len(iv)}>"
                        else:
                            inner[ik] = iv
                    summarized[k] = inner
                elif isinstance(v, str) and len(v) > 2000:
                    summarized[k] = f"<str length={len(v)}>"
                else:
                    summarized[k] = v
            try:
                detail_str = _json.dumps(summarized, ensure_ascii=False)
            except Exception:
                detail_str = str(summarized)
        else:
            detail_str = details

        self.log("MSG", f"{arrow} {msg_type}: {detail_str}")
    
    def close(self):
        """Close the log file and write footer"""
        if self.file:
            try:
                end_time = datetime.now()
                duration = (end_time - self.start_time).total_seconds()
                footer = [
                    "",
                    "=" * 60,
                    f"Session Ended: {end_time.isoformat()}",
                    f"Duration: {duration:.1f} seconds",
                    "=" * 60
                ]
                for line in footer:
                    self._write(line)
                self.file.close()
                self.file = None
                logger.info(f"Session log saved: {self.filepath}")
            except Exception as e:
                logger.error(f"Failed to close session log: {e}")


# Dictionary to track active session loggers
active_session_loggers: dict[str, SessionLogger] = {}

# GitHub Configuration
# Try environment variable first; if missing, attempt to read from GCP Secret Manager.
# You can set GITHUB_SECRET_NAME to the full resource name:
#   projects/PROJECT_ID/secrets/SECRET_NAME/versions/latest
# Or set GCP_PROJECT and GITHUB_SECRET_ID to build the resource name:
#   projects/<GCP_PROJECT>/secrets/<GITHUB_SECRET_ID>/versions/latest

def _get_secret_from_gcp(resource_name: str) -> str:
    try:
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(request={"name": resource_name})
        return response.payload.data.decode("utf-8")
    except Exception as e:
        logger.warning(f"Secret Manager access failed for {resource_name}: {e}")
        return ""


def _fetch_github_token() -> str:
    # Prefer explicit env var
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token

    # If a full secret resource name is provided, use it
    secret_name = os.environ.get("GITHUB_SECRET_NAME")
    if secret_name:
        token = _get_secret_from_gcp(secret_name)
        if token:
            return token

    # Try building a resource name from project + secret id
    project = os.environ.get("GCP_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    secret_id = os.environ.get("GITHUB_SECRET_ID", "GITHUB_TOKEN")
    if project:
        resource = f"projects/{project}/secrets/{secret_id}/versions/latest"
        token = _get_secret_from_gcp(resource)
        if token:
            return token

    logger.info("GITHUB_TOKEN not found in env or Secret Manager; continuing without token.")
    return ""

GITHUB_TOKEN = _fetch_github_token()
GITHUB_REPO = os.environ.get('GITHUB_REPO', 'program-at/ProgramAT')  # Format: owner/repo
PAUSE_DURATION = float(os.environ.get('PAUSE_DURATION', '5.0'))  # seconds to wait before creating issue

# Gemini Configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash-lite')  # Default to gemini-2.5-flash-lite

# Configure Gemini if API key is available
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Gemini Live manager for custom-GPT streaming mode
gemini_live_manager = GeminiLiveManager(GEMINI_API_KEY) if GEMINI_API_KEY else None

if SAVE_FRAMES:
    FRAMES_DIR.mkdir(exist_ok=True)
    # ensure text log exists
    (FRAMES_DIR / "received_texts.log").touch(exist_ok=True)

# Connected clients
connected_clients = set()

# Thread pool for blocking database operations
db_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="db")

# Statistics
stats = {
    'total_frames': 0,
    'total_bytes': 0,
    'total_texts': 0,
    'start_time': None
}

# Async wrapper for database operations
async def async_insert_logs_batch(batch):
    """Run batch insert in thread pool to avoid blocking event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(db_executor, copilot_db.insert_logs_batch, batch)

# Text tracking for GitHub issue creation
# prev_raw stores the last full transcript we received so we can compute a suffix (delta)
last_text = {'content': None, 'timestamp': None, 'task': None, 'prev_raw': None}

# Incomplete issue tracking for multi-turn conversations
# Global state for tracking per-user data
# Incomplete issue tracking for GitHub issue creation
incomplete_issue = {'data': None, 'missing_fields': [], 'timestamp': None}

# Issue iteration tracking - for updating existing issues
selected_issue = {'number': None, 'title': None, 'mode': 'create'}  # mode can be 'create' or 'update'
issue_cache = {'issues': [], 'last_fetch': None, 'cache_duration': 300}  # Cache for 5 minutes

# Store the last received frame for tool execution
last_frame = {'image': None, 'timestamp': None, 'base64': None}

# Active streaming tools - tracks which tools are running continuously per client
# Format: {client_id: {'tool': {...}, 'last_run': timestamp, 'throttle_ms': 1000}}
active_streaming_tools = {}

# Conversation images - stores images for chat follow-up questions
# Format: {conversation_id: PIL.Image}
conversation_images = {}

# YOLO model cache for tool executions - shared across all tool runs
# This prevents reloading models on every frame, enabling true real-time performance
yolo_model_cache = {}

# Active Copilot session streams - tracks streaming subprocesses per client
# Format: {client_id: {'process': subprocess, 'session_id': str, 'task': asyncio.Task}}
active_copilot_streams = {}

# Issues awaiting Copilot PR creation - tracks issues we're monitoring for PR creation
# Format: {issue_number: {'created_at': datetime, 'websocket': websocket, 'client_id': str}}
pending_copilot_issues = {}

# Issue template paths
TEMPLATE_DIR = Path(__file__).parent.parent / '.github' / 'ISSUE_TEMPLATE'


async def run_streaming_tools(websocket, client_id: str, image, image_base64: str): 
    """
    Run all active streaming tools for this client on the current frame.
    Respects throttle settings to avoid overwhelming the client.
    """
    global active_streaming_tools
    
    logger.info(f"run_streaming_tools called for {client_id}")
    
    if client_id not in active_streaming_tools:
        logger.warning(f"Client {client_id} not in active_streaming_tools")
        return
    
    tool_config = active_streaming_tools[client_id]
    now = datetime.now()
    
    # Gemini Live tools handle their own query loop — don't exec code here
    if tool_config.get('gemini_live'):
        return
    
    # Check if tool is installing modules (skip execution during installation)
    if tool_config.get('installing_module'):
        return  # Skip this frame, module installation in progress
    
    # Check if enough time has passed since last run (throttling)
    last_run = tool_config.get('last_run')
    throttle_ms = tool_config.get('throttle_ms', 500)  # Reduce to 500ms for better responsiveness
    
    if last_run:
        elapsed_ms = (now - last_run).total_seconds() * 1000
        if elapsed_ms < throttle_ms:
            logger.debug(f"Throttling {tool_name} for {client_id} - only {elapsed_ms:.0f}ms since last run")
            return  # Skip this frame, too soon
    
    # Update last run time
    tool_config['last_run'] = now
    
    # Get tool info
    tool = tool_config.get('tool', {})
    tool_name = tool.get('name', 'unknown')
    tool_code = tool.get('code', '')
    tool_language = tool.get('language', 'python')
    tool_input = tool.get('input', '')
    
    # Get cached execution environment or create it
    if 'exec_env' not in tool_config:
        # Parse input once and cache it
        import json as json_module
        parsed_input = tool_input
        if isinstance(tool_input, str):
            if tool_input.strip().startswith('{'):
                try:
                    parsed_input = json_module.loads(tool_input)
                except:
                    parsed_input = {'value': tool_input}
            else:
                parsed_input = {'value': tool_input} if tool_input else {}
        elif not isinstance(tool_input, dict):
            parsed_input = {'value': tool_input}
        
        # Cache the execution environment setup
        module_mgr = get_module_manager()
        common_modules = module_mgr.get_common_modules()
        
        # Cache stdout capture objects
        import io
        stdout_capture = io.StringIO()
        
        tool_config['exec_env'] = {
            'parsed_input': parsed_input,
            'common_modules': common_modules,
            'stdout_capture': stdout_capture,
            'exec_globals_base': {
                '__builtins__': __builtins__,
                '__file__': str(Path(__file__).parent.parent / 'tools' / 'dynamic_tool.py'),
                'yolo_model_cache': yolo_model_cache,
                **common_modules
            }
        }
    
    # Get cached environment
    exec_env = tool_config['exec_env']
    parsed_input = exec_env['parsed_input']  # Make it available as local variable
    
    # Execute the tool (Python only for now)
    if tool_language == 'python' and tool_code:
        logger.info(f"Starting execution of streaming tool {tool_name} for {client_id}")
        try:
            # Use cached environment with frame-specific data
            exec_globals = {
                **exec_env['exec_globals_base'],
                'input_data': exec_env['parsed_input'],
                'image': image,
                'image_base64': image_base64,
            }
            exec_locals = {}
            
            # Use cached stdout capture and clear it
            stdout_capture = exec_env['stdout_capture']
            stdout_capture.seek(0)
            stdout_capture.truncate(0)
            
            # Capture stdout to get print() output
            import sys
            old_stdout = sys.stdout
            
            # Retry loop for automatic module installation
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    logger.info(f"Executing {tool_name} code for {client_id}, attempt {retry_count + 1}")
                    # Redirect stdout to capture print statements
                    sys.stdout = stdout_capture
                    try:
                        # Execute tool code
                        logger.info(f"About to exec() tool code for {client_id}")
                        exec(tool_code, exec_globals, exec_locals)
                        logger.info(f"Successfully exec()d tool code for {client_id}")
                    finally:
                        sys.stdout = old_stdout
                    break  # Success, exit retry loop
                except (ImportError, ModuleNotFoundError) as e:
                    sys.stdout = old_stdout  # Restore stdout for logging
                    error_msg = str(e)
                    logger.warning(f"Module error in streaming {tool_name}: {error_msg}")
                    
                    # PAUSE STREAMING: Mark as installing
                    tool_config['installing_module'] = True
                    
                    # Notify client that streaming is paused for installation
                    await websocket.send(json.dumps({
                        'type': 'module_installing',
                        'tool_name': tool_name,
                        'message': f"Pausing streaming to install required module...",
                        'timestamp': datetime.now().isoformat()
                    }))
                    
                    # Install the module synchronously (just like one-shot mode)
                    # This will block, but streaming is already paused
                    installed_module = module_mgr.install_from_error(error_msg)
                    
                    # RESUME STREAMING: Clear installing flag
                    tool_config['installing_module'] = False
                    
                    if installed_module:
                        logger.info(f"Installed {installed_module}, resuming streaming execution...")
                        
                        # Notify client that streaming is resuming
                        await websocket.send(json.dumps({
                            'type': 'module_installed',
                            'tool_name': tool_name,
                            'module': installed_module,
                            'message': f"Successfully installed {installed_module}. Resuming streaming...",
                            'timestamp': datetime.now().isoformat()
                        }))
                        
                        # Reload common modules to include newly installed one
                        common_modules = module_mgr.get_common_modules()
                        exec_globals.update(common_modules)
                        retry_count += 1
                    else:
                        # Could not identify/install module
                        await websocket.send(json.dumps({
                            'type': 'module_install_failed',
                            'tool_name': tool_name,
                            'error': error_msg,
                            'message': f"Failed to install module: {error_msg}",
                            'timestamp': datetime.now().isoformat()
                        }))
                        raise
                except Exception:
                    # Non-module errors should be raised immediately
                    sys.stdout = old_stdout  # Restore stdout before raising
                    raise
            
            # Get captured print output
            printed_output = stdout_capture.getvalue()
            
            # IMPORTANT: Merge exec_locals into exec_globals so functions can see each other
            # This allows main() to call helper functions defined in the same file
            exec_globals.update(exec_locals)
            
            # Get result from function or variable
            # Check function signatures to avoid calling helper functions
            import inspect
            
            result = None
            
            # Try main() - highest priority
            if 'main' in exec_locals and callable(exec_locals['main']):
                try:
                    logger.info(f"Calling main() function for {tool_name} on {client_id}")
                    sig = inspect.signature(exec_locals['main'])
                    if len(sig.parameters) == 2:  # Should take (image, input_data)
                        result = exec_locals['main'](image, parsed_input)
                        logger.info(f"main() returned result for {tool_name}: {type(result)}")
                    else:
                        logger.warning(f"main() has {len(sig.parameters)} params, expected 2")
                except Exception as e:
                    logger.error(f"Error calling main(): {e}")
            
            # Try run() - second priority
            elif 'run' in exec_locals and callable(exec_locals['run']):
                try:
                    sig = inspect.signature(exec_locals['run'])
                    if len(sig.parameters) == 2:
                        result = exec_locals['run'](image, parsed_input)
                    else:
                        logger.warning(f"run() has {len(sig.parameters)} params, expected 2")
                except Exception as e:
                    logger.error(f"Error calling run(): {e}")
            
            # Try process_image() - but check signature carefully
            elif 'process_image' in exec_locals and callable(exec_locals['process_image']):
                try:
                    sig = inspect.signature(exec_locals['process_image'])
                    # Only call if it takes 2 params (entry point), not 1 (helper function)
                    if len(sig.parameters) == 2:
                        result = exec_locals['process_image'](image, parsed_input)
                    else:
                        logger.info(f"Skipping process_image() - has {len(sig.parameters)} params (likely a helper function)")
                except Exception as e:
                    logger.error(f"Error calling process_image(): {e}")
            
            # Try process_frame_for_text() - special case for text tools
            elif 'process_frame_for_text' in exec_locals and callable(exec_locals['process_frame_for_text']):
                try:
                    # Initialize verbalizer if needed
                    if 'initialize_verbalizer' in exec_locals and 'get_verbalizer' in exec_locals:
                        verbalizer = exec_locals['get_verbalizer']()
                        if not verbalizer and GEMINI_API_KEY:
                            exec_locals['initialize_verbalizer'](GEMINI_API_KEY)
                    result = exec_locals['process_frame_for_text'](image_base64)
                except Exception as e:
                    logger.error(f"Error calling process_frame_for_text(): {e}")
            
            # Check for result variable
            elif 'result' in exec_locals:
                result = exec_locals['result']
            else:
                # Tool is a library - show available functions
                available_funcs = [name for name, obj in exec_locals.items() 
                                 if callable(obj) and not name.startswith('_')]
                available_classes = [name for name, obj in exec_locals.items() 
                                   if isinstance(obj, type) and not name.startswith('_')]
                
                if available_funcs or available_classes:
                    result = f"Tool loaded successfully.\n\nAvailable functions: {', '.join(available_funcs) if available_funcs else 'none'}\nAvailable classes: {', '.join(available_classes) if available_classes else 'none'}\n\nNote: This tool is a library. To use it, you may need to call one of these functions."
                else:
                    result = None
            
            # Combine printed output with result
            # Check if result is advanced format with audio config
            if isinstance(result, dict) and ('audio' in result or 'text' in result):
                # Preserve audio configuration
                response_data = {
                    'type': 'tool_stream_result',
                    'tool_name': tool_name,
                    'status': 'success',
                    'result': result.get('text', ''),
                    'timestamp': now.isoformat()
                }
                
                # Add audio configuration if provided
                if 'audio' in result:
                    response_data['audio'] = result['audio']
                
                # Prepend printed output if any
                if printed_output.strip():
                    response_data['result'] = printed_output.strip() + '\n' + response_data['result']
                    if 'audio' in response_data and 'text' in response_data['audio']:
                        response_data['audio']['text'] = printed_output.strip() + '. ' + response_data['audio']['text']
                
                await websocket.send(json.dumps(response_data))
                logger.info(f"Sent tool_stream_result to {client_id} for {tool_name}")
            else:
                # Simple string format - add default audio
                output_parts = []
                if printed_output.strip():
                    output_parts.append(printed_output.strip())
                if result is not None:
                    output_parts.append(str(result))
                
                final_result = '\n'.join(output_parts) if output_parts else "Tool executed (no output)"
                
                # Send result back to client
                await websocket.send(json.dumps({
                    'type': 'tool_stream_result',
                    'tool_name': tool_name,
                    'status': 'success',
                    'result': final_result,
                    'audio': {
                        'type': 'speech',
                        'text': final_result,
                        'rate': 1.0,
                        'interrupt': False
                    },
                    'timestamp': now.isoformat()
                }))
            
        except Exception as e:
            logger.error(f"Error in streaming tool {tool_name}: {e}")
            # Don't send errors for streaming (would be too noisy)
            # Just log them for debugging


# =============================================================================
# Copilot Session Streaming Functions
# =============================================================================

def looks_like_code(line: str) -> bool:
    """
    Detect if a line is within a Markdown code block or looks like code.
    
    Markdown code indicators:
    - Inside triple backticks (```)
    - Indented code (4+ spaces)
    - Inline code with backticks
    
    General code patterns:
    - Contains common code syntax
    """
    if not line.strip():
        return False
    
    # Markdown code fence
    if line.strip().startswith('```'):
        return True
    
    # Markdown indented code block (4 spaces)
    if line.startswith('    ') and not line.strip().startswith('-') and not line.strip().startswith('*'):
        return True
    
    # Heavy indentation that's not a list
    if (line.startswith('        ') or line.startswith('\t\t')):
        return True
    
    # Code patterns (only if heavily syntactic)
    code_patterns = [
        r'=>', r'===?', r'!==?', r'<=', r'>=', r'&&', r'\|\|',
        r'[{}\[\]()]{2,}',  # Multiple brackets
        r'\bfunction\s*\(', r'\bclass\s+\w+', r'\bdef\s+\w+\(',
        r'\bimport\s+\{', r'\bexport\s+(default|const)',
    ]
    
    for pattern in code_patterns:
        if re.search(pattern, line):
            return True
    
    return False


def is_entry_boundary(line: str, prev_line: str) -> bool:
    """
    Detect entry boundaries using Markdown structure and log patterns.
    
    Markdown entry boundaries:
    - Headers (# ## ###)
    - Horizontal rules (---, ***)
    - New top-level list item (not nested)
    - Code fence boundaries (```)
    - Timestamp/log-style prefixes (common in Copilot logs)
    - Common log action prefixes (Running, Creating, Step N, etc.)
    """
    if not line.strip():
        return False  # Blank lines don't start entries
    
    # Markdown headers
    if re.match(r'^\s*#{1,6}\s+', line):
        return True
    
    # Horizontal rules
    if re.match(r'^\s*(-{3,}|\*{3,}|_{3,})\s*$', line):
        return True
    
    # Top-level list items (not indented)
    # Match: - item, * item, + item, 1. item (at start of line)
    if re.match(r'^[•\-\*\+](\s|$)|^\d+\.\s', line) and prev_line.strip():
        return True
    
    # Code fence boundaries
    if line.strip().startswith('```'):
        return True
    
    # Timestamp-prefixed lines (common in Copilot agent logs)
    # e.g., "2026-02-15T10:30:00Z ..." or "[10:30:00]" or "10:30:00 -"
    if re.match(r'^\d{4}-\d{2}-\d{2}[T ]|^\[\d{2}:\d{2}', line):
        return True
    
    # Lines starting with common log prefixes
    if re.match(r'^\s*(INFO|DEBUG|WARN|ERROR|Step \d|Running|Executing|Creating|Updating|Reading|Writing|Checking)\b', line, re.IGNORECASE):
        return True
    
    return False


async def _process_entry_batch(websocket, session_id: str, pr_or_session: str, entries: list):
    """
    Process a batch of entries: generate summary and send to client.
    
    Args:
        websocket: WebSocket connection
        session_id: Copilot session ID
        pr_or_session: PR number or session ID (for client reference)
        entries: List of entry dicts with 'entry_num', 'text', 'is_code'
    """
    if not entries:
        return
    
    # Generate summary (run in executor to avoid blocking)
    loop = asyncio.get_event_loop()
    try:
        summary = await loop.run_in_executor(None, summarize_entries_sync, entries)
        
        # Get entry range
        start_entry = entries[0]['entry_num']
        end_entry = entries[-1]['entry_num']
        
        # Store summary in database
        copilot_db.insert_summary(session_id, summary, start_entry, end_entry)
        
        # Send summary to client
        print(f"  📝 SUMMARY: {summary}", flush=True)
        await websocket.send(json.dumps({
            'type': 'copilot_summary',
            'pr_or_session': str(pr_or_session),
            'session_id': session_id,
            'summary': summary,
            'start_entry': start_entry,
            'end_entry': end_entry,
            'timestamp': datetime.now().isoformat()
        }))
        
        logger.info(f"Generated and sent summary for entries {start_entry}-{end_entry}")
        
    except Exception as e:
        logger.error(f"Error processing entry batch: {e}")


async def fetch_copilot_sessions() -> list:
    """
    Fetch active Copilot coding agent tasks for the repository.
    Uses `gh agent-task list` CLI command.
    
    Note: gh agent-task list doesn't support --json, so we parse text output
    or just return empty and rely on PR-based lookup instead.
    
    Returns:
        List of session dictionaries (may be empty if parsing fails)
    """
    try:
        # gh agent-task list doesn't have --json flag, so we'll return empty
        # and rely on the PR number based lookup in poll_for_copilot_session
        logger.info("fetch_copilot_sessions called - using PR-based lookup instead")
        return []
        
    except Exception as e:
        logger.error(f"Error fetching Copilot sessions: {e}")
        return []


async def stream_copilot_session_logs(websocket, client_id: str, pr_number_or_session: str, pr_number: int = None, skip_wait: bool = False, _last_summary_time=None, _skip_lines: int = 0, _start_entry_count: int = 0):
    """
    Stream Copilot agent task logs to a client using `gh agent-task view --follow`.
    
    Args:
        websocket: WebSocket connection to send logs to
        client_id: Client identifier for tracking
        pr_number_or_session: PR number or session ID to stream logs for
        pr_number: Explicit PR number (optional, used when pr_number_or_session is a session ID)
        skip_wait: If True, skip the 45-second initialization wait (caller already confirmed session exists)
        _last_summary_time: Internal: carry forward last summary time on reconnect to prevent rapid-fire summaries
        _skip_lines: Internal: number of log lines to skip on reconnect (already processed)
        _start_entry_count: Internal: starting entry count on reconnect (continue numbering)
    """
    global active_copilot_streams
    
    logger.info(f"Starting Copilot agent task log stream for {client_id}, pr/session: {pr_number_or_session}, pr_number: {pr_number}")
    
    try:
        # Create clean environment for gh CLI - remove token env vars so it uses OAuth from gh auth login
        # PAT tokens (GITHUB_TOKEN) don't work with agent-task, it requires OAuth credentials
        env = os.environ.copy()
        env.pop('GH_TOKEN', None)
        env.pop('GITHUB_TOKEN', None)
        
        session_id = str(pr_number_or_session)
        
        # If given a PR number, first fetch the session ID
        if session_id.isdigit():
            import pty
            import fcntl
            import re
            
            pr_number = int(session_id)  # Store the PR number
            logger.info(f"PR number provided, fetching session ID for PR #{session_id}")
            
            # Wait for the session to initialize on GitHub's side before querying
            # Skip this wait if the caller already confirmed the session exists
            if not skip_wait:
                logger.info("Waiting 45 seconds for GitHub to initialize the Copilot session...")
                await asyncio.sleep(45)
            else:
                logger.info("Skipping 45-second wait (session already confirmed)")
            
            # Use Python's pty module to create a real PTY and send enter after delay
            logger.info(f"Creating PTY for: gh agent-task view {session_id} -R {GITHUB_REPO}")
            
            # Create a PTY
            master, slave = pty.openpty()
            
            # Spawn the gh command in the PTY
            proc = await asyncio.create_subprocess_exec(
                'gh', 'agent-task', 'view', str(session_id),
                '-R', GITHUB_REPO,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                env=env
            )
            
            # Close slave in parent process
            os.close(slave)
            
            # Send multiple enters with delays to ensure we catch the prompt
            # First wait for gh to query and display sessions
            logger.info("Waiting 0.8 seconds then sending first enter...")
            await asyncio.sleep(0.8)
            os.write(master, b'\r\n')
            logger.info("Sent first enter, waiting 0.3 seconds...")
            await asyncio.sleep(0.3)
            os.write(master, b'\r\n')
            logger.info("Sent second enter to PTY")
            
            # Read output
            output = b''
            try:
                # Set non-blocking
                flags = fcntl.fcntl(master, fcntl.F_GETFL)
                fcntl.fcntl(master, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                
                # Read with timeout
                for _ in range(50):  # 5 seconds total
                    await asyncio.sleep(0.1)
                    try:
                        chunk = os.read(master, 4096)
                        if chunk:
                            output += chunk
                    except OSError:
                        pass
                    
                    # Check if process has exited
                    if proc.returncode is not None:
                        break
            finally:
                os.close(master)
                
            # Wait for process to complete
            await proc.wait()
            
            logger.info(f"Command completed with return code: {proc.returncode}")
            output_str = output.decode('utf-8', errors='replace')
            logger.info(f"Output length: {len(output_str)}")
            
            # Parse session ID from output - look for UUID pattern
            uuid_pattern = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
            matches = re.findall(uuid_pattern, output_str)
            logger.info(f"Found {len(matches)} UUID matches in output")
            
            if matches:
                session_id = matches[0]
                logger.info(f"Selected most recent session: {session_id}")
            else:
                logger.error(f"Could not find session ID in output: {output_str[:200]}")
                await websocket.send(json.dumps({
                    'type': 'copilot_session_error',
                    'pr_or_session': str(pr_number),
                    'error': 'Could not find session ID for this PR'
                }))
                return
        
        # Spawn the gh CLI process with --follow and --log for continuous streaming
        proc = await asyncio.create_subprocess_exec(
            'gh', 'agent-task', 'view',
            session_id,
            '-R', GITHUB_REPO,
            '--follow',
            '--log',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        # Create or get session in database with PR number
        if pr_number is not None:
            logger.info(f"Creating session {session_id} for PR #{pr_number} in database")
            copilot_db.create_session(session_id, pr_number)
        else:
            logger.warning(f"Session {session_id} created without PR number - will not be queryable by PR")
        
        # Store process reference for cleanup
        active_copilot_streams[client_id] = {
            'process': proc,
            'pr_or_session': pr_number_or_session,
            'started_at': datetime.now(),
            'session_id': session_id
        }
        
        # Notify client that streaming has started
        print(f"\n{'='*60}\n🤖 Copilot Session Started: {session_id}\n{'='*60}", flush=True)
        await websocket.send(json.dumps({
            'type': 'copilot_session_stream_started',
            'pr_or_session': str(pr_number_or_session),
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }))
        
        # Entry tracking state
        current_entry_lines = []
        entry_buffer = []  # Buffer of complete entries
        entry_count = _start_entry_count  # Continue from where we left off on reconnect
        log_index = _skip_lines  # Continue log index from reconnect
        prev_line = ""
        in_code_section = False
        lines_seen = 0  # Count lines seen this invocation (for skip logic)
        last_summary_time = _last_summary_time or datetime.now()  # Don't fire immediately on first run
        summary_interval = 30  # Generate summary every 30 seconds
        check_interval = 5  # Check for new entries every 5 seconds
        max_entry_lines = 50  # Force a boundary if an entry exceeds this many lines
        
        # Background task to periodically generate summaries
        async def periodic_summarizer():
            """Check every 5 seconds, summarize if 30 seconds have passed since last summary."""
            nonlocal entry_buffer, last_summary_time
            while True:
                try:
                    await asyncio.sleep(check_interval)
                    now = datetime.now()
                    
                    # Only summarize if we have entries
                    if not entry_buffer:
                        continue
                    
                    # If first batch or 30 seconds have passed since last summary
                    if last_summary_time is None:
                        time_since_last = float('inf')
                    else:
                        time_since_last = (now - last_summary_time).total_seconds()
                    
                    if time_since_last >= summary_interval:
                        # Copy and clear buffer
                        entries_to_process = entry_buffer.copy()
                        entry_buffer.clear()
                        last_summary_time = now
                        
                        if not entries_to_process:
                            continue
                        
                        logger.info(f"Periodic summarizer: Processing {len(entries_to_process)} entries ({time_since_last:.1f}s since last summary)")
                        
                        # Cap total text size to avoid overwhelming the summarizer
                        # Any entries that don't fit go back in the buffer for next cycle
                        total_chars = sum(len(e.get('text', '')) for e in entries_to_process)
                        if total_chars > 15000:
                            kept = []
                            deferred = []
                            char_count = 0
                            for entry in entries_to_process:
                                entry_text = entry.get('text', '')
                                if char_count + len(entry_text) > 15000 and kept:
                                    # This entry and all remaining go back in the buffer
                                    deferred.append(entry)
                                else:
                                    kept.append(entry)
                                    char_count += len(entry_text)
                            # Put un-processed entries back in the buffer for next cycle
                            if deferred:
                                entry_buffer = deferred + entry_buffer
                                logger.info(f"Capped at {len(kept)} entries ({char_count} chars), deferred {len(deferred)} entries to next cycle")
                            entries_to_process = kept
                        
                        if entries_to_process:
                            await _process_entry_batch(websocket, session_id, pr_number_or_session, entries_to_process)
                        
                        logger.info(f"Summary complete, next in {summary_interval}s")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in periodic summarizer: {e}")
        
        # Start the periodic summarizer task
        summarizer_task = asyncio.create_task(periodic_summarizer())
        
        try:
            # Read and forward log lines as they arrive
            while True:
                line = await proc.stdout.readline()
                
                if not line:
                    # Process ended or EOF - stop the periodic summarizer first
                    summarizer_task.cancel()
                    try:
                        await summarizer_task
                    except asyncio.CancelledError:
                        pass
                    
                    # Flush any remaining entry
                    if current_entry_lines:
                        entry_text = '\n'.join(current_entry_lines)
                        entry_buffer.append({
                            'entry_num': entry_count,
                            'lines': current_entry_lines[:],
                            'text': entry_text,
                            'is_code': in_code_section
                        })
                        entry_count += 1
                        
                        # Store in DB using batch insert (non-blocking)
                        batch = [(session_id, log_index + i, line_text, in_code_section, entry_count - 1) 
                                for i, line_text in enumerate(current_entry_lines)]
                        asyncio.create_task(async_insert_logs_batch(batch))
                        log_index += len(current_entry_lines)
                    
                    # Summarize any remaining entries
                    if entry_buffer:
                        await _process_entry_batch(websocket, session_id, pr_number_or_session, entry_buffer)
                        entry_buffer.clear()  # Clear after final summary
                    
                    break
                
                log_line = line.decode('utf-8', errors='replace').rstrip()
                
                if log_line:
                    lines_seen += 1
                    
                    # Skip lines already processed in previous invocation (reconnect)
                    if lines_seen <= _skip_lines:
                        prev_line = log_line
                        continue
                    
                    # Print to terminal for server-side visibility
                    print(f"  {log_line}", flush=True)
                    
                    # Send raw log to client
                    await websocket.send(json.dumps({
                        'type': 'copilot_session_log',
                        'pr_or_session': str(pr_number_or_session),
                        'line': log_line,
                        'timestamp': datetime.now().isoformat()
                    }))
                    
                    # Track code fence state (```)
                    if log_line.strip().startswith('```'):
                        in_code_section = not in_code_section  # Toggle
                    
                    # Check if this line looks like code
                    line_is_code = looks_like_code(log_line) or in_code_section
                    
                    # Detect entry boundary OR force boundary if entry is too large
                    force_boundary = len(current_entry_lines) >= max_entry_lines
                    natural_boundary = is_entry_boundary(log_line, prev_line)
                    
                    if (natural_boundary or force_boundary) and current_entry_lines:
                        if force_boundary and not natural_boundary:
                            logger.debug(f"Forcing entry boundary at {len(current_entry_lines)} lines")
                        
                        # Complete the current entry
                        entry_text = '\n'.join(current_entry_lines)
                        entry_buffer.append({
                            'entry_num': entry_count,
                            'lines': current_entry_lines[:],
                            'text': entry_text,
                            'is_code': in_code_section
                        })
                        
                        # Store in DB using batch insert (non-blocking)
                        batch = [(session_id, log_index + i, line_text, in_code_section, entry_count) 
                                for i, line_text in enumerate(current_entry_lines)]
                        asyncio.create_task(async_insert_logs_batch(batch))
                        log_index += len(current_entry_lines)
                        
                        entry_count += 1
                        current_entry_lines = []
                        # Note: Don't reset in_code_section here - it persists across entries
                    
                    # Add line to current entry
                    if log_line.strip():  # Don't add blank lines
                        current_entry_lines.append(log_line)
                    
                    prev_line = log_line
        finally:
            # Cancel the summarizer task when done
            summarizer_task.cancel()
            try:
                await summarizer_task
            except asyncio.CancelledError:
                pass

        
        # Check for any stderr output
        stderr_output = await proc.stderr.read()
        if stderr_output:
            stderr_text = stderr_output.decode('utf-8', errors='replace').strip()
            if stderr_text:
                # Print to terminal
                print(f"⚠️  STDERR: {stderr_text}", flush=True)
                logger.warning(f"Copilot agent task stderr: {stderr_text}")
                # Only send as error if it looks like an actual error (not just warnings/info)
                if any(err_word in stderr_text.lower() for err_word in ['error', 'failed', 'fatal']):
                    await websocket.send(json.dumps({
                        'type': 'copilot_session_error',
                        'pr_or_session': str(pr_number_or_session),
                        'error': stderr_text,
                        'timestamp': datetime.now().isoformat()
                    }))
        
        # Get exit code
        exit_code = await proc.wait()
        
        logger.info(f"gh agent-task view exited with code {exit_code} for session {session_id}")
        
        # Check if the session is actually complete or if it's still running
        # (This can happen when a sub-agent finishes but the parent is still active)
        try:
            # Query the session status
            status_proc = await asyncio.create_subprocess_exec(
                'gh', 'agent-task', 'view',
                session_id,
                '-R', GITHUB_REPO,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            stdout, stderr = await status_proc.communicate()
            output = stdout.decode('utf-8', errors='replace')
            
            # Check if session shows as "in progress" or "active"
            is_still_active = any(status in output.lower() for status in ['in progress', 'active', 'running'])
            
            if is_still_active:
                if exit_code == 0:
                    # Session is still active - the --follow just ended (likely due to sub-agent completion)
                    logger.info(f"Session {session_id} is still active, restarting follow stream...")
                    print(f"  🔄 Sub-agent completed, continuing to follow parent session...", flush=True)
                else:
                    # Non-zero exit but session still active - connection issue, try to reconnect
                    logger.warning(f"Session {session_id} is still active but stream exited with code {exit_code}, reconnecting...")
                    print(f"  🔄 Connection lost, attempting to reconnect to active session...", flush=True)
                
                # Restart the follow by calling stream_copilot_session_logs recursively
                # Use the same websocket and parameters, carry forward summary timing
                await stream_copilot_session_logs(websocket, client_id, session_id, pr_number, skip_wait=True, _last_summary_time=last_summary_time, _skip_lines=lines_seen, _start_entry_count=entry_count)
                return  # Exit this instance, the recursive call handles the rest
                
        except Exception as e:
            logger.warning(f"Could not check session status: {e}, assuming complete")
        
        # Determine session status based on exit code and whether we got any logs
        # Get log count to see if we captured anything
        logs_count = copilot_db.get_log_count(session_id)
        
        if exit_code == 0:
            status = 'completed'
        elif logs_count > 0:
            # We got some logs but stream ended abnormally - mark as partial, not failed
            status = 'partial'
            logger.info(f"Session {session_id} recording ended early (exit {exit_code}) but captured {logs_count} log entries - marking as partial")
        else:
            # No logs captured and non-zero exit
            status = 'failed'
            
        copilot_db.update_session_status(session_id, status, exit_code)
        
        # Notify client that streaming has ended
        print(f"{'='*60}\n✅ Copilot Session Ended (exit code: {exit_code})\n{'='*60}\n", flush=True)
        await websocket.send(json.dumps({
            'type': 'copilot_session_stream_ended',
            'pr_or_session': str(pr_number_or_session),
            'session_id': session_id,
            'exit_code': exit_code,
            'timestamp': datetime.now().isoformat()
        }))
        
        logger.info(f"Copilot agent task log stream ended for {client_id}, exit code: {exit_code}")
        
    except asyncio.CancelledError:
        logger.info(f"Copilot agent task stream cancelled for {client_id}")
        raise
    except Exception as e:
        logger.error(f"Error streaming Copilot agent task logs: {e}")
        try:
            await websocket.send(json.dumps({
                'type': 'copilot_session_error',
                'pr_or_session': str(pr_number_or_session),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }))
        except:
            pass  # Connection may already be closed
    finally:
        # Cleanup
        if client_id in active_copilot_streams:
            stream_info = active_copilot_streams[client_id]
            if stream_info.get('process'):
                try:
                    stream_info['process'].terminate()
                    await asyncio.wait_for(stream_info['process'].wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    stream_info['process'].kill()
                except:
                    pass
            del active_copilot_streams[client_id]


async def stop_copilot_session_stream(client_id: str):
    """
    Stop an active Copilot session stream for a client.
    
    Args:
        client_id: Client identifier
    """
    global active_copilot_streams
    
    if client_id not in active_copilot_streams:
        logger.warning(f"No active Copilot stream for {client_id}")
        return
    
    stream_info = active_copilot_streams[client_id]
    
    # Cancel the streaming task if it exists
    if 'task' in stream_info and stream_info['task']:
        stream_info['task'].cancel()
    
    # Terminate the subprocess
    if stream_info.get('process'):
        try:
            stream_info['process'].terminate()
            await asyncio.wait_for(stream_info['process'].wait(), timeout=5.0)
        except asyncio.TimeoutError:
            stream_info['process'].kill()
        except Exception as e:
            logger.error(f"Error stopping Copilot stream process: {e}")
    
    if client_id in active_copilot_streams:
        del active_copilot_streams[client_id]
    
    logger.info(f"Stopped Copilot session stream for {client_id}")


async def fetch_and_store_pr_sessions(websocket, client_id: str, pr_number: int):
    """
    Fetch sessions for a PR and return them to the client.
    
    Note: GitHub CLI doesn't provide a way to list historical sessions.
    We can only access:
    1. Sessions already stored in our database from previous live streaming
    2. The currently active session (if any) by streaming it live
    
    Args:
        websocket: WebSocket connection to send updates to
        client_id: Client identifier
        pr_number: PR number to fetch sessions for
    """
    logger.info(f"Fetching sessions for PR #{pr_number}")
    
    try:
        # Check database for existing sessions
        sessions = copilot_db.get_sessions_for_pr(pr_number)
        
        if sessions:
            logger.info(f"Found {len(sessions)} session(s) in database for PR #{pr_number}")
            await websocket.send(json.dumps({
                'type': 'pr_sessions_list',
                'pr_number': pr_number,
                'sessions': sessions,
                'timestamp': datetime.now().isoformat()
            }))
            return
        
        # No sessions in DB - check if there's an active session we can fetch
        logger.info(f"No sessions in DB for PR #{pr_number}, checking for active session")
        
        # Try to get active session by running gh agent-task view
        env = os.environ.copy()
        env.pop('GH_TOKEN', None)
        env.pop('GITHUB_TOKEN', None)
        
        proc = await asyncio.create_subprocess_exec(
            'gh', 'agent-task', 'view',
            str(pr_number),
            '-R', GITHUB_REPO,
            '--log',  # Just get log info, not streaming
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            # No active session for this PR
            logger.info(f"No active session found for PR #{pr_number}")
            await websocket.send(json.dumps({
                'type': 'pr_sessions_list',
                'pr_number': pr_number,
                'sessions': [],
                'timestamp': datetime.now().isoformat()
            }))
            return
        
        # Parse output to extract session ID
        output = stdout.decode('utf-8', errors='replace')
        import re
        uuid_pattern = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
        session_ids = re.findall(uuid_pattern, output)
        
        if session_ids:
            session_id = session_ids[0]  # Use first found session ID
            logger.info(f"Found active session {session_id} for PR #{pr_number}, creating in DB")
            
            # Create session in database
            copilot_db.create_session(session_id, pr_number)
            
            # Fetch logs for this session without streaming
            await fetch_and_store_session_logs(session_id, pr_number)
            
            # Get sessions from database and send to client
            sessions = copilot_db.get_sessions_for_pr(pr_number)
            await websocket.send(json.dumps({
                'type': 'pr_sessions_list',
                'pr_number': pr_number,
                'sessions': sessions,
                'timestamp': datetime.now().isoformat()
            }))
        else:
            logger.info(f"Could not extract session ID for PR #{pr_number}")
            await websocket.send(json.dumps({
                'type': 'pr_sessions_list',
                'pr_number': pr_number,
                'sessions': [],
                'timestamp': datetime.now().isoformat()
            }))
        
    except Exception as e:
        logger.error(f"Error fetching sessions for PR #{pr_number}: {e}")
        await websocket.send(json.dumps({
            'type': 'error',
            'message': f'Failed to fetch sessions: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }))


async def fetch_and_store_session_logs(session_id: str, pr_number: int):
    """
    Fetch logs for a specific session from GitHub and store in database.
    Uses gh agent-task view --log (without --follow) to get historical logs.
    
    Args:
        session_id: Copilot session ID
        pr_number: Associated PR number
    """
    logger.info(f"Fetching historical logs for session {session_id}")
    
    try:
        # Create clean environment for gh CLI
        env = os.environ.copy()
        env.pop('GH_TOKEN', None)
        env.pop('GITHUB_TOKEN', None)
        
        # Run gh agent-task view with --log but WITHOUT --follow for historical data
        proc = await asyncio.create_subprocess_exec(
            'gh', 'agent-task', 'view',
            session_id,
            '-R', GITHUB_REPO,
            '--log',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        # Process logs line by line
        current_entry_lines = []
        entry_buffer = []
        entry_count = 0
        log_index = 0
        prev_line = ""
        in_code_section = False
        max_entry_lines = 50  # Force boundary if entry exceeds this
        
        while True:
            line = await proc.stdout.readline()
            
            if not line:
                # EOF - flush any remaining entry
                if current_entry_lines:
                    entry_text = '\n'.join(current_entry_lines)
                    entry_buffer.append({
                        'entry_num': entry_count,
                        'lines': current_entry_lines[:],
                        'text': entry_text,
                        'is_code': in_code_section
                    })
                    entry_count += 1
                    
                    # Store in DB using batch insert (non-blocking)
                    batch = [(session_id, log_index + i, line_text, in_code_section, entry_count - 1) 
                            for i, line_text in enumerate(current_entry_lines)]
                    asyncio.create_task(async_insert_logs_batch(batch))
                    log_index += len(current_entry_lines)
                
                # Summarize any remaining entries
                if entry_buffer:
                    await _process_historical_batch(session_id, entry_buffer)
                
                break
            
            log_line = line.decode('utf-8', errors='replace').rstrip()
            
            if log_line:
                # Track code fence state (```)
                if log_line.strip().startswith('```'):
                    in_code_section = not in_code_section  # Toggle
                
                # Check if this line looks like code
                line_is_code = looks_like_code(log_line) or in_code_section
                
                # Detect entry boundary or force if too large
                force_boundary = len(current_entry_lines) >= max_entry_lines
                natural_boundary = is_entry_boundary(log_line, prev_line)
                
                if (natural_boundary or force_boundary) and current_entry_lines:
                    # Complete the current entry
                    entry_text = '\n'.join(current_entry_lines)
                    entry_buffer.append({
                        'entry_num': entry_count,
                        'lines': current_entry_lines[:],
                        'text': entry_text,
                        'is_code': in_code_section
                    })
                    
                    # Store in DB using batch insert (non-blocking)
                    batch = [(session_id, log_index + i, line_text, in_code_section, entry_count) 
                            for i, line_text in enumerate(current_entry_lines)]
                    asyncio.create_task(async_insert_logs_batch(batch))
                    log_index += len(current_entry_lines)
                    
                    entry_count += 1
                    current_entry_lines = []
                    # Don't reset in_code_section - it persists across entries
                    
                    # For historical logs, batch every 10 entries to avoid too many summaries
                    if len(entry_buffer) >= 10:
                        await _process_historical_batch(session_id, entry_buffer[:10])
                        entry_buffer = entry_buffer[10:]
                
                # Add line to current entry
                if log_line.strip():
                    current_entry_lines.append(log_line)
                
                prev_line = log_line
        
        # Get exit code
        exit_code = await proc.wait()
        
        # Determine session status based on exit code and whether we got any logs
        if exit_code == 0:
            status = 'completed'
        elif log_index > 0:
            # We got some logs but stream ended abnormally - mark as partial, not failed
            status = 'partial'
            logger.info(f"Session {session_id} fetch ended early (exit {exit_code}) but captured {log_index} log entries - marking as partial")
        else:
            # No logs captured and non-zero exit
            status = 'failed'
            
        copilot_db.update_session_status(session_id, status, exit_code)
        
        logger.info(f"Stored {log_index} log lines and {entry_count} entries for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error fetching logs for session {session_id}: {e}")


async def _process_historical_batch(session_id: str, entries: list):
    """
    Process a batch of historical entries: generate summary and store in DB.
    Similar to _process_entry_batch but without WebSocket sending.
    
    Args:
        session_id: Copilot session ID
        entries: List of entry dicts
    """
    if not entries:
        return
    
    # Generate summary (including code now)
    loop = asyncio.get_event_loop()
    try:
        summary = await loop.run_in_executor(None, summarize_entries_sync, entries)
        
        # Get entry range
        start_entry = entries[0]['entry_num']
        end_entry = entries[-1]['entry_num']
        
        # Store summary in database
        copilot_db.insert_summary(session_id, summary, start_entry, end_entry)
        
        logger.info(f"Generated summary for historical entries {start_entry}-{end_entry}: {summary[:50]}...")
        
    except Exception as e:
        logger.error(f"Error processing historical batch: {e}")


async def poll_for_copilot_session_on_pr(pr_number: int, websocket, client_id: str):
    """
    Poll for a Copilot agent task on a specific PR (used when commenting on existing PRs).
    Much simpler than poll_for_copilot_session since we already know the PR number.
    
    Args:
        pr_number: The PR number where @copilot was mentioned
        websocket: WebSocket connection to stream logs to
        client_id: Client identifier
    """
    logger.info(f"Polling for Copilot session on PR #{pr_number}")
    
    max_wait = 120  # Wait up to 2 minutes for session to start
    poll_interval = 5  # Check every 5 seconds
    start_time = datetime.now()
    
    try:
        # Notify client we're watching
        await websocket.send(json.dumps({
            'type': 'copilot_watching',
            'pr_number': pr_number,
            'message': f'Waiting for Copilot session to start on PR #{pr_number}...',
            'timestamp': datetime.now().isoformat()
        }))
        
        # Create clean environment for gh CLI
        env = os.environ.copy()
        env.pop('GH_TOKEN', None)
        env.pop('GITHUB_TOKEN', None)
        
        while (datetime.now() - start_time).total_seconds() < max_wait:
            try:
                # Check if session exists for this PR using gh agent-task list
                proc = await asyncio.create_subprocess_exec(
                    'gh', 'agent-task', 'list',
                    '--limit', '50',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                stdout, stderr = await proc.communicate()
                stdout_text = stdout.decode('utf-8', errors='replace').strip()
                
                # Look for our PR in the output
                if f'#{pr_number}' in stdout_text and 'program-at/ProgramAT' in stdout_text:
                    logger.info(f"Found session on PR #{pr_number}, starting stream")
                    
                    # Notify client
                    await websocket.send(json.dumps({
                        'type': 'copilot_session_starting',
                        'pr_number': pr_number,
                        'message': f'Copilot session found, streaming logs...',
                        'timestamp': datetime.now().isoformat()
                    }))
                    
                    # Start streaming logs - stream function will extract the session ID
                    # skip_wait=True because we already confirmed the session exists via agent-task list
                    await stream_copilot_session_logs(
                        websocket,
                        client_id,
                        str(pr_number),  # Pass PR number, function will get session ID
                        pr_number=pr_number,
                        skip_wait=True
                    )
                    return
                else:
                    logger.debug(f"No session yet on PR #{pr_number}, waiting...")
                    
            except Exception as e:
                logger.warning(f"Error checking for session on PR #{pr_number}: {e}")
            
            await asyncio.sleep(poll_interval)
        
        # Timeout
        logger.info(f"Timed out waiting for Copilot session on PR #{pr_number}")
        await websocket.send(json.dumps({
            'type': 'copilot_session_timeout',
            'pr_number': pr_number,
            'message': f'No Copilot session started within {max_wait} seconds',
            'timestamp': datetime.now().isoformat()
        }))
        
    except Exception as e:
        logger.error(f"Error polling for session on PR #{pr_number}: {e}")


async def poll_for_copilot_session(issue_number: int, websocket, client_id: str):
    """
    Poll for a Copilot agent task associated with a newly created issue.
    Watches for PR creation and then streams the agent task logs using gh agent-task view.
    
    Args:
        issue_number: The GitHub issue number to monitor
        websocket: WebSocket connection to stream logs to
        client_id: Client identifier
    """
    global pending_copilot_issues
    
    logger.info(f"Starting to poll for Copilot agent task for issue #{issue_number}")
    
    max_poll_duration = 600  # Poll for up to 10 minutes
    poll_interval = 15  # Check every 15 seconds
    session_poll_interval = 5  # Check for session more frequently once PR is found
    start_time = datetime.now()
    found_pr = None
    
    # Helper to send to websocket, tolerating disconnects
    async def _safe_send(msg_dict):
        """Send JSON to the websocket. Returns False if send failed (client gone)."""
        nonlocal websocket
        # Refresh websocket from pending_copilot_issues in case client reconnected
        info = pending_copilot_issues.get(issue_number)
        if info and info.get('websocket'):
            websocket = info['websocket']
        try:
            await websocket.send(json.dumps(msg_dict))
            return True
        except Exception:
            logger.debug(f"WebSocket send failed for issue #{issue_number} poll (client disconnected)")
            return False
    
    try:
        # Notify client we're watching for Copilot
        await _safe_send({
            'type': 'copilot_watching',
            'issue_number': issue_number,
            'message': f'Watching for Copilot to create a PR for issue #{issue_number}...',
            'timestamp': datetime.now().isoformat()
        })
        
        while (datetime.now() - start_time).total_seconds() < max_poll_duration:
            # Check if client disconnected
            if issue_number not in pending_copilot_issues:
                logger.info(f"Stopped polling for issue #{issue_number} - removed from pending")
                return
            
            try:
                # If we haven't found a PR yet, look for one
                if not found_pr:
                    g = Github(GITHUB_TOKEN)
                    repo = g.get_repo(GITHUB_REPO)
                    pulls = repo.get_pulls(state='open', sort='created', direction='desc')
                    
                    for pr in list(pulls)[:10]:  # Check recent PRs
                        pr_text = f"{pr.title} {pr.body or ''}"
                        if f"#{issue_number}" in pr_text or f"Fixes #{issue_number}" in pr_text or f"Closes #{issue_number}" in pr_text:
                            logger.info(f"Found PR #{pr.number} for issue #{issue_number}")
                            found_pr = pr
                            
                            # Store PR number in pending info for reconnecting clients
                            if issue_number in pending_copilot_issues:
                                pending_copilot_issues[issue_number]['pr_number'] = pr.number
                            
                            # Notify client about PR (tolerate disconnects)
                            await _safe_send({
                                'type': 'copilot_pr_found',
                                'issue_number': issue_number,
                                'pr_number': pr.number,
                                'pr_title': pr.title,
                                'pr_url': pr.html_url,
                                'message': f'Copilot created PR #{pr.number}: {pr.title}. Waiting for agent session to start...',
                                'timestamp': datetime.now().isoformat()
                            })
                            break
                
                # If we found a PR, try to stream its session
                # The stream function will use gh agent-task list to find the session ID
                if found_pr:
                    logger.info(f"Found PR #{found_pr.number}, attempting to stream session")
                    
                    # Refresh websocket from pending info (client may have reconnected)
                    info = pending_copilot_issues.get(issue_number)
                    if info and info.get('websocket'):
                        websocket = info['websocket']
                        client_id = info.get('client_id', client_id)
                    
                    # Remove from pending (done polling, about to stream)
                    if issue_number in pending_copilot_issues:
                        del pending_copilot_issues[issue_number]
                    
                    # Notify client session is starting (tolerate disconnects)
                    await _safe_send({
                        'type': 'copilot_session_starting',
                        'pr_number': found_pr.number,
                        'message': f'Agent session found, streaming logs...',
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Start streaming - pass PR number, stream function will extract session ID
                    # skip_wait=True because poll already confirmed the session exists
                    await stream_copilot_session_logs(
                        websocket, 
                        client_id, 
                        str(found_pr.number),  # Pass PR number as string
                        pr_number=found_pr.number,  # Explicitly pass PR number for database
                        skip_wait=True
                    )
                    return
                    
                    # Session not ready, wait shorter interval
                    await asyncio.sleep(session_poll_interval)
                    continue
                
            except Exception as e:
                logger.warning(f"Error polling for Copilot agent task: {e}")
            
            await asyncio.sleep(poll_interval)
        
        # Timeout reached
        logger.info(f"Stopped polling for issue #{issue_number} - timeout reached")
        if issue_number in pending_copilot_issues:
            del pending_copilot_issues[issue_number]
        
        await _safe_send({
            'type': 'copilot_watch_timeout',
            'issue_number': issue_number,
            'message': f'Timed out waiting for Copilot to create a PR for issue #{issue_number}',
            'timestamp': datetime.now().isoformat()
        })
        
    except asyncio.CancelledError:
        logger.info(f"Copilot polling cancelled for issue #{issue_number}")
        if issue_number in pending_copilot_issues:
            del pending_copilot_issues[issue_number]
        raise
    except Exception as e:
        logger.error(f"Error in poll_for_copilot_session: {e}")
        if issue_number in pending_copilot_issues:
            del pending_copilot_issues[issue_number]


def fetch_open_prs():
    """
    Fetch open pull requests from GitHub repository.
    Returns PRs with their branch names and linked issues.
    
    Returns:
        List of PR dictionaries with number, title, branch, and mentioned issues
    """
    if not GITHUB_TOKEN:
        logger.warning("GitHub token not configured, cannot fetch PRs")
        return []
    
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        
        # Fetch open PRs (limit to 30 most recent)
        pulls = repo.get_pulls(state='open', sort='updated', direction='desc')
        
        pr_list = []
        for pr in pulls[:30]:  # Limit to 30 PRs
            # Extract mentioned issue numbers from PR text
            import re
            pr_text = f"{pr.title} {pr.body or ''}"
            mentioned_issues = re.findall(r'#(\d+)', pr_text)
            
            pr_list.append({
                'number': pr.number,
                'title': pr.title,
                'branch': pr.head.ref,
                'state': pr.state,
                'mentioned_issues': mentioned_issues,
                'created_at': pr.created_at.isoformat(),
                'updated_at': pr.updated_at.isoformat()
            })
        
        logger.info(f"Fetched {len(pr_list)} open PRs from GitHub")
        return pr_list
        
    except Exception as e:
        logger.error(f"Failed to fetch PRs: {e}")
        return []


def fetch_open_issues():
    """
    Fetch open issues from GitHub repository.
    Uses caching to avoid excessive API calls.
    
    Returns:
        List of issue dictionaries with number, title, and labels
    """
    global issue_cache
    
    if not GITHUB_TOKEN:
        logger.warning("GitHub token not configured, cannot fetch issues")
        return []
    
    # Check cache
    now = datetime.now()
    if (issue_cache['last_fetch'] and 
        (now - issue_cache['last_fetch']).total_seconds() < issue_cache['cache_duration'] and
        issue_cache['issues']):
        logger.info(f"Returning {len(issue_cache['issues'])} cached issues")
        return issue_cache['issues']
    
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        
        # Fetch open issues (limit to 50 most recent)
        issues = repo.get_issues(state='open', sort='updated', direction='desc')
        
        issue_list = []
        for issue in issues[:50]:  # Limit to 50 issues
            issue_list.append({
                'number': issue.number,
                'title': issue.title,
                'labels': [label.name for label in issue.labels],
                'created_at': issue.created_at.isoformat(),
                'updated_at': issue.updated_at.isoformat()
            })
        
        # Update cache
        issue_cache['issues'] = issue_list
        issue_cache['last_fetch'] = now
        
        logger.info(f"Fetched {len(issue_list)} open issues from GitHub")
        return issue_list
        
    except Exception as e:
        logger.error(f"Failed to fetch issues: {e}")
        return []


def fetch_pr_title(pr_number: int) -> str:
    """
    Fetch the title of a pull request.
    
    Args:
        pr_number: GitHub pull request number
        
    Returns:
        PR title string, or f"PR #{pr_number}" if fetch fails
    """
    if not GITHUB_TOKEN:
        return f"PR #{pr_number}"
    
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        pr = repo.get_pull(pr_number)
        return pr.title
    except Exception as e:
        logger.error(f"Failed to fetch title for PR #{pr_number}: {e}")
        return f"PR #{pr_number}"


def fetch_pr_tools(pr_number: int) -> list:
    """
    Fetch tools/code files from a specific pull request.
    Prioritizes local tools over stale PR tools when available.
    
    Args:
        pr_number: GitHub pull request number
        
    Returns:
        List of tool dictionaries with name, path, description, code, and metadata
    """
    if not GITHUB_TOKEN:
        logger.warning("GitHub token not configured, cannot fetch PR tools")
        return []
    
    # First, get local tools as a baseline
    local_tools = get_local_tools_for_pr_merge()
    
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        pr = repo.get_pull(pr_number)
        
        logger.info(f"Fetching tools from PR #{pr_number}: {pr.title}")
        logger.info(f"PR #{pr_number} branch: {pr.head.ref}, commit SHA: {pr.head.sha}")
        
        # Get PR tools
        pr_tools = fetch_pr_tools_from_github(pr, repo)
        
        # Merge with priority: local tools override PR tools for freshness
        merged_tools = merge_local_and_pr_tools(local_tools, pr_tools, pr_number)
        
        return merged_tools
        
    except Exception as e:
        logger.error(f"Failed to fetch tools for PR #{pr_number}: {e}")
        # Fall back to local tools if PR fetch fails
        logger.info(f"Falling back to local tools for PR #{pr_number}")
        return local_tools


def get_local_tools_for_pr_merge() -> list:
    """Get local tools that can be used for PR merging"""
    local_tools = []
    local_tools_dir = Path(__file__).parent.parent / 'tools'
    
    if not local_tools_dir.exists():
        return []
    
    for py_file in local_tools_dir.glob('*.py'):
        if py_file.name.startswith('__') or py_file.name.startswith('test_'):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                code = f.read()
                
            # Extract description from docstring
            description = "Local tool"
            if code.strip().startswith('"""') or code.strip().startswith("'''"):
                docstring_lines = []
                in_docstring = False
                quote_type = None
                
                for line in code.split('\n'):
                    line = line.strip()
                    if not in_docstring:
                        if line.startswith('"""') or line.startswith("'''"):
                            quote_type = '"""' if line.startswith('"""') else "'''"
                            in_docstring = True
                            docstring_lines.append(line[3:])
                    else:
                        if line.endswith(quote_type):
                            docstring_lines.append(line[:-3])
                            break
                        else:
                            docstring_lines.append(line)
                
                if docstring_lines:
                    description = ' '.join(docstring_lines).strip()
                    if len(description) > 200:
                        description = description[:200] + "..."
            
            tool_name = py_file.stem
            local_tools.append({
                'name': tool_name,
                'path': f'tools/{py_file.name}',
                'description': description,
                'code': code,
                'language': 'python',
                'source': 'local'
            })
            
        except Exception as e:
            logger.warning(f"Failed to read local tool {py_file}: {e}")
    
    logger.info(f"Found {len(local_tools)} local tools for PR merging")
    return local_tools


def fetch_pr_tools_from_github(pr, repo) -> list:
    """
    Fetch tool code directly from GitHub PR branch.
    """
    tools = []
    
    # Parse custom_gpt fields from the linked issue (if any)
    pr_custom_gpt = False
    pr_gpt_query = ''
    pr_system_instruction = ''
    pr_query_interval = 3.0
    try:
        pr_text = f"{pr.title or ''} {pr.body or ''}"
        issue_ref_match = re.search(r'#(\d+)', pr_text)
        if issue_ref_match:
            linked_issue = repo.get_issue(int(issue_ref_match.group(1)))
            linked_body = linked_issue.body or ''
            
            cgpt_match = re.search(r'\*\*Custom GPT\*\*\s*(?:<!--.*?-->)?\s*(.+?)(?:\n\n|\n\*\*|$)', linked_body, re.IGNORECASE | re.DOTALL)
            if cgpt_match:
                pr_custom_gpt = cgpt_match.group(1).strip().lower() in ('yes', 'true', '1')
            
            gq_match = re.search(r'\*\*GPT Query\*\*\s*(?:<!--.*?-->)?\s*(.+?)(?:\n\n|\n\*\*|$)', linked_body, re.IGNORECASE | re.DOTALL)
            pr_gpt_query = gq_match.group(1).strip() if gq_match else ''
            
            si_match = re.search(r'\*\*System Instruction\*\*\s*(?:<!--.*?-->)?\s*(.+?)(?:\n\n|\n\*\*|$)', linked_body, re.IGNORECASE | re.DOTALL)
            pr_system_instruction = si_match.group(1).strip() if si_match else ''
            
            qi_match = re.search(r'\*\*Query Interval\*\*\s*(?:<!--.*?-->)?\s*(\d+\.?\d*)', linked_body, re.IGNORECASE)
            pr_query_interval = float(qi_match.group(1)) if qi_match else 3.0
            
            if pr_custom_gpt:
                logger.info(f"PR #{pr.number} linked to custom GPT issue: gpt_query='{pr_gpt_query[:60]}'")
    except Exception as e:
        logger.debug(f"Could not parse custom_gpt fields from PR #{pr.number} linked issue: {e}")
    
    # Get the tree of the PR's head commit
    commit = repo.get_commit(pr.head.sha)
    tree = commit.commit.tree
    logger.info(f"Successfully fetched tree for PR #{pr.number}, tree SHA: {tree.sha}")
    
    # Recursively traverse the tree to find files in tools/ directory
    def find_tool_files(tree_obj, current_path=""):
        """Recursively find tool files in the tree"""
        tool_files = []
        
        for element in tree_obj.tree:
            element_path = f"{current_path}/{element.path}" if current_path else element.path
            
            if element.type == 'tree':
                # Explore tools directories
                should_explore = (
                    element.path == 'tools' or
                    element_path.endswith('/tools') or
                    current_path.startswith('tools') or
                    current_path.startswith('backend/tools') or
                    current_path.startswith('ProgramATApp/tools')
                )
                
                if should_explore:
                    try:
                        logger.info(f"Exploring directory: {element_path}")
                        subtree = repo.get_git_tree(element.sha)
                        tool_files.extend(find_tool_files(subtree, element_path))
                    except Exception as e:
                        logger.warning(f"Failed to traverse {element_path}: {e}")
            
            elif element.type == 'blob':
                # Check if this file is in a tools directory
                is_in_tools = (
                    current_path == 'tools' or
                    current_path.startswith('tools/') or
                    current_path.endswith('/tools') or
                    '/tools/' in current_path
                )
                
                if is_in_tools and element_path.endswith(('.py', '.js', '.ts', '.tsx', '.jsx')):
                    tool_files.append({
                        'path': element_path,
                        'sha': element.sha
                    })
                    logger.info(f"Found PR tool file: {element_path}")
        
        return tool_files
    
    # Find all tool files in this PR's branch
    tool_files = find_tool_files(tree)
    logger.info(f"Found {len(tool_files)} tool files in PR #{pr.number}")
    
    # Fetch content for each tool file
    for file_info in tool_files:
        try:
            blob = repo.get_git_blob(file_info['sha'])
            if blob.encoding == 'base64':
                import base64
                code = base64.b64decode(blob.content).decode('utf-8')
            else:
                code = blob.content
            
            # Extract tool name and description
            tool_name = Path(file_info['path']).stem
            description = f"Tool from PR #{pr.number}: {pr.title}"
            
            # Try to extract description from docstring
            if code.strip().startswith('"""') or code.strip().startswith("'''"):
                lines = code.split('\n')
                for i, line in enumerate(lines):
                    if i == 0 and ('"""' in line or "'''" in line):
                        quote_type = '"""' if '"""' in line else "'''"
                        desc_lines = [line.replace(quote_type, '')]
                        for j in range(i + 1, len(lines)):
                            if quote_type in lines[j]:
                                desc_lines.append(lines[j].replace(quote_type, ''))
                                break
                            desc_lines.append(lines[j])
                        
                        if desc_lines:
                            description = ' '.join(desc_lines).strip()[:200]
                        break
            
            # Determine language
            file_ext = file_info['path'].split('.')[-1]
            language_map = {
                'py': 'python',
                'js': 'javascript',
                'ts': 'typescript',
                'tsx': 'typescript',
                'jsx': 'javascript'
            }
            language = language_map.get(file_ext, file_ext)
            
            tools.append({
                'name': tool_name,
                'path': file_info['path'],
                'description': description,
                'code': code,
                'language': language,
                'pr_number': pr.number,
                'pr_title': pr.title,
                'branch_name': pr.head.ref,
                'source': 'github_pr',
                'custom_gpt': pr_custom_gpt,
                'gpt_query': pr_gpt_query,
                'system_instruction': pr_system_instruction,
                'query_interval': pr_query_interval,
            })
            logger.info(f"Added PR tool: {tool_name} from {file_info['path']} (PR #{pr.number})")
        except Exception as e:
            logger.warning(f"Failed to fetch content for {file_info['path']}: {e}")
    
    return tools


def merge_local_and_pr_tools(local_tools: list, pr_tools: list, pr_number: int) -> list:
    """
    Merge local tools and PR tools, prioritizing PR tools over local tools.
    """
    merged_tools = []
    
    # Create lookup maps
    local_by_name = {tool['name']: tool for tool in local_tools}
    pr_by_name = {tool['name']: tool for tool in pr_tools}
    
    # Get all unique tool names
    all_tool_names = set(local_by_name.keys()) | set(pr_by_name.keys())
    
    stale_warnings = []
    
    for tool_name in all_tool_names:
        local_tool = local_by_name.get(tool_name)
        pr_tool = pr_by_name.get(tool_name)
        
        # Special case: skip camera_aiming_assistant (local) if camera_aiming (PR) exists
        if tool_name == 'camera_aiming_assistant' and 'camera_aiming' in pr_by_name:
            # Skip local camera_aiming_assistant if PR has camera_aiming
            logger.info(f"⏭️  Skipping local 'camera_aiming_assistant' (PR has preferred 'camera_aiming')")
            continue
        
        if local_tool and pr_tool:
            # Both exist - ALWAYS USE PR TOOL (as requested)
            if local_tool['name'] != pr_tool['name']:
                stale_warnings.append(f"Tool name mismatch: local='{local_tool['name']}' vs PR='{pr_tool['name']}'")
            
            # Use PR tool (preferred behavior)
            merged_tools.append(pr_tool)
            logger.info(f"🎯 Using PR tool '{tool_name}' (PR preferred over local)")
            
        elif local_tool:
            # Only local exists - use local but mark as fallback
            merged_tool = local_tool.copy()
            merged_tool.update({
                'pr_number': pr_number,
                'pr_title': f'PR #{pr_number}',
                'branch_name': 'local',
                'source': 'local_only'
            })
            merged_tools.append(merged_tool)
            logger.info(f"📁 Using LOCAL-only tool '{tool_name}' (no PR version available)")
            
        elif pr_tool:
            # Only PR exists - use it
            merged_tools.append(pr_tool)
            logger.info(f"📥 Using PR-only tool '{tool_name}'")
    
    # Check for known stale patterns
    pr_names = set(pr_by_name.keys())
    local_names = set(local_by_name.keys())
    
    # Note: PR tools are now preferred over local tools
    if 'camera_aiming' in pr_names and 'camera_aiming_assistant' in local_names:
        logger.info(f"🎯 PR #{pr_number} has 'camera_aiming' tool (preferred over local 'camera_aiming_assistant')")
    elif 'camera_aiming_assistant' in pr_names and 'camera_aiming' in local_names:
        stale_warnings.append("PR contains outdated 'camera_aiming_assistant', should use 'camera_aiming' instead")
    
    # Log stale warnings
    if stale_warnings:
        logger.warning(f"🚨 STALE TOOL DETECTION for PR #{pr_number}:")
        for warning in stale_warnings:
            logger.warning(f"   - {warning}")
        logger.warning(f"   💡 Consider updating PR #{pr_number} branch to latest commits")
    
    logger.info(f"✅ Merged {len(merged_tools)} tools for PR #{pr_number} (prioritizing PR versions)")
    return merged_tools


def fetch_branch_tools(branch_name: str = 'main') -> list:
    """
    Fetch tools/code files from a specific branch (for production mode).
    
    Args:
        branch_name: GitHub branch name (default: 'main')
        
    Returns:
        List of tool dictionaries with name, path, description, code, and metadata
    """
    if not GITHUB_TOKEN:
        logger.warning("GitHub token not configured, cannot fetch branch tools")
        return []
    
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        
        logger.info(f"Fetching production tools from branch: {branch_name}")
        
        # Get the branch
        branch = repo.get_branch(branch_name)
        commit_sha = branch.commit.sha
        
        logger.info(f"Branch {branch_name} commit SHA: {commit_sha}")
        
        tools = []
        
        # Get the tree of the branch's head commit
        commit = repo.get_commit(commit_sha)
        tree = commit.commit.tree
        logger.info(f"Successfully fetched tree for branch {branch_name}, tree SHA: {tree.sha}")
        
        # Recursively traverse the tree to find files in tools/ directory
        def find_tool_files(tree_obj, current_path=""):
            """Recursively find tool files in the tree"""
            tool_files = []
            
            for element in tree_obj.tree:
                element_path = f"{current_path}/{element.path}" if current_path else element.path
                
                if element.type == 'tree':
                    # Explore tools directories
                    should_explore = (
                        element.path == 'tools' or
                        element_path.endswith('/tools') or
                        current_path.startswith('tools') or
                        current_path.startswith('backend/tools') or
                        current_path.startswith('ProgramATApp/tools')
                    )
                    
                    if should_explore:
                        try:
                            logger.info(f"Exploring directory: {element_path}")
                            subtree = repo.get_git_tree(element.sha)
                            tool_files.extend(find_tool_files(subtree, element_path))
                        except Exception as e:
                            logger.warning(f"Failed to traverse {element_path}: {e}")
                
                elif element.type == 'blob':
                    # Check if this file is in a tools directory
                    is_in_tools = (
                        current_path == 'tools' or
                        current_path.startswith('tools/') or
                        current_path.endswith('/tools') or
                        '/tools/' in current_path
                    )
                    
                    if is_in_tools and element_path.endswith(('.py', '.js', '.ts', '.tsx', '.jsx')):
                        tool_files.append({
                            'path': element_path,
                            'sha': element.sha
                        })
                        logger.info(f"Found tool file: {element_path}")
            
            return tool_files
        
        # Find all tool files in this branch
        tool_files = find_tool_files(tree)
        logger.info(f"Found {len(tool_files)} tool files in branch {branch_name}")
        
        # Fetch content for each tool file
        for file_info in tool_files:
            try:
                # Fetch file content from GitHub using the blob SHA
                blob = repo.get_git_blob(file_info['sha'])
                import base64
                code = base64.b64decode(blob.content).decode('utf-8')
                
                # Extract tool name from filename
                tool_name = file_info['path'].split('/')[-1].rsplit('.', 1)[0]
                
                # Try to extract description from first docstring/comment
                description = extract_tool_description(code)
                
                # Normalize language extension to full name
                file_ext = file_info['path'].split('.')[-1]
                language_map = {
                    'py': 'python',
                    'js': 'javascript',
                    'ts': 'typescript',
                    'tsx': 'typescript',
                    'jsx': 'javascript'
                }
                language = language_map.get(file_ext, file_ext)
                
                tools.append({
                    'name': tool_name,
                    'path': file_info['path'],
                    'description': description,
                    'code': code,
                    'language': language,
                    'branch_name': branch_name,
                    'is_production': True
                })
                logger.info(f"Added production tool: {tool_name} from {file_info['path']} (branch {branch_name})")
            except Exception as e:
                logger.warning(f"Failed to fetch content for {file_info['path']}: {e}")
        
        logger.info(f"Found {len(tools)} production tools from branch {branch_name}")
        return tools
        
    except Exception as e:
        logger.error(f"Failed to fetch tools from branch {branch_name}: {e}")
        return []


def fetch_issue_tools(issue_number: int) -> list:
    """
    Fetch tools/code files associated with an issue.
    Looks for PRs linked to the issue (by mention, branch name, or recent activity)
    and extracts Python/JS/TS files from tools/ directory.
    
    Fetches files from PR's branch via GitHub API - no local branch switching needed.
    
    Args:
        issue_number: GitHub issue number
        
    Returns:
        List of tool dictionaries with name, path, description, and code
    """
    if not GITHUB_TOKEN:
        logger.warning("GitHub token not configured, cannot fetch issue tools")
        return []
    
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        issue = repo.get_issue(issue_number)
        
        # Parse custom_gpt / Gemini Live fields from issue body
        issue_body = issue.body or ''
        custom_gpt_match = re.search(r'\*\*Custom GPT\*\*\s*(?:<!--.*?-->)?\s*(.+?)(?:\n\n|\n\*\*|$)', issue_body, re.IGNORECASE | re.DOTALL)
        issue_custom_gpt = False
        if custom_gpt_match:
            custom_gpt_value = custom_gpt_match.group(1).strip().lower()
            issue_custom_gpt = custom_gpt_value in ('yes', 'true', '1')
        
        gpt_query_match = re.search(r'\*\*GPT Query\*\*\s*(?:<!--.*?-->)?\s*(.+?)(?:\n\n|\n\*\*|$)', issue_body, re.IGNORECASE | re.DOTALL)
        issue_gpt_query = gpt_query_match.group(1).strip() if gpt_query_match else ''
        
        si_match = re.search(r'\*\*System Instruction\*\*\s*(?:<!--.*?-->)?\s*(.+?)(?:\n\n|\n\*\*|$)', issue_body, re.IGNORECASE | re.DOTALL)
        issue_system_instruction = si_match.group(1).strip() if si_match else ''
        
        qi_match = re.search(r'\*\*Query Interval\*\*\s*(?:<!--.*?-->)?\s*(\d+\.?\d*)', issue_body, re.IGNORECASE)
        issue_query_interval = float(qi_match.group(1)) if qi_match else 3.0
        
        logger.info(f"Issue #{issue_number} custom_gpt={issue_custom_gpt}, gpt_query='{issue_gpt_query[:60]}'")
        
        tools = []
        matched_prs = []
        
        # Get all pull requests (open and recently closed)
        pulls = repo.get_pulls(state='all', sort='updated', direction='desc')
        
        # Strategy 1: Find PRs that explicitly mention this issue
        for pr in pulls[:50]:  # Limit to 50 most recent
            pr_text = f"{pr.title} {pr.body or ''}"
            if f"#{issue_number}" in pr_text or f"issue {issue_number}" in pr_text.lower():
                # Extract all issue numbers mentioned in this PR for logging
                mentioned_issues = re.findall(r'#(\d+)', pr_text)
                logger.info(f"Found PR #{pr.number} explicitly linked to issue #{issue_number} (state: {pr.state}, mentions: {mentioned_issues})")
                matched_prs.append(pr)
        
        # Strategy 2: If no explicit mentions, look for PRs with branch names containing the issue number
        if not matched_prs:
            for pr in pulls[:50]:
                branch_name = pr.head.ref.lower()
                issue_str = str(issue_number)
                # Common patterns: issue-42, fix-42, feature/42, 42-description, etc.
                if (f"issue-{issue_str}" in branch_name or 
                    f"issue/{issue_str}" in branch_name or
                    f"fix-{issue_str}" in branch_name or
                    f"fix/{issue_str}" in branch_name or
                    f"feature-{issue_str}" in branch_name or
                    f"feature/{issue_str}" in branch_name or
                    f"{issue_str}-" in branch_name or
                    f"/{issue_str}-" in branch_name or
                    f"-{issue_str}" in branch_name or
                    branch_name.endswith(f"/{issue_str}")):
                    logger.info(f"Found PR #{pr.number} with branch '{pr.head.ref}' matching issue #{issue_number} (state: {pr.state})")
                    matched_prs.append(pr)
        
        # If multiple PRs found, prioritize by:
        # 1. Open PRs over closed ones
        # 2. Most recently updated
        if len(matched_prs) > 1:
            logger.info(f"Found {len(matched_prs)} PRs for issue #{issue_number}, prioritizing...")
            # Separate open and closed PRs
            open_prs = [pr for pr in matched_prs if pr.state == 'open']
            closed_prs = [pr for pr in matched_prs if pr.state == 'closed']
            
            # Prefer open PRs, most recent first
            if open_prs:
                matched_prs = sorted(open_prs, key=lambda pr: pr.updated_at, reverse=True)
                logger.info(f"Using most recent open PR #{matched_prs[0].number}: {matched_prs[0].title}")
            else:
                # All closed, use most recent
                matched_prs = sorted(closed_prs, key=lambda pr: pr.updated_at, reverse=True)
                logger.info(f"All PRs closed, using most recent PR #{matched_prs[0].number}: {matched_prs[0].title}")
        
        # If no matches found, return empty list (or synthetic custom GPT tool)
        if not matched_prs:
            logger.warning(f"No PRs found for issue #{issue_number}. Please ensure PR mentions #{issue_number} or has issue number in branch name.")
            
            # For custom GPT tools, no PR/code is needed — the gpt_query IS the tool
            if issue_custom_gpt and issue_gpt_query:
                logger.info(f"Custom GPT issue #{issue_number} has no PR — creating synthetic tool from gpt_query")
                issue_title = issue.title or f'Issue #{issue_number}'
                # Derive tool name from issue title
                tool_name = re.sub(r'[^a-z0-9]+', '_', issue_title.lower()).strip('_')[:40]
                tools.append({
                    'name': tool_name,
                    'path': f'custom_gpt/{tool_name}.py',
                    'description': issue_title,
                    'code': '# Custom GPT tool - uses Gemini Live, no code execution\ndef main(image, input_data):\n    return {"success": True, "text": "This tool uses Gemini Live mode"}',
                    'language': 'python',
                    'pr_number': None,
                    'pr_title': None,
                    'branch_name': None,
                    'issue_number': issue_number,
                    'custom_gpt': True,
                    'gpt_query': issue_gpt_query,
                    'system_instruction': issue_system_instruction,
                    'query_interval': issue_query_interval,
                })
                return tools
        
        # Extract tools from the best matched PR (only use the first one)
        for pr in matched_prs[:1]:  # Only use the best match
            logger.info(f"Extracting tools from PR #{pr.number}: {pr.title}")
            logger.info(f"PR #{pr.number} branch: {pr.head.ref}, commit SHA: {pr.head.sha}")
            
            try:
                # Get the tree of the PR's head commit (fetches from remote branch)
                commit = repo.get_commit(pr.head.sha)
                tree = commit.commit.tree
                logger.info(f"Successfully fetched tree for PR #{pr.number}, tree SHA: {tree.sha}")
                
                # Recursively traverse the tree to find files in tools/ directory
                def find_tool_files(tree_obj, current_path=""):
                    """Recursively find tool files in the tree"""
                    tool_files = []
                    
                    # Debug: log what we see at the root level
                    if current_path == "":
                        logger.info(f"Root tree contains: {[element.path for element in tree_obj.tree]}")
                    
                    for element in tree_obj.tree:
                        element_path = f"{current_path}/{element.path}" if current_path else element.path
                        
                        # If this element is a directory, check if we should explore it
                        if element.type == 'tree':
                            # Always explore these directories
                            should_explore = (
                                element.path == 'tools' or  # Root-level tools directory
                                element_path.endswith('/tools') or  # tools subdirectory
                                current_path.startswith('tools') or  # Already inside tools
                                current_path.startswith('backend/tools') or
                                current_path.startswith('ProgramATApp/tools')
                            )
                            
                            if should_explore:
                                try:
                                    logger.info(f"Exploring directory: {element_path}")
                                    subtree = repo.get_git_tree(element.sha)
                                    tool_files.extend(find_tool_files(subtree, element_path))
                                except Exception as e:
                                    logger.warning(f"Failed to traverse {element_path}: {e}")
                        
                        # If it's a file in a tools directory
                        elif element.type == 'blob':
                            # Check if this file is in a tools directory
                            is_in_tools = (
                                current_path == 'tools' or
                                current_path.startswith('tools/') or
                                current_path.endswith('/tools') or
                                '/tools/' in current_path
                            )
                            
                            if is_in_tools and element_path.endswith(('.py', '.js', '.ts', '.tsx', '.jsx')):
                                tool_files.append({
                                    'path': element_path,
                                    'sha': element.sha
                                })
                                logger.info(f"Found tool file: {element_path}")
                    
                    return tool_files
                
                # Find all tool files in this PR's branch
                tool_files = find_tool_files(tree)
                logger.info(f"Found {len(tool_files)} tool files in PR #{pr.number}")
                
                # If no tools found in tools/ directories, check what files were actually changed in the PR
                if len(tool_files) == 0:
                    logger.info(f"No tools/ directory found, checking PR changed files instead")
                    try:
                        pr_files = pr.get_files()
                        for pr_file in pr_files:
                            # Look for code files that might be tools (not in tests, docs, configs)
                            if (pr_file.filename.endswith(('.py', '.js', '.ts', '.tsx', '.jsx')) and
                                pr_file.status != 'removed' and
                                not pr_file.filename.startswith('.') and
                                '/test' not in pr_file.filename.lower() and
                                '/__tests__/' not in pr_file.filename and
                                not pr_file.filename.endswith(('.test.ts', '.test.tsx', '.test.js', '.test.jsx'))):
                                tool_files.append({
                                    'path': pr_file.filename,
                                    'sha': pr_file.sha
                                })
                                logger.info(f"Found changed code file: {pr_file.filename}")
                    except Exception as e:
                        logger.warning(f"Failed to get PR files: {e}")
                
                logger.info(f"Total files to process: {len(tool_files)}")
                
                # Fetch content for each tool file
                for file_info in tool_files:
                    try:
                        # Fetch file content from GitHub using the blob SHA
                        blob = repo.get_git_blob(file_info['sha'])
                        # GitBlob.content is base64 encoded, need to decode it
                        import base64
                        code = base64.b64decode(blob.content).decode('utf-8')
                        
                        # Extract tool name from filename
                        tool_name = file_info['path'].split('/')[-1].rsplit('.', 1)[0]
                        
                        # Try to extract description from first docstring/comment
                        description = extract_tool_description(code)
                        
                        # Normalize language extension to full name
                        file_ext = file_info['path'].split('.')[-1]
                        language_map = {
                            'py': 'python',
                            'js': 'javascript',
                            'ts': 'typescript',
                            'tsx': 'typescript',
                            'jsx': 'javascript'
                        }
                        language = language_map.get(file_ext, file_ext)
                        
                        tools.append({
                            'name': tool_name,
                            'path': file_info['path'],
                            'description': description,
                            'code': code,
                            'language': language,
                            'pr_number': pr.number,
                            'pr_title': pr.title,
                            'branch_name': pr.head.ref,
                            'issue_number': issue_number,
                            'custom_gpt': issue_custom_gpt,
                            'gpt_query': issue_gpt_query,
                            'system_instruction': issue_system_instruction,
                            'query_interval': issue_query_interval,
                        })
                        logger.info(f"Added tool: {tool_name} from {file_info['path']} (PR #{pr.number}, branch {pr.head.ref})")
                    except Exception as e:
                        logger.warning(f"Failed to fetch content for {file_info['path']}: {e}")
                        
            except Exception as e:
                logger.error(f"Failed to process PR #{pr.number}: {e}")
        
        logger.info(f"Found {len(tools)} tools for issue #{issue_number}")
        return tools
        
    except Exception as e:
        logger.error(f"Failed to fetch tools for issue #{issue_number}: {e}")
        return []


def extract_tool_description(code: str) -> str:
    """
    Extract description from code file (first docstring or comment block).
    
    Args:
        code: Source code content
        
    Returns:
        Description string or empty string
    """
    lines = code.split('\n')
    description_lines = []
    in_docstring = False
    
    for line in lines[:20]:  # Only check first 20 lines
        stripped = line.strip()
        
        # Python docstring
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if in_docstring:
                break  # End of docstring
            in_docstring = True
            # Extract text after opening quotes
            desc = stripped[3:].strip()
            if desc.endswith('"""') or desc.endswith("'''"):
                return desc[:-3].strip()
            if desc:
                description_lines.append(desc)
            continue
        
        if in_docstring:
            if stripped.endswith('"""') or stripped.endswith("'''"):
                description_lines.append(stripped[:-3].strip())
                break
            description_lines.append(stripped)
            continue
        
        # Single-line comment
        if stripped.startswith('#') or stripped.startswith('//'):
            desc = stripped[1:].strip() if stripped.startswith('#') else stripped[2:].strip()
            if desc:
                description_lines.append(desc)
        
        # Stop at first non-comment line
        if stripped and not stripped.startswith(('#', '//', '"""', "'''")):
            break
    
    return ' '.join(description_lines)[:200]  # Limit to 200 chars


def parse_issue_selection(transcript: str, available_issues: list) -> dict:
    """
    Use AI to parse voice command to select an issue or switch to create mode.
    
    Args:
        transcript: Voice transcript
        available_issues: List of available issue dicts
        
    Returns:
        Dictionary with 'mode' ('create' or 'update'), 'issue_number', and 'issue_title'
    """
    if not GEMINI_API_KEY or not available_issues:
        return {'mode': 'create', 'issue_number': None, 'issue_title': None}
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Build list of available issues for the prompt
        issue_list_str = "\n".join([f"- Issue #{issue['number']}: {issue['title']}" 
                                     for issue in available_issues])
        
        prompt = f"""Parse the following voice command to determine if the user wants to:
1. Select an existing issue to update/iterate on
2. Create a new issue
3. Show/list available issues

Available open issues:
{issue_list_str}

Voice command: {transcript}

If the user is selecting an issue (e.g., "select issue 42", "work on the bug about login", "update the camera issue"):
- Return mode: "update"
- Return the issue number that best matches their description
- Return the issue title

If the user wants to create a new issue or doesn't mention an existing issue:
- Return mode: "create"

If the user wants to see the list of issues (e.g., "show issues", "list issues", "what issues are open"):
- Return mode: "list"

Return ONLY a valid JSON object:
{{
  "mode": "create" | "update" | "list",
  "issue_number": <number or null>,
  "issue_title": "<title or null>",
  "confidence": <0.0 to 1.0>
}}"""

        response = model.generate_content(prompt)
        ai_response = response.text.strip()
        
        # Remove markdown code blocks if present
        if ai_response.startswith('```json'):
            ai_response = ai_response[7:]
        if ai_response.startswith('```'):
            ai_response = ai_response[3:]
        if ai_response.endswith('```'):
            ai_response = ai_response[:-3]
        ai_response = ai_response.strip()
        
        result = json.loads(ai_response)
        logger.info(f"Issue selection parsed: mode={result.get('mode')}, issue={result.get('issue_number')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to parse issue selection: {e}")
        return {'mode': 'create', 'issue_number': None, 'issue_title': None}


def should_mention_copilot(text: str) -> bool:
    """
    Determine if the text is asking for code changes (should mention @copilot)
    vs just status updates or issue selection (should not mention @copilot).
    
    Args:
        text: The comment text to analyze
        
    Returns:
        True if text appears to be requesting code changes, False otherwise
    """
    text_lower = text.lower()
    
    # Keywords that indicate code change requests
    code_change_keywords = [
        'fix', 'change', 'update', 'modify', 'add', 'remove', 'implement', 
        'create', 'delete', 'refactor', 'improve', 'optimize',
        'bug', 'error', 'problem',
        'make', 'set', 'adjust', 'replace', 'rewrite'
    ]
    
    # Keywords that indicate status updates (should NOT mention copilot)
    status_keywords = [
        'switch', 'select', 'go to', 'open', 'view', 'show',
        'switching to', 'selected issue', 'working on'
    ]
    
    # Check for status keywords first - these take precedence
    for keyword in status_keywords:
        if keyword in text_lower:
            return False
    
    # Check for code change keywords
    for keyword in code_change_keywords:
        if keyword in text_lower:
            return True
    
    # Default to mentioning copilot if unsure (better to over-mention than under-mention)
    return True


async def update_github_issue(issue_number: int, comment_text: str, mention_copilot: bool = True):
    """
    Add a comment to an existing GitHub issue or PR.
    Optionally mentions @copilot in the comment to get Copilot's attention.
    
    Args:
        issue_number: The issue or PR number to update
        comment_text: The comment to add
        mention_copilot: Whether to add @copilot mention (True for code changes, False for status updates)
    """
    _log_to_all_sessions("INFO", f"update_github_issue called: issue/PR #{issue_number}, comment: {comment_text}, mention_copilot: {mention_copilot}")
    
    if not GITHUB_TOKEN:
        _log_to_all_sessions("WARNING", "GitHub token not configured, cannot update issue")
        logger.warning("GitHub token not configured, cannot update issue")
        return
    
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        issue = repo.get_issue(issue_number)
        
        # Append @copilot to the comment if requested (for code change requests)
        final_comment = f"{comment_text}\n\n@copilot" if mention_copilot else comment_text
        
        # Add comment to the issue/PR
        issue.create_comment(final_comment)
        _log_to_all_sessions("INFO", f"GitHub API: Added comment to #{issue_number}" + (" with @copilot mention" if mention_copilot else ""))
        logger.info(f"Added comment to #{issue_number}" + (" with @copilot mention" if mention_copilot else ""))
        
        # Check if this is a PR (pull_request attribute exists) or an issue
        is_pr = issue.pull_request is not None
        pr_number = issue_number if is_pr else None
        pr_url = issue.html_url if is_pr else None
        
        # If it's an issue (not a PR), try to find an associated PR
        if not is_pr:
            try:
                # Search for PR that references this issue
                pulls = repo.get_pulls(state='open')
                for pr in pulls:
                    # Check if PR body or title references this issue
                    if pr.body and f"#{issue_number}" in pr.body:
                        pr_number = pr.number
                        pr.create_issue_comment(final_comment)
                        pr_url = pr.html_url
                        _log_to_all_sessions("INFO", f"GitHub API: Added comment to associated PR #{pr_number}" + (" with @copilot mention" if mention_copilot else ""))
                        logger.info(f"Added comment to associated PR #{pr_number}" + (" with @copilot mention" if mention_copilot else ""))
                        break
            except Exception as e:
                _log_to_all_sessions("WARNING", f"Could not add comment to PR for issue #{issue_number}: {e}")
                logger.warning(f"Could not add comment to PR for issue #{issue_number}: {e}")
        
        # Send success notification to client
        if connected_clients:
            success_data = {
                'type': 'issue_updated',
                'message': f"Comment added to issue #{issue_number}" + (f" and PR #{pr_number}" if pr_number else ""),
                'issue_number': issue_number,
                'issue_url': issue.html_url,
                'pr_number': pr_number,
                'pr_url': pr_url
            }
            import websockets
            websockets.broadcast(connected_clients, json.dumps(success_data))
            
            # If we mentioned @copilot, start polling for the Copilot session
            if mention_copilot and connected_clients:
                # Get the first connected client's websocket and derive its ID
                for ws in connected_clients:
                    try:
                        ws_client_id = f"{ws.remote_address[0]}:{ws.remote_address[1]}"
                        
                        if pr_number:
                            # If there's already a PR, poll that specific PR for a session
                            logger.info(f"@copilot mentioned on PR #{pr_number}, starting direct session polling")
                            asyncio.create_task(
                                poll_for_copilot_session_on_pr(pr_number, ws, ws_client_id)
                            )
                            logger.info(f"Started Copilot session polling for PR #{pr_number}")
                        else:
                            # If there's no PR yet, wait for Copilot to create one
                            logger.info(f"@copilot mentioned on issue #{issue_number}, waiting for PR creation")
                            asyncio.create_task(
                                poll_for_copilot_session(issue_number, ws, ws_client_id)
                            )
                            logger.info(f"Started Copilot session polling for issue #{issue_number}")
                        break  # Only start one polling task
                    except Exception as e:
                        logger.error(f"Error starting Copilot poll for @copilot comment: {e}")
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        _log_to_all_sessions("ERROR", f"Failed to update issue #{issue_number}: {e}")
        _log_to_all_sessions("ERROR", tb)
        logger.error(f"Failed to update issue #{issue_number}: {e}")
        logger.error(tb)


def parse_transcript_with_ai(transcript: str, existing_data: dict = None) -> dict:
    """
    Use AI to parse the transcript and extract structured information for issue template.
    
    Args:
        transcript: The voice transcript to parse
        existing_data: Optional existing incomplete issue data for context
        
    Returns:
        Dictionary with parsed fields including 'missing_fields' list
    """
    if not GEMINI_API_KEY:
        logger.warning("Gemini API key not configured, using simple parsing")
        existing_prompts = existing_data.get('original_prompts', []) if existing_data else []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return {
            'type': 'visual AT',
            'title': transcript[:100],
            'description': transcript,
            'problem': '',
            'solution': '',
            'implementation_details': '',
            'example_usage': '',
            'custom_gpt': '',
            'gpt_query': '',
            'alternatives': '',
            'additional': '',
            'missing_fields': ['custom_gpt'],
            'original_prompts': existing_prompts + [f"[{timestamp}] {transcript}"]
        }
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Build context from existing data if this is a follow-up
        context_info = ""
        if existing_data:
            # Filter out empty values and missing_fields for cleaner context
            existing_fields = {k: v for k, v in existing_data.items() 
                             if k != 'missing_fields' and v and (isinstance(v, str) and v.strip() or not isinstance(v, str))}
            context_info = f"""
IMPORTANT: This is a follow-up response to an incomplete issue. The user is providing additional information.

Previously captured information:
{json.dumps(existing_fields, indent=2)}

Fields that were previously missing: {existing_data.get('missing_fields', [])}

The new transcript below is the user providing the missing information. Focus on extracting the information they are providing now.
"""
        
        prompt = f"""Parse the following voice transcript and extract information for a visual assistive technology (AT) tool request.

CRITICAL: Do not make up, infer, or extrapolate ANY content that was not explicitly said. If a field is empty or not mentioned, leave it empty - it can be filled in later. Only extract information that was directly stated in the transcript.

For a visual AT request, extract:
- title: A concise summary (max 100 characters)
- description: description of the desired visual assistive technology
- problem: Problem it solves
- solution: Proposed solution
- implementation_details: any specific tech stack details desired
- example_usage: an example use case and how the tool should handle it
- custom_gpt: should this tool operate similarly to a custom GPT (using Gemini Live with a repeated query on each camera frame)? ONLY set to "yes" or "no" if the user EXPLICITLY stated their preference. If they did not mention it at all, leave this field EMPTY (empty string "").
- gpt_query: if custom_gpt is "yes", what is the exact prompt to re-ask on every frame?
- alternatives: Alternative solutions considered
- additional: Any other context

CRITICAL: Evaluate which IMPORTANT fields are missing or insufficiently specified.
ONLY mark IMPORTANT fields in the missing_fields array. Optional/nice-to-have fields should NOT be included even if empty.

Important fields for visual AT: title, description, problem, solution, example_usage, custom_gpt, gpt_query
In the event custom_gpt is explicitly "no", gpt_query is no longer important.


Guidelines for determining if an IMPORTANT field is missing:
1. A field should ONLY be marked as missing if it is completely absent or so vague it provides no useful information
2. If a field has ANY meaningful content that addresses its purpose, it should NOT be marked as missing
3. For "problem": If the user explains why they need the tool, this is sufficient
4. For "solution": If the user describes what they want or how it should work, this is sufficient
5. For "example_usage": If the user provides any concrete example use case, this is sufficient
6. For "custom_gpt": Leave this field as an EMPTY STRING unless the user EXPLICITLY said "yes" or "no". If the transcript does not clearly contain the user saying they want or don't want custom GPT mode, the field MUST be empty. When it is empty, ALWAYS include "custom_gpt" in missing_fields. Do NOT guess, infer, or assume — this is a question we must ask the user directly.
7. For "gpt_query": Always include this in missing_fields if custom_gpt is "yes" and no query has been provided. If custom_gpt is "no", do not include this.
8. ALWAYS include example_usage in missing_fields if no concrete example use case (>= 10 characters) is present

If ALL important fields have meaningful content (even if brief), return an empty missing_fields array: []

Return ONLY a valid JSON object with the fields. Leave fields empty if not mentioned in the transcript.
{context_info}
Transcript: {transcript}

Return format:
{{
  "type": "visual AT",
  "title": "...",
  "description": "...",
  "problem": "...",
  "solution": "...",
  "implementation_details": "...",
  "example_usage": "...",
  "alternatives": "...",
  "custom_gpt": "...",
  "gpt_query": "...",
  "additional": "...",
  "missing_fields": ["field1", "field2"]  // Only truly missing/empty important fields. Use [] if all important fields have content.
}}"""

        response = model.generate_content(prompt)
        
        # Parse the AI response
        ai_response = response.text.strip()
        
        # Remove markdown code blocks if present
        if ai_response.startswith('```json'):
            ai_response = ai_response[7:]
        if ai_response.startswith('```'):
            ai_response = ai_response[3:]
        if ai_response.endswith('```'):
            ai_response = ai_response[:-3]
        ai_response = ai_response.strip()
        
        parsed_data = json.loads(ai_response)
        
        # Ensure missing_fields exists
        if 'missing_fields' not in parsed_data:
            parsed_data['missing_fields'] = []
        
        # Preserve the original transcript for bookkeeping
        existing_prompts = existing_data.get('original_prompts', []) if existing_data else []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        parsed_data['original_prompts'] = existing_prompts + [f"[{timestamp}] {transcript}"]
        
        logger.info(f"Successfully parsed transcript with AI: type={parsed_data.get('type', 'unknown')}, missing={len(parsed_data.get('missing_fields', []))}")
        return parsed_data
        
    except Exception as e:
        logger.error(f"Failed to parse transcript with AI: {e}")
        # Fallback to simple parsing
        existing_prompts = existing_data.get('original_prompts', []) if existing_data else []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return {
            'type': 'visual AT',
            'title': transcript[:100],
            'description': transcript,
            'problem': '',
            'solution': '',
            'implementation_details': '',
            'example_usage': '',
            'custom_gpt': '',
            'gpt_query': '',
            'alternatives': '',
            'additional': '',
            'missing_fields': ['custom_gpt'],
            'original_prompts': existing_prompts + [f"[{timestamp}] {transcript}"]
        }


def fill_template(template_content: str, parsed_data: dict) -> str:
    """
    Fill a GitHub issue template with parsed data.
    
    Args:
        template_content: The template markdown content
        parsed_data: Parsed data from AI
        
    Returns:
        Filled template content
    """
    filled = template_content
    
    # Helper function to ensure value is a string
    def ensure_string(value):
        if isinstance(value, list):
            return '\n'.join(str(item) for item in value)
        elif value is None:
            return ''
        return str(value)
    
    # Inject original prompts into the ORIGINAL_PROMPTS comment block
    original_prompts = parsed_data.get('original_prompts', [])
    if original_prompts:
        prompts_text = '\n'.join(original_prompts)
        filled = filled.replace(
            '<!-- ORIGINAL_PROMPTS\n-->',
            f'<!-- ORIGINAL_PROMPTS\n{prompts_text}\n-->'
        )
    
    # Fill visual AT template
    filled = filled.replace('<!-- A clear and concise description of the tool you\'d like. -->', 
                           ensure_string(parsed_data.get('description', '')))
    filled = filled.replace('<!-- Describe the problem this tool would solve. -->', 
                           ensure_string(parsed_data.get('problem', '')))
    filled = filled.replace('<!-- Describe how you envision this tool working. -->', 
                           ensure_string(parsed_data.get('solution', '')))
    filled = filled.replace('<!-- Any particular models or libraries that should be employed -->', 
                           ensure_string(parsed_data.get('implementation_details', '')))
    filled = filled.replace("<!-- Describe any alternative solutions or features you've considered. -->", 
                           ensure_string(parsed_data.get('alternatives', '')))
    filled = filled.replace('<!-- Describe an example situation the tool would be used in and how it could work -->', 
                           ensure_string(parsed_data.get('example_usage', '')))
    filled = filled.replace('<!-- Should this tool, in live mode, leverage Gemini live and work basically as a custom GPT without the need to ask again?-->', 
                           ensure_string(parsed_data.get('custom_gpt', '')))
    filled = filled.replace('<!-- If custom GPT, what is the query to be reasked every few seconds. Otherwise leave empty-->', 
                           ensure_string(parsed_data.get('gpt_query', '')))
    filled = filled.replace('<!-- Add any other context or screenshots about the feature request here. -->', 
                           ensure_string(parsed_data.get('additional', '')))
    
    return filled


def merge_parsed_data(existing_data: dict, new_data: dict) -> dict:
    """
    Merge new parsed data with existing incomplete issue data.
    New non-empty values override existing values.
    
    Args:
        existing_data: Previously parsed issue data
        new_data: Newly parsed data from follow-up
        
    Returns:
        Merged parsed data dictionary
    """
    merged = existing_data.copy()
    
    # Carry forward original prompts (already accumulated in parse_transcript_with_ai)
    if 'original_prompts' in new_data:
        merged['original_prompts'] = new_data['original_prompts']
    
    # Merge all fields - new non-empty values override old ones
    for key, value in new_data.items():
        if key == 'missing_fields':
            continue  # Don't merge missing_fields, we'll recalculate
        if key == 'original_prompts':
            continue  # Already handled above
        
        # Update if new value is non-empty
        if value:
            # Handle both strings and lists
            if isinstance(value, str) and value.strip():
                merged[key] = value
            elif isinstance(value, list) and value:
                merged[key] = value
            elif not isinstance(value, (str, list)):
                merged[key] = value
    
    # Update missing_fields from new data (AI re-evaluated what's still missing)
    merged['missing_fields'] = new_data.get('missing_fields', [])
    
    logger.info(f"Merged data: old fields={list(existing_data.keys())}, new fields={list(new_data.keys())}, merged fields={list(merged.keys())}")
    
    return merged


def generate_feedback_message(missing_fields: list, issue_type: str) -> str:
    """
    Generate a simple feedback message for missing fields.
    Ask for ONE field at a time instead of listing all missing fields.
    
    Args:
        missing_fields: List of missing field names
        issue_type: Type of issue (always 'visual AT')
        
    Returns:
        Feedback message string asking for the first missing field
    """
    if not missing_fields:
        return ""
    
    # Map field names to user-friendly descriptions for visual AT
    field_names = {
        'problem': 'problem description',
        'solution': 'proposed solution',
        'description': 'tool description',
        'implementation_details': 'implementation details',
        'example_usage': 'example usage',
        'custom_gpt': 'to know should this behave like a custom GPT, minus the need to reask often. Answer yes or no',
        'gpt_query': 'query for custom GPT',
        'additional': 'additional context',
        'alternatives': 'alternative solutions',
        'title': 'title'
    }
    
    # Only ask for the FIRST missing field
    first_missing = missing_fields[0]
    friendly_name = field_names.get(first_missing, first_missing)
    
    return f"I need {friendly_name}."


def _log_to_all_sessions(level: str, message: str):
    """Helper to log a message to all active session logs."""
    for session_log in active_session_loggers.values():
        session_log.log(level, message)


async def create_github_issue(text: str):
    """
    Create a GitHub issue OR update an existing one based on selected mode.
    Supports multi-turn conversations to fill missing fields.
    
    Args:
        text: Text to parse and use for creating/updating the issue
    """
    global incomplete_issue, selected_issue
    
    _log_to_all_sessions("INFO", f"create_github_issue called with text: {text}")
    _log_to_all_sessions("INFO", f"Current mode: {selected_issue.get('mode')}, issue: {selected_issue.get('number')}")
    logger.info(f"create_github_issue called with text: {text[:100]}")
    logger.info(f"Current mode: {selected_issue.get('mode')}, issue: {selected_issue.get('number')}")
    
    if not GITHUB_TOKEN:
        _log_to_all_sessions("WARNING", "GitHub token not configured, skipping issue creation")
        logger.warning("GitHub token not configured, skipping issue creation")
        return
    
    if not text or not text.strip():
        _log_to_all_sessions("WARNING", "Empty text, skipping issue creation")
        logger.warning("Empty text, skipping issue creation")
        return
    
    try:
        # FIRST: Check if we're already in update mode - if so, just add the comment
        logger.info(f"[DEBUG] Checking update mode: mode={selected_issue.get('mode')} (type: {type(selected_issue.get('mode'))}), number={selected_issue.get('number')} (type: {type(selected_issue.get('number'))})")
        logger.info(f"[DEBUG] Condition check: mode=='update': {selected_issue['mode'] == 'update'}, has number: {bool(selected_issue['number'])}")
        
        if selected_issue['mode'] == 'update' and selected_issue['number']:
            logger.info(f"Already in update mode for issue #{selected_issue['number']}, adding comment")
            # Determine if we should mention @copilot based on the comment content
            mention_copilot = should_mention_copilot(text.strip())
            logger.info(f"Comment mentions copilot: {mention_copilot}")
            await update_github_issue(selected_issue['number'], text.strip(), mention_copilot)
            return
        
        # SECOND: Parse the text to see if it's a selection/mode change command
        logger.info(f"[DEBUG] NOT in update mode, parsing text for issue selection")
        available_issues = fetch_open_issues()
        selection = parse_issue_selection(text.strip(), available_issues)
        
        # Handle list mode
        if selection.get('mode') == 'list':
            if connected_clients and available_issues:
                issue_list_msg = "Open issues:\n" + "\n".join([
                    f"Issue {issue['number']}: {issue['title']}" 
                    for issue in available_issues[:10]  # Limit to 10 for voice
                ])
                list_data = {
                    'type': 'issue_list',
                    'message': issue_list_msg,
                    'issues': available_issues[:10]
                }
                import websockets
                websockets.broadcast(connected_clients, json.dumps(list_data))
            return
        
        # Handle update mode selection
        if selection.get('mode') == 'update' and selection.get('issue_number'):
            selected_issue['number'] = selection['issue_number']
            selected_issue['title'] = selection.get('issue_title', '')
            selected_issue['mode'] = 'update'
            
            # Send confirmation to client
            if connected_clients:
                confirm_data = {
                    'type': 'issue_selected',
                    'message': f"Selected issue #{selected_issue['number']}: {selected_issue['title']}",
                    'issue_number': selected_issue['number'],
                    'issue_title': selected_issue['title']
                }
                import websockets
                websockets.broadcast(connected_clients, json.dumps(confirm_data))
            
            logger.info(f"Switched to update mode for issue #{selected_issue['number']}")
            return
        
        # If mode is explicitly set to create, reset to create mode
        if selection.get('mode') == 'create':
            selected_issue['mode'] = 'create'
            selected_issue['number'] = None
            selected_issue['title'] = None
        
        # Otherwise, create a new issue (existing logic below)
        # Check if we have an incomplete issue from previous interaction
        if incomplete_issue['data'] and incomplete_issue['missing_fields']:
            logger.info(f"Continuing incomplete issue conversation. Missing: {incomplete_issue['missing_fields']}")
            _log_to_all_sessions("INFO", f"Continuing incomplete issue. Missing fields: {incomplete_issue['missing_fields']}")
            
            # Parse new transcript focusing on filling missing fields
            # Pass existing data as context so AI knows this is a follow-up
            _log_to_all_sessions("INFO", "Calling parse_transcript_with_ai (follow-up)")
            new_parsed = parse_transcript_with_ai(text.strip(), existing_data=incomplete_issue['data'])
            _log_to_all_sessions("INFO", f"AI parsing result: {new_parsed}")
            
            # Merge new data with existing incomplete data
            parsed_data = merge_parsed_data(incomplete_issue['data'], new_parsed)
            
            logger.info(f"Merged issue data. Still missing: {parsed_data.get('missing_fields', [])}")
            _log_to_all_sessions("INFO", f"Merged data. Still missing: {parsed_data.get('missing_fields', [])}")
        else:
            # First time parsing this issue
            _log_to_all_sessions("INFO", "Calling parse_transcript_with_ai (initial)")
            parsed_data = parse_transcript_with_ai(text.strip())
            _log_to_all_sessions("INFO", f"AI parsing result: {parsed_data}")
            logger.info(f"Initial parsing. Type: {parsed_data.get('type')}, Missing: {parsed_data.get('missing_fields', [])}")
        
        # Check for missing fields
        missing_fields = parsed_data.get('missing_fields', [])
        issue_type = 'visual AT'  # Always visual AT now
        
        # If there are important missing fields, send feedback and save incomplete data
        if missing_fields:
            # Store incomplete issue for next round
            incomplete_issue['data'] = parsed_data
            incomplete_issue['missing_fields'] = missing_fields
            incomplete_issue['timestamp'] = datetime.now()
            
            feedback_msg = generate_feedback_message(missing_fields, issue_type)
            if feedback_msg and connected_clients:
                logger.info(f"Sending feedback for missing fields: {missing_fields}")
                feedback_data = {
                    'type': 'feedback',
                    'message': feedback_msg,
                    'missing_fields': missing_fields
                }
                # Send to all connected clients
                import websockets
                websockets.broadcast(connected_clients, json.dumps(feedback_data))
            
            # Don't create issue yet, wait for more info
            logger.info("Issue incomplete, waiting for user to provide more details")
            return
        
        # All required fields are filled! Clear incomplete state and create issue
        logger.info("All required fields filled, creating issue")
        incomplete_issue['data'] = None
        incomplete_issue['missing_fields'] = []
        incomplete_issue['timestamp'] = None
        
        # Use visual AT template
        template_file = TEMPLATE_DIR / 'visual_at.md'
        
        # Load and fill template
        body = ""
        if template_file.exists():
            with open(template_file, 'r', encoding='utf-8') as f:
                template_content = f.read()
                
            # Remove front matter (YAML header)
            if template_content.startswith('---'):
                parts = template_content.split('---', 2)
                if len(parts) >= 3:
                    template_content = parts[2].strip()
            
            # Fill template with parsed data
            body = fill_template(template_content, parsed_data)
        else:
            # Fallback if template doesn't exist
            body = f"**Transcript:**\n{text}\n\n**Parsed Data:**\n{json.dumps(parsed_data, indent=2)}"
        
        # Create GitHub client
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        
        # Create issue with parsed title and filled template
        title = parsed_data.get('title', text[:100])
        
        # Ensure title is a string (AI might return unexpected formats)
        if isinstance(title, list):
            title = ' '.join(str(item) for item in title)
        elif not isinstance(title, str):
            title = str(title)
        
        # Truncate title if too long (GitHub has a limit)
        if len(title) > 256:
            title = title[:253] + '...'
        
        labels = ['enhancement']  # Always enhancement for visual AT
        
        _log_to_all_sessions("INFO", f"Creating GitHub issue - title: {title}, labels: {labels}")
        issue = repo.create_issue(
            title=title,
            body=body,
            labels=labels
        )
        _log_to_all_sessions("INFO", f"GitHub API: Created issue #{issue.number}: {title} (url: {issue.html_url})")
        logger.info(f"Created GitHub issue #{issue.number}: {title[:50]}... (type: {issue_type})")
        
        # Send success notification to client
        if connected_clients:
            success_data = {
                'type': 'issue_created',
                'message': f"Issue #{issue.number} created successfully",
                'issue_number': issue.number,
                'issue_url': issue.html_url
            }
            import websockets
            websockets.broadcast(connected_clients, json.dumps(success_data))
            
            # Start polling for Copilot session for each connected client
            for ws in connected_clients:
                try:
                    ws_client_id = f"{ws.remote_address[0]}:{ws.remote_address[1]}"
                    
                    # Track this issue as pending (websocket may go stale on disconnect)
                    pending_copilot_issues[issue.number] = {
                        'created_at': datetime.now(),
                        'websocket': ws,
                        'client_id': ws_client_id,
                        'pr_number': None,  # filled in once PR is found
                    }
                    
                    # Start polling in background task
                    asyncio.create_task(
                        poll_for_copilot_session(issue.number, ws, ws_client_id)
                    )
                    logger.info(f"Started Copilot session polling for issue #{issue.number}")
                except Exception as e:
                    logger.error(f"Error starting Copilot poll for client: {e}")
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        _log_to_all_sessions("ERROR", f"Failed to create GitHub issue: {e}")
        _log_to_all_sessions("ERROR", tb)
        logger.error(f"Failed to create GitHub issue: {e}")
        logger.error(tb)
        # Clear incomplete state on error
        incomplete_issue['data'] = None
        incomplete_issue['missing_fields'] = []
        incomplete_issue['timestamp'] = None


async def monitor_text_pause():
    """
    Monitor for text pause and create GitHub issue when detected.
    Waits for PAUSE_DURATION seconds after last text received.
    """
    try:
        while True:
            await asyncio.sleep(1)  # Check every second
            
            if last_text['content'] and last_text['timestamp']:
                elapsed = (datetime.now() - last_text['timestamp']).total_seconds()
                
                # If enough time has passed since last text and we haven't created an issue yet
                if elapsed >= PAUSE_DURATION and last_text['task'] is None:
                    # Mark that we're creating an issue to avoid duplicates
                    last_text['task'] = 'creating'
                    
                    logger.info(f"[DEBUG] Pause detected ({elapsed:.1f}s), mode: {selected_issue.get('mode')}, issue: {selected_issue.get('number')}")
                    logger.info(f"Pause detected ({elapsed:.1f}s), calling create_github_issue with content: {last_text['content'][:100]}")
                    
                    # Save the content before clearing in case create_github_issue needs it
                    content_to_process = last_text['content']
                    
                    # Clear the buffer BEFORE processing to allow new text during processing
                    last_text['content'] = None
                    last_text['timestamp'] = None
                    last_text['task'] = None
                    
                    # Create the issue (or update if in update mode)
                    await create_github_issue(content_to_process)
                    
                    logger.info("Finished processing text")
    except Exception as e:
        logger.error(f"Error in monitor_text_pause: {e}", exc_info=True)


def decode_frame(base64_data: str) -> np.ndarray:
    """
    Decode base64 image data to numpy array
    
    Args:
        base64_data: Base64 encoded image string
        
    Returns:
        numpy array of image
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        # Decode base64 to bytes
        img_bytes = base64.b64decode(base64_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        logger.error(f"Failed to decode frame: {e}")
        return None


# New helper to process/save incoming text
def process_text(text_payload) -> dict:
    """
    Process and optionally save incoming text messages.
    Accepts a string or dict (will stringify).

    Uses a word-based suffix (delta) strategy for CREATE mode: only the newly appended words
    compared to the previous full transcript are queued in `last_text['content']`.
    This avoids repeatedly creating issues with the entire growing transcript.
    
    For UPDATE mode: uses the full text as-is for comments.
    """
    try:
        if text_payload is None:
            return {'status': 'empty'}

        # Normalize to string
        if isinstance(text_payload, dict):
            text_str = json.dumps(text_payload, ensure_ascii=False)
        else:
            text_str = str(text_payload)

        text_str = text_str.strip()

        # Append full incoming text to log for audit
        if SAVE_FRAMES:
            log_file = FRAMES_DIR / "received_texts.log"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} - {text_str}\n")

        # Check if we're in UPDATE mode - if so, use full text, not delta
        if selected_issue.get('mode') == 'update' and selected_issue.get('number'):
            logger.info(f"[DEBUG] UPDATE mode detected in process_text, using full text")
            last_text['content'] = text_str
            last_text['prev_raw'] = text_str
            last_text['timestamp'] = datetime.now()
            last_text['task'] = None
            return {'status': 'saved' if SAVE_FRAMES else 'received', 'text_preview': text_str[:200]}

        # CREATE mode: use delta calculation
        prev = last_text.get('prev_raw') or ""

        logger.info(f"[DEBUG] process_text: prev='{prev[:100]}', current='{text_str[:100]}'")

        # Tokenize into words (simple split on whitespace). This makes the delta
        # computation robust to small edits and focuses on appended words.
        prev_words = prev.split()
        curr_words = text_str.split()

        # find common prefix length in words
        i = 0
        maxlen = min(len(prev_words), len(curr_words))
        while i < maxlen and prev_words[i] == curr_words[i]:
            i += 1

        # suffix/delta is remaining words in current after the common prefix
        delta_words = curr_words[i:]
        delta = " ".join(delta_words).strip()

        logger.info(f"[DEBUG] Delta calculation: common_prefix_len={i}, prev_words={len(prev_words)}, curr_words={len(curr_words)}, delta='{delta[:100]}'")

        # Update trackers
        last_text['prev_raw'] = text_str
        last_text['timestamp'] = datetime.now()
        last_text['task'] = None

        if not delta:
            # nothing new appended
            logger.info(f"[DEBUG] No delta detected, returning no_change")
            return {'status': 'no_change', 'text_preview': text_str[:200]}

        # Only queue the delta for issue creation
        last_text['content'] = delta

        return {'status': 'saved' if SAVE_FRAMES else 'received', 'text_preview': delta[:200]}
    except Exception as e:
        logger.error(f"Failed to process text: {e}")
        return {'status': 'error', 'error': str(e)}


def cleanup_old_frames(frames_dir: Path, keep_last: int = 20):
    """
    Clean up old frame files, keeping only the most recent ones.
    
    Args:
        frames_dir: Directory containing frame files
        keep_last: Number of most recent frames to keep
    """
    try:
        # Get all frame files sorted by modification time (most recent first)
        frame_files = sorted(
            [f for f in frames_dir.glob("frame_*.jpg")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if len(frame_files) <= keep_last:
            return  # Nothing to clean up
        
        # Delete old frames, keep only the last N
        files_to_delete = frame_files[keep_last:]
        for file_path in files_to_delete:
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete frame {file_path}: {e}")
        
        if files_to_delete:
            logger.info(f"Cleaned up {len(files_to_delete)} old frames, kept {len(frame_files) - len(files_to_delete)} recent frames")
            
    except Exception as e:
        logger.error(f"Error during frame cleanup: {e}")


def process_frame(frame_data: dict) -> dict:
    """
    Process received frame data (image only)

    Args:
        frame_data: Dictionary containing frame information

    Returns:
        Processing results
    """
    global last_frame
    
    now = datetime.now()
    results = {
        'status': 'processed',
        'timestamp': now.isoformat()
    }

    try:
        # If frame contains base64 image data, decode and process it
        if 'base64Image' in frame_data.get('data', {}):
            base64_data = frame_data['data']['base64Image']
            img = decode_frame(base64_data)

            if img is not None:
                height, width = img.shape[:2]
                results['frame_shape'] = f"{width}x{height}"
                
                # Store the last frame for tool execution
                last_frame['image'] = img
                last_frame['timestamp'] = now
                last_frame['base64'] = base64_data

                # Note: Streaming tools will be run in handle_client after process_frame returns

                # later: insert processing here, rn they are just images

                # Save frame if enabled
                if SAVE_FRAMES:
                    # Use the already created timestamp for unique filenames
                    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                    filename = FRAMES_DIR / f"frame_{timestamp}.jpg"
                    cv2.imwrite(str(filename), img)
                    results['saved'] = str(filename)
                    
                    # Clean up old frames occasionally (every 10th frame) to avoid lag
                    if stats['total_frames'] % 10 == 0:
                        cleanup_old_frames(FRAMES_DIR, keep_last=100)
        else:
            results['status'] = 'no_image'

        # Add more processing here (ML models, object detection, etc.)

    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        results['error'] = str(e)

    return results


async def handle_client(websocket):
    """
    Handle WebSocket client connection
    
    Args:
        websocket: WebSocket connection
    """
    global active_streaming_tools
    
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    path = websocket.request.path if hasattr(websocket, 'request') else '/'
    logger.info(f"Client connected: {client_id} (path: {path})")
    connected_clients.add(websocket)
    
    # Create session logger for this client
    session_log = SessionLogger(client_id)
    active_session_loggers[client_id] = session_log
    session_log.log("INFO", f"Client connected from path: {path}")
    
    if stats['start_time'] is None:
        stats['start_time'] = datetime.now()
    
    try:
        # Send welcome message
        await websocket.send(json.dumps({
            'type': 'connection',
            'status': 'connected',
            'message': 'Ready to receive frames and text',
            'server_time': datetime.now().isoformat()
        }))
        session_log.log_message("send", "connection", "Welcome message sent")
        
        async for message in websocket:
            try:
                message_size_kb = len(message) / 1024
                # Parse message
                try:
                    data = json.loads(message)
                except Exception:
                    data = None
                # Always count bytes
                stats['total_bytes'] += len(message)
                
                # Handle control messages FIRST (they don't need frames/text)
                msg_type = data.get('type')
                
                # Log received message (skip frame data to avoid huge logs)
                if msg_type:
                    if msg_type not in ['frame', 'ping']:  # Don't log frequent frame/ping messages
                        # Log full message details (excluding large binary fields)
                        filtered_data = {k: v for k, v in data.items() if k not in ['frame', 'data', 'tool_code']}
                        session_log.log_message("recv", msg_type, json.dumps(filtered_data))
                elif 'text' in data:
                    # Log full text, no truncation
                    session_log.log_message("recv", "text", data.get('text', ''))
                
                # Handle start_streaming_tool message type (for continuous tool execution)
                if msg_type == 'start_streaming_tool':
                    logger.info(f"Client {client_id} started streaming tool: {data.get('tool_name')}")
                    session_log.log("INFO", f"Started streaming tool: {data.get('tool_name')}")
                    
                    custom_gpt = data.get('custom_gpt', False)
                    gpt_query = data.get('gpt_query', '')
                    
                    active_streaming_tools[client_id] = {
                        'tool': {
                            'name': data.get('tool_name', 'unknown'),
                            'code': data.get('tool_code', ''),
                            'language': data.get('tool_language', 'python'),
                            'input': data.get('input', ''),
                        },
                        'last_run': None,
                        'throttle_ms': data.get('throttle_ms', 1000),
                    }
                    
                    # Custom GPT mode: use Gemini Live instead of code execution
                    if custom_gpt and gpt_query and gemini_live_manager:
                        logger.info(f"Starting Gemini Live session for {client_id} (custom GPT mode)")
                        session_log.log("INFO", f"Gemini Live mode: {gpt_query[:80]}")
                        
                        active_streaming_tools[client_id]['gemini_live'] = True
                        
                        # Response handler sends results back to client
                        async def make_live_response_handler(ws, cid, tname):
                            async def handler(text: str, is_partial: bool):
                                if not is_partial and text:
                                    logger.info(f"[GEMINI->APP] Handler invoked for {cid}: '{text[:120]}'")
                                    try:
                                        await ws.send(json.dumps({
                                            'type': 'tool_stream_result',
                                            'tool_name': tname,
                                            'result': text,
                                            'audio': {
                                                'type': 'speech',
                                                'text': text,
                                                'rate': 1.0,
                                                'interrupt': False,
                                            },
                                            'mode': 'gemini_live',
                                            'timestamp': datetime.now().isoformat(),
                                        }))
                                        logger.info(f"[GEMINI->APP] WebSocket send OK for {cid}")
                                    except Exception as e:
                                        logger.error(f"[GEMINI->APP] WebSocket send FAILED for {cid}: {e}")
                                        import traceback as tb
                                        logger.error(tb.format_exc())
                            return handler
                        
                        response_handler = await make_live_response_handler(
                            websocket, client_id, data.get('tool_name', 'unknown')
                        )
                        
                        try:
                            system_instruction = data.get('system_instruction', '')
                            await gemini_live_manager.start_session(
                                client_id, system_instruction, response_handler
                            )
                            
                            # Start the periodic query loop in the background
                            interval = data.get('query_interval', 3.0)
                            
                            def get_current_frame():
                                b64 = last_frame.get('base64')
                                img = last_frame.get('image')
                                return (b64, img)
                            
                            query_task = asyncio.create_task(
                                gemini_live_manager.run_query_loop(
                                    client_id, gpt_query, get_current_frame, interval
                                )
                            )
                            gemini_live_manager._query_tasks[client_id] = query_task
                            
                            await websocket.send(json.dumps({
                                'type': 'streaming_started',
                                'tool_name': data.get('tool_name'),
                                'mode': 'gemini_live',
                                'throttle_ms': data.get('throttle_ms', 1000),
                                'timestamp': datetime.now().isoformat()
                            }))
                        except Exception as e:
                            logger.error(f"Failed to start Gemini Live session for {client_id}: {e}")
                            # Fall back to normal streaming
                            active_streaming_tools[client_id].pop('gemini_live', None)
                            await websocket.send(json.dumps({
                                'type': 'streaming_started',
                                'tool_name': data.get('tool_name'),
                                'mode': 'code_execution',
                                'error': f'Gemini Live failed, using code execution: {e}',
                                'throttle_ms': data.get('throttle_ms', 1000),
                                'timestamp': datetime.now().isoformat()
                            }))
                    else:
                        # Normal streaming mode (code execution per frame)
                        await websocket.send(json.dumps({
                            'type': 'streaming_started',
                            'tool_name': data.get('tool_name'),
                            'mode': 'code_execution',
                            'throttle_ms': data.get('throttle_ms', 1000),
                            'timestamp': datetime.now().isoformat()
                        }))
                    continue

                # Handle stop_streaming_tool message type
                if msg_type == 'stop_streaming_tool':
                    logger.info(f"Client {client_id} stopped streaming tool")
                    if client_id in active_streaming_tools:
                        tool_name = active_streaming_tools[client_id]['tool']['name']
                        # Clean up Gemini Live session if active
                        if active_streaming_tools[client_id].get('gemini_live') and gemini_live_manager:
                            await gemini_live_manager.stop_session(client_id)
                            logger.info(f"Stopped Gemini Live session for {client_id}")
                        del active_streaming_tools[client_id]
                        
                        await websocket.send(json.dumps({
                            'type': 'streaming_stopped',
                            'tool_name': tool_name,
                            'timestamp': datetime.now().isoformat()
                        }))
                    else:
                        # Not currently streaming, but still acknowledge
                        await websocket.send(json.dumps({
                            'type': 'streaming_stopped',
                            'tool_name': None,
                            'timestamp': datetime.now().isoformat()
                        }))
                    continue

                # Handle pause/resume of the Gemini Live query loop (sent by client during follow-up voice capture)
                if msg_type == 'pause_live_query':
                    logger.info(f"Client {client_id} requested pause of live query loop")
                    if gemini_live_manager:
                        gemini_live_manager.pause_query_loop(client_id)
                    continue

                if msg_type == 'resume_live_query':
                    logger.info(f"Client {client_id} requested resume of live query loop")
                    if gemini_live_manager:
                        gemini_live_manager.resume_query_loop(client_id)
                    continue

                # Handle live_followup message type (send follow-up to Gemini Live session)
                if msg_type == 'live_followup':
                    followup_text = data.get('text', '')
                    logger.info(f"Client {client_id} sent live follow-up: {followup_text[:80]}")
                    if gemini_live_manager and client_id in getattr(gemini_live_manager, 'sessions', {}):
                        try:
                            # Pause periodic queries so they don't race with the follow-up
                            gemini_live_manager.pause_query_loop(client_id)
                            response = await gemini_live_manager.send_followup(client_id, followup_text)
                            await websocket.send(json.dumps({
                                'type': 'live_followup_response',
                                'text': response,
                                'timestamp': datetime.now().isoformat()
                            }))
                        except Exception as e:
                            logger.error(f"Error sending live follow-up for {client_id}: {e}")
                            await websocket.send(json.dumps({
                                'type': 'live_followup_response',
                                'text': f'Follow-up error: {e}',
                                'error': True,
                                'timestamp': datetime.now().isoformat()
                            }))
                        finally:
                            # Always resume the query loop, even if the follow-up failed
                            gemini_live_manager.resume_query_loop(client_id)
                            logger.info(f"Resumed query loop after follow-up for {client_id}")
                    else:
                        await websocket.send(json.dumps({
                            'type': 'live_followup_response',
                            'text': 'No active Gemini Live session',
                            'error': True,
                            'timestamp': datetime.now().isoformat()
                        }))
                    continue

                # Handle request_issue_list message type
                if msg_type == 'request_issue_list':
                    logger.info(f"Client {client_id} requested issue list")
                    available_issues = fetch_open_issues()
                    if available_issues:
                        issue_list_msg = "Open issues:\n" + "\n".join([
                            f"Issue {issue['number']}: {issue['title']}" 
                            for issue in available_issues[:10]
                        ])
                        await websocket.send(json.dumps({
                            'type': 'issue_list',
                            'message': issue_list_msg,
                            'issues': available_issues[:10],
                            'timestamp': datetime.now().isoformat()
                        }))
                    continue

                # Handle request_pr_list message type
                if msg_type == 'request_pr_list':
                    logger.info(f"Client {client_id} requested PR list")
                    available_prs = fetch_open_prs()
                    if available_prs:
                        await websocket.send(json.dumps({
                            'type': 'pr_list',
                            'prs': available_prs,
                            'timestamp': datetime.now().isoformat()
                        }))
                        logger.info(f"Sent {len(available_prs)} open PRs to {client_id}")
                    else:
                        await websocket.send(json.dumps({
                            'type': 'pr_list',
                            'prs': [],
                            'message': 'No open PRs found',
                            'timestamp': datetime.now().isoformat()
                        }))
                    continue

                # Handle request_pr_tools message type
                if msg_type == 'request_pr_tools':
                    pr_number = data.get('pr_number')
                    if pr_number:
                        logger.info(f"Client {client_id} requested tools for PR #{pr_number}")
                        # Fetch PR title and tools separately to ensure we always have the title
                        pr_title = fetch_pr_title(pr_number)
                        tools = fetch_pr_tools(pr_number)
                        await websocket.send(json.dumps({
                            'type': 'pr_tools',
                            'pr_number': pr_number,
                            'pr_title': pr_title,
                            'tools': tools,
                            'timestamp': datetime.now().isoformat()
                        }))
                        logger.info(f"Sent {len(tools)} tools for PR #{pr_number} ({pr_title}) to {client_id}")
                    else:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'PR number is required',
                            'timestamp': datetime.now().isoformat()
                        }))
                    continue

                # Handle request_production_tools message type
                if msg_type == 'request_production_tools':
                    branch = data.get('branch', 'main')
                    logger.info(f"Client {client_id} requested production tools from branch: {branch}")
                    tools = fetch_branch_tools(branch)
                    await websocket.send(json.dumps({
                        'type': 'production_tools',
                        'branch': branch,
                        'tools': tools,
                        'timestamp': datetime.now().isoformat()
                    }))
                    logger.info(f"Sent {len(tools)} production tools from {branch} to {client_id}")
                    continue

                # Handle ping message type
                if msg_type == 'ping':
                    await websocket.send(json.dumps({
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    }))
                    continue

                # Handle request_copilot_sessions message type
                if msg_type == 'request_copilot_sessions':
                    logger.info(f"Client {client_id} requested Copilot sessions")
                    session_log.log("INFO", "Fetching Copilot sessions")
                    sessions = await fetch_copilot_sessions()
                    await websocket.send(json.dumps({
                        'type': 'copilot_sessions',
                        'sessions': sessions,
                        'timestamp': datetime.now().isoformat()
                    }))
                    continue

                # Handle start_copilot_session_stream message type
                if msg_type == 'start_copilot_session_stream':
                    # Accept either session_id or pr_number
                    pr_or_session = data.get('session_id') or data.get('pr_number')
                    explicit_pr_number = data.get('pr_number') if data.get('session_id') else None
                    
                    if not pr_or_session:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'session_id or pr_number is required',
                            'timestamp': datetime.now().isoformat()
                        }))
                        continue
                    
                    logger.info(f"Client {client_id} starting Copilot agent task stream: {pr_or_session}")
                    session_log.log("INFO", f"Starting Copilot agent task stream: {pr_or_session}")
                    
                    # Re-attach to any pending Copilot poll for this PR/session
                    _pr_num_to_check = int(pr_or_session) if str(pr_or_session).isdigit() else explicit_pr_number
                    if _pr_num_to_check:
                        for _issue_num, _info in pending_copilot_issues.items():
                            if (_info.get('pr_number') == _pr_num_to_check or 
                                (not _info.get('pr_number') and _info.get('websocket') is None)):
                                _info['websocket'] = websocket
                                _info['client_id'] = client_id
                                logger.info(f"Re-attached client {client_id} to pending Copilot poll for issue #{_issue_num}")
                    
                    # Stop any existing stream for this client
                    if client_id in active_copilot_streams:
                        await stop_copilot_session_stream(client_id)
                    
                    # Check if this is a reconnect to an already-known active session
                    # If the session exists in the DB and is still active, skip the
                    # 45-second wait and resume from where we left off.
                    _reconnect_skip_wait = False
                    _reconnect_skip_lines = 0
                    _reconnect_entry_count = 0
                    _session_id_for_reconnect = str(pr_or_session)
                    
                    # If given a session_id directly, check DB
                    if not _session_id_for_reconnect.isdigit():
                        existing = copilot_db.get_session(_session_id_for_reconnect)
                        if existing and existing.get('status') == 'active':
                            _reconnect_skip_wait = True
                            _reconnect_skip_lines = copilot_db.get_log_count(_session_id_for_reconnect)
                            _reconnect_entry_count = _reconnect_skip_lines  # approximate
                            logger.info(
                                f"Reconnecting to active session {_session_id_for_reconnect}, "
                                f"skipping {_reconnect_skip_lines} already-stored log lines"
                            )
                            session_log.log("INFO", f"Reconnecting to active session (resuming after {_reconnect_skip_lines} lines)")
                    else:
                        # PR number given — check if there's an active session for this PR
                        pr_sessions = copilot_db.get_sessions_for_pr(int(_session_id_for_reconnect))
                        active_sessions = [s for s in pr_sessions if s.get('status') == 'active']
                        if active_sessions:
                            # Use the most recent active session
                            active_session = active_sessions[-1]
                            _reconnect_skip_wait = True
                            sid = active_session['session_id']
                            _reconnect_skip_lines = copilot_db.get_log_count(sid)
                            _reconnect_entry_count = _reconnect_skip_lines
                            logger.info(
                                f"Found active session {sid} for PR #{pr_or_session}, "
                                f"reconnecting (skipping {_reconnect_skip_lines} lines)"
                            )
                            session_log.log("INFO", f"Reconnecting to active session {sid} for PR #{pr_or_session}")
                    
                    # Start streaming in a background task with optional PR number
                    task = asyncio.create_task(
                        stream_copilot_session_logs(
                            websocket, client_id, pr_or_session,
                            pr_number=explicit_pr_number,
                            skip_wait=_reconnect_skip_wait,
                            _skip_lines=_reconnect_skip_lines,
                            _start_entry_count=_reconnect_entry_count,
                        )
                    )
                    # Store task reference if the stream was initialized
                    if client_id in active_copilot_streams:
                        active_copilot_streams[client_id]['task'] = task
                    continue

                # Handle stop_copilot_session_stream message type
                if msg_type == 'stop_copilot_session_stream':
                    logger.info(f"Client {client_id} stopping Copilot session stream")
                    session_log.log("INFO", "Stopping Copilot session stream")
                    await stop_copilot_session_stream(client_id)
                    await websocket.send(json.dumps({
                        'type': 'copilot_session_stream_stopped',
                        'timestamp': datetime.now().isoformat()
                    }))
                    continue

                # Handle get_pr_sessions - fetch all sessions for a PR
                if msg_type == 'get_pr_sessions':
                    pr_number = data.get('pr_number')
                    logger.info(f"Client {client_id} requesting sessions for PR #{pr_number}")
                    session_log.log("INFO", f"Requesting sessions for PR #{pr_number}")
                    
                    if not pr_number:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'pr_number required',
                            'timestamp': datetime.now().isoformat()
                        }))
                        continue
                    
                    # Re-attach this client to any pending Copilot poll that
                    # found this PR (e.g. client disconnected and reconnected)
                    for _issue_num, _info in pending_copilot_issues.items():
                        if _info.get('pr_number') == pr_number and _info.get('websocket') is None:
                            _info['websocket'] = websocket
                            _info['client_id'] = client_id
                            logger.info(
                                f"Re-attached client {client_id} to pending Copilot poll "
                                f"for issue #{_issue_num} (PR #{pr_number})"
                            )
                            session_log.log("INFO", f"Re-attached to pending Copilot poll for issue #{_issue_num}")
                    
                    # Get sessions from database
                    sessions = copilot_db.get_sessions_for_pr(pr_number)
                    
                    # If no sessions in DB, try to fetch from GitHub
                    if not sessions:
                        logger.info(f"No sessions in DB for PR #{pr_number}, fetching from GitHub")
                        # This will be handled by a background task
                        asyncio.create_task(fetch_and_store_pr_sessions(websocket, client_id, pr_number))
                    else:
                        # Recheck any "active" sessions against GitHub to see
                        # if they actually finished while the client was away.
                        active_sessions = [s for s in sessions if s.get('status') == 'active']
                        if active_sessions:
                            _env = os.environ.copy()
                            _env.pop('GH_TOKEN', None)
                            _env.pop('GITHUB_TOKEN', None)
                            for _s in active_sessions:
                                _sid = _s['session_id']
                                try:
                                    _proc = await asyncio.create_subprocess_exec(
                                        'gh', 'agent-task', 'view', _sid,
                                        '-R', GITHUB_REPO,
                                        stdout=asyncio.subprocess.PIPE,
                                        stderr=asyncio.subprocess.PIPE,
                                        env=_env
                                    )
                                    _stdout, _ = await asyncio.wait_for(_proc.communicate(), timeout=10)
                                    _output = _stdout.decode('utf-8', errors='replace').lower()
                                    _still_active = any(
                                        kw in _output for kw in ['in progress', 'active', 'running']
                                    )
                                    if not _still_active:
                                        # Session finished — determine status
                                        if 'completed' in _output or 'succeeded' in _output:
                                            _new_status = 'completed'
                                        elif 'failed' in _output or 'error' in _output:
                                            _new_status = 'failed'
                                        else:
                                            _new_status = 'completed'
                                        copilot_db.update_session_status(_sid, _new_status)
                                        _s['status'] = _new_status
                                        logger.info(f"Session {_sid} is no longer active, updated to '{_new_status}'")
                                except asyncio.TimeoutError:
                                    logger.warning(f"Timeout checking status of session {_sid}")
                                except Exception as _e:
                                    logger.warning(f"Could not recheck session {_sid}: {_e}")
                        
                        # Send (possibly updated) sessions
                        await websocket.send(json.dumps({
                            'type': 'pr_sessions_list',
                            'pr_number': pr_number,
                            'sessions': sessions,
                            'timestamp': datetime.now().isoformat()
                        }))
                    continue

                # Handle get_session_summaries - fetch summaries for a session
                if msg_type == 'get_session_summaries':
                    session_id = data.get('session_id')
                    logger.info(f"Client {client_id} requesting summaries for session {session_id}")
                    session_log.log("INFO", f"Requesting summaries for session {session_id}")
                    
                    if not session_id:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'session_id required',
                            'timestamp': datetime.now().isoformat()
                        }))
                        continue
                    
                    # Get summaries from database
                    summaries = copilot_db.get_summaries_for_session(session_id)
                    
                    await websocket.send(json.dumps({
                        'type': 'session_summaries',
                        'session_id': session_id,
                        'summaries': summaries,
                        'timestamp': datetime.now().isoformat()
                    }))
                    continue

                # Handle get_session_logs - fetch full logs for a session
                if msg_type == 'get_session_logs':
                    session_id = data.get('session_id')
                    start_entry = data.get('start_entry')  # Optional: filter by entry range
                    end_entry = data.get('end_entry')
                    logger.info(f"Client {client_id} requesting logs for session {session_id}")
                    session_log.log("INFO", f"Requesting logs for session {session_id}")
                    
                    if not session_id:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'session_id required',
                            'timestamp': datetime.now().isoformat()
                        }))
                        continue
                    
                    # Get logs from database
                    if start_entry is not None and end_entry is not None:
                        logs = copilot_db.get_logs_for_entry_range(session_id, start_entry, end_entry)
                    else:
                        logs = copilot_db.get_logs_for_session(session_id)
                    
                    await websocket.send(json.dumps({
                        'type': 'session_logs',
                        'session_id': session_id,
                        'logs': logs,
                        'timestamp': datetime.now().isoformat()
                    }))
                    continue

                # Now handle data messages (frames and text)
                # Prepare results container for combined ack
                combined_results = {}

                # Check for image inside data field
                data_field = data.get('data') or {}
                image_present = 'base64Image' in data_field

                if image_present:
                    # Update frame stats
                    stats['total_frames'] += 1
                    
                    # Log large messages
                    if message_size_kb > 500:
                        logger.warning(f"Large frame received: {message_size_kb:.1f} KB")

                    # Process image
                    frame_results = process_frame(data)
                    combined_results['frame'] = frame_results
                    
                    # Run active streaming tools asynchronously
                    if last_frame['image'] is not None and client_id in active_streaming_tools:
                        logger.info(f"About to run streaming tools for {client_id}")
                        # Run tool execution in background task to avoid blocking frame processing
                        task = asyncio.create_task(run_streaming_tools(websocket, client_id, last_frame['image'], last_frame['base64']))
                        # Add error handling to the task
                        task.add_done_callback(lambda t: logger.error(f"Streaming tool task error: {t.exception()}") if t.exception() else None)
                    elif last_frame['image'] is None:
                        logger.debug(f"No image available for {client_id}")
                    elif client_id not in active_streaming_tools:
                        logger.debug(f"No active streaming tools for {client_id}")
                    else:
                        logger.warning(f"Unknown condition - not running streaming tools for {client_id}")

                    # Log progress every 10 frames
                    if stats['total_frames'] % 10 == 0:
                        elapsed = (datetime.now() - stats['start_time']).total_seconds()
                        fps = stats['total_frames'] / elapsed if elapsed > 0 else 0
                        mb_received = stats['total_bytes'] / (1024 * 1024)
                        logger.info(
                            f"Frames: {stats['total_frames']}, "
                            f"FPS: {fps:.2f}, "
                            f"Data: {mb_received:.2f} MB"
                        )

                # Check for text either embedded in data or top-level
                text_payload = None
                if isinstance(data_field, dict):
                    text_payload = data_field.get('text') or data_field.get('caption')
                if not text_payload:
                    text_payload = data.get('text')

                if text_payload:
                    logger.info(f"[DEBUG] Text received: {text_payload[:100]}, current mode: {selected_issue.get('mode')}, issue: {selected_issue.get('number')}")
                    text_results = process_text(text_payload)
                    logger.info(f"[DEBUG] Text processing result: {text_results}, last_text content: {last_text.get('content', '')[:100] if last_text.get('content') else 'None'}")
                    combined_results['text'] = text_results
                    if text_results.get('status') in ('saved', 'received'):
                        stats['total_texts'] += 1

                # Handle issue_selection message type (for mode switching)
                if data.get('type') == 'issue_selection':
                    logger.info(f"Client {client_id} sent issue selection: {data}")
                    mode = data.get('mode', 'create')
                    
                    if mode == 'create':
                        selected_issue['mode'] = 'create'
                        selected_issue['number'] = None
                        selected_issue['title'] = None
                        
                        # Clear any pending text
                        last_text['content'] = None
                        last_text['prev_raw'] = ""
                        last_text['timestamp'] = None
                        last_text['task'] = None
                        
                        logger.info(f"Switched to CREATE mode and cleared text buffer")
                        
                        # Send confirmation
                        confirm_data = {
                            'type': 'mode_switched',
                            'mode': 'create',
                            'message': 'Switched to create new issue mode'
                        }
                        await websocket.send(json.dumps(confirm_data))
                    
                    elif mode == 'update':
                        issue_number = data.get('issue_number')
                        issue_title = data.get('issue_title', '')
                        
                        if issue_number:
                            selected_issue['mode'] = 'update'
                            selected_issue['number'] = issue_number
                            selected_issue['title'] = issue_title
                            
                            # Clear any pending text
                            last_text['content'] = None
                            last_text['prev_raw'] = ""
                            last_text['timestamp'] = None
                            last_text['task'] = None
                            
                            logger.info(f"Switched to UPDATE mode for issue #{issue_number} and cleared text buffer")
                            
                            # Fetch tools associated with this issue
                            logger.info(f"Fetching tools for issue #{issue_number}...")
                            issue_tools = fetch_issue_tools(issue_number)
                            
                            # Send confirmation
                            confirm_data = {
                                'type': 'mode_switched',
                                'mode': 'update',
                                'issue_number': issue_number,
                                'issue_title': issue_title,
                                'message': f"Switched to update mode for issue #{issue_number}",
                                'tools': issue_tools  # Include tools in the response
                            }
                            await websocket.send(json.dumps(confirm_data))
                        else:
                            logger.warning("Update mode requested but no issue_number provided")
                    
                    # Send acknowledgment back to client
                    await websocket.send(json.dumps({
                        'type': 'ack',
                        'mode': mode,
                        'timestamp': datetime.now().isoformat()
                    }))
                    continue

                # Handle run_tool message type (for executing tools)
                if data.get('type') == 'run_tool':
                    logger.info(f"Client {client_id} requested tool execution: {data.get('tool_name')}")
                    tool_name = data.get('tool_name', 'unknown')
                    tool_code = data.get('tool_code', '')
                    tool_language = data.get('tool_language', 'python')
                    tool_input = data.get('input', '')
                    frame_data = data.get('frame', None)  # Get optional frame data
                    conversation_id = data.get('conversation_id', None)  # Get conversation ID if this is a conversation
                    
                    logger.info(f"[RUN_TOOL] Received run_tool request for {tool_name}")
                    logger.info(f"[RUN_TOOL] Has frame_data: {frame_data is not None}")
                    if frame_data:
                        logger.info(f"[RUN_TOOL] Frame data keys: {frame_data.keys()}")
                        logger.info(f"[RUN_TOOL] Frame has base64: {'base64' in frame_data}")
                        if 'base64' in frame_data:
                            base64_preview = frame_data['base64'][:50] if frame_data['base64'] else 'empty'
                            logger.info(f"[RUN_TOOL] Frame base64 preview: {base64_preview}")
                            logger.info(f"[RUN_TOOL] Frame base64 length: {len(frame_data.get('base64', ''))}")
                    
                    if conversation_id:
                        logger.info(f"This is a conversation run with ID: {conversation_id}")
                    
                    # Parse input if it's JSON, otherwise wrap in dict with 'value' key
                    import json as json_module
                    parsed_input = tool_input
                    if isinstance(tool_input, str):
                        if tool_input.strip().startswith('{'):
                            # Try to parse as JSON
                            try:
                                parsed_input = json_module.loads(tool_input)
                            except:
                                # JSON parsing failed, wrap in dict
                                parsed_input = {'value': tool_input}
                        else:
                            # Plain string, wrap in dict so tools can use .get()
                            parsed_input = {'value': tool_input} if tool_input else {}
                    elif not isinstance(tool_input, dict):
                        # Not a string or dict, wrap it
                        parsed_input = {'value': tool_input}
                    
                    # Execute the tool (Python tools only for now)
                    if tool_language == 'python' and tool_code:
                        try:
                            # Use frame from message if provided, otherwise use last streaming frame
                            frame_image = None
                            frame_base64 = None
                            
                            if frame_data:
                                # One-shot mode: decode the frame sent with the message
                                logger.info(f"Using one-shot frame from message (size: {frame_data.get('width')}x{frame_data.get('height')})")
                                base64_data = frame_data.get('base64', '')
                                logger.info(f"ONE-SHOT: Base64 data length: {len(base64_data)}, starts with: {base64_data[:50] if base64_data else 'empty'}")
                                
                                # Decode base64 to image
                                if base64_data.startswith('data:image'):
                                    base64_data = base64_data.split(',')[1]
                                    logger.info(f"ONE-SHOT: Stripped data URL prefix, new length: {len(base64_data)}")
                                
                                try:
                                    image_bytes = base64.b64decode(base64_data)
                                    logger.info(f"ONE-SHOT: Decoded {len(image_bytes)} bytes")
                                    nparr = np.frombuffer(image_bytes, np.uint8)
                                    frame_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    frame_base64 = frame_data.get('base64', '')
                                    logger.info(f"ONE-SHOT: Decoded frame shape: {frame_image.shape if frame_image is not None else 'None - decode failed!'}")
                                    
                                    # If this is a conversation, save the image for follow-up questions
                                    if conversation_id and frame_image is not None:
                                        try:
                                            from PIL import Image
                                            import io
                                            # Convert CV2 image to PIL Image
                                            pil_image = Image.fromarray(cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB))
                                            conversation_images[conversation_id] = pil_image
                                            logger.info(f"CONVERSATION: Saved image for conversation {conversation_id}, size: {pil_image.size}")
                                        except Exception as e:
                                            logger.error(f"CONVERSATION: Error saving image: {e}")
                                    
                                    if frame_image is None:
                                        logger.error("ONE-SHOT: cv2.imdecode returned None! Falling back to streaming mode.")
                                        # Fall back to streaming mode
                                        if last_frame['image'] is not None:
                                            frame_image = last_frame['image']
                                            frame_base64 = last_frame['base64']
                                            logger.info("ONE-SHOT FALLBACK: Using last streaming frame")
                                        else:
                                            logger.error("ONE-SHOT FALLBACK: No streaming frame available either!")
                                except Exception as e:
                                    logger.error(f"ONE-SHOT: Error decoding frame: {e}")
                                    # Fall back to streaming mode
                                    if last_frame['image'] is not None:
                                        frame_image = last_frame['image']
                                        frame_base64 = last_frame['base64']
                                        logger.info("ONE-SHOT ERROR FALLBACK: Using last streaming frame")
                            
                            if not frame_data or frame_image is None:
                                # Streaming mode: use last received frame (or fallback from failed one-shot)
                                if frame_data:
                                    logger.info(f"ONE-SHOT FAILED: Attempting to use buffered frame as fallback")
                                else:
                                    logger.info(f"STREAMING MODE: Using last buffered frame")
                                    
                                if last_frame['image'] is None:
                                    await websocket.send(json.dumps({
                                        'type': 'tool_result',
                                        'tool_name': tool_name,
                                        'status': 'error',
                                        'error': 'No camera frame available. Please ensure camera is active or streaming.',
                                        'timestamp': datetime.now().isoformat()
                                    }))
                                    continue
                                
                                frame_image = last_frame['image']
                                frame_base64 = last_frame['base64']
                            
                            # Get module manager and load common modules dynamically
                            module_mgr = get_module_manager()
                            common_modules = module_mgr.get_common_modules()
                            
                            # Create a sandboxed execution environment with image data
                            exec_globals = {
                                '__builtins__': __builtins__,
                                'input_data': parsed_input,  # Use parsed input (dict or string)
                                'image': frame_image,  # OpenCV image (numpy array)
                                'image_base64': frame_base64,  # Base64 string
                                'yolo_model_cache': yolo_model_cache,  # Shared YOLO model cache for performance
                                **common_modules  # Dynamically loaded modules
                            }
                            
                            # Debug: Log what we're passing to the tool
                            logger.info(f"EXEC_GLOBALS: image is None: {frame_image is None}, image type: {type(frame_image)}, image shape: {frame_image.shape if frame_image is not None else 'N/A'}")
                            logger.info(f"EXEC_GLOBALS: image_base64 length: {len(frame_base64) if frame_base64 else 0}")
                            
                            exec_locals = {}
                            
                            # Capture stdout to get print() output
                            import io
                            import sys
                            stdout_capture = io.StringIO()
                            old_stdout = sys.stdout
                            
                            # Execute the tool code with retry on module errors
                            max_retries = 3
                            retry_count = 0
                            
                            while retry_count < max_retries:
                                try:
                                    # Redirect stdout to capture print statements
                                    sys.stdout = stdout_capture
                                    try:
                                        exec(tool_code, exec_globals, exec_locals)
                                    finally:
                                        sys.stdout = old_stdout
                                    break  # Success, exit retry loop
                                except (ImportError, ModuleNotFoundError) as e:
                                    error_msg = str(e)
                                    logger.warning(f"Module error in {tool_name}: {error_msg}")
                                    
                                    # Try to install the missing module
                                    installed_module = module_mgr.install_from_error(error_msg)
                                    
                                    if installed_module:
                                        logger.info(f"Installed {installed_module}, retrying execution...")
                                        # Reload common modules to include newly installed one
                                        common_modules = module_mgr.get_common_modules()
                                        exec_globals.update(common_modules)
                                        retry_count += 1
                                    else:
                                        # Could not identify/install module, raise error
                                        raise
                                except Exception:
                                    # Non-module errors should be raised immediately
                                    sys.stdout = old_stdout  # Restore stdout before raising
                                    raise
                            
                            # Get captured print output
                            printed_output = stdout_capture.getvalue()
                            
                            # IMPORTANT: Merge exec_locals into exec_globals so functions can see each other
                            # This allows main() to call helper functions defined in the same file
                            exec_globals.update(exec_locals)
                            
                            # Try to find a main function or use the result
                            # Check function signatures to avoid calling helper functions
                            import inspect
                            
                            result = None
                            
                            # Try main() - highest priority
                            if 'main' in exec_locals and callable(exec_locals['main']):
                                try:
                                    sig = inspect.signature(exec_locals['main'])
                                    if len(sig.parameters) == 2:  # Should take (image, input_data)
                                        result = exec_locals['main'](frame_image, parsed_input)
                                    else:
                                        logger.warning(f"main() has {len(sig.parameters)} params, expected 2")
                                except Exception as e:
                                    logger.error(f"Error calling main(): {e}")
                            
                            # Try run() - second priority
                            elif 'run' in exec_locals and callable(exec_locals['run']):
                                try:
                                    sig = inspect.signature(exec_locals['run'])
                                    if len(sig.parameters) == 2:
                                        result = exec_locals['run'](frame_image, parsed_input)
                                    else:
                                        logger.warning(f"run() has {len(sig.parameters)} params, expected 2")
                                except Exception as e:
                                    logger.error(f"Error calling run(): {e}")
                            
                            # Try process_image() - but check signature carefully
                            elif 'process_image' in exec_locals and callable(exec_locals['process_image']):
                                try:
                                    sig = inspect.signature(exec_locals['process_image'])
                                    # Only call if it takes 2 params (entry point), not 1 (helper function)
                                    if len(sig.parameters) == 2:
                                        result = exec_locals['process_image'](frame_image, parsed_input)
                                    else:
                                        logger.info(f"Skipping process_image() - has {len(sig.parameters)} params (likely a helper function)")
                                except Exception as e:
                                    logger.error(f"Error calling process_image(): {e}")
                            
                            # Try process_frame_for_text() - special case for text tools
                            elif 'process_frame_for_text' in exec_locals and callable(exec_locals['process_frame_for_text']):
                                try:
                                    # Initialize verbalizer if needed
                                    if 'initialize_verbalizer' in exec_locals and 'get_verbalizer' in exec_locals:
                                        verbalizer = exec_locals['get_verbalizer']()
                                        if not verbalizer and GEMINI_API_KEY:
                                            exec_locals['initialize_verbalizer'](GEMINI_API_KEY)
                                    result = exec_locals['process_frame_for_text'](frame_base64)
                                except Exception as e:
                                    logger.error(f"Error calling process_frame_for_text(): {e}")
                            
                            # Check for result variable
                            elif 'result' in exec_locals:
                                result = exec_locals['result']
                            else:
                                # Tool is a library - show available functions
                                available_funcs = [name for name, obj in exec_locals.items() 
                                                 if callable(obj) and not name.startswith('_')]
                                available_classes = [name for name, obj in exec_locals.items() 
                                                   if isinstance(obj, type) and not name.startswith('_')]
                                
                                if available_funcs or available_classes:
                                    result = f"Tool loaded successfully.\n\nAvailable functions: {', '.join(available_funcs) if available_funcs else 'none'}\nAvailable classes: {', '.join(available_classes) if available_classes else 'none'}\n\nNote: This tool is a library. To use it, you may need to call one of these functions."
                                else:
                                    result = None
                            
                            # Combine printed output with result
                            # Check if result is a dict with 'audio' and 'text' keys (advanced format)
                            if isinstance(result, dict) and ('audio' in result or 'text' in result):
                                # Advanced format - preserve audio configuration
                                response_data = {
                                    'type': 'tool_result',
                                    'tool_name': tool_name,
                                    'status': 'success',
                                    'result': result.get('text', ''),
                                    'timestamp': datetime.now().isoformat()
                                }
                                
                                # Add audio configuration if provided
                                if 'audio' in result:
                                    response_data['audio'] = result['audio']
                                
                                # Prepend printed output to text if any
                                if printed_output.strip():
                                    response_data['result'] = printed_output.strip() + '\n' + response_data['result']
                                    # Also update audio text to include printed output
                                    if 'audio' in response_data and 'text' in response_data['audio']:
                                        response_data['audio']['text'] = printed_output.strip() + '. ' + response_data['audio']['text']
                                
                                json_str = json.dumps(response_data)
                                logger.info(f"Sending tool_result: result length={len(response_data.get('result', ''))}, audio.text length={len(response_data.get('audio', {}).get('text', ''))}")
                                await websocket.send(json_str)
                            else:
                                # Simple format - just strings
                                output_parts = []
                                if printed_output.strip():
                                    output_parts.append(printed_output.strip())
                                if result is not None:
                                    output_parts.append(str(result))
                                
                                final_result = '\n'.join(output_parts) if output_parts else "Tool executed (no output)"
                                
                                # Send result back to client (with default audio config)
                                await websocket.send(json.dumps({
                                    'type': 'tool_result',
                                    'tool_name': tool_name,
                                    'status': 'success',
                                    'result': final_result,
                                    'audio': {
                                        'type': 'speech',
                                        'text': final_result,
                                        'rate': 1.0,
                                        'interrupt': False
                                    },
                                    'timestamp': datetime.now().isoformat()
                                }))
                            
                            logger.info(f"Tool {tool_name} executed successfully")
                            
                        except Exception as e:
                            logger.error(f"Error executing tool {tool_name}: {e}")
                            await websocket.send(json.dumps({
                                'type': 'tool_result',
                                'tool_name': tool_name,
                                'status': 'error',
                                'error': str(e),
                                'timestamp': datetime.now().isoformat()
                            }))
                    else:
                        await websocket.send(json.dumps({
                            'type': 'tool_result',
                            'tool_name': tool_name,
                            'status': 'error',
                            'error': f'Unsupported language: {tool_language}',
                            'timestamp': datetime.now().isoformat()
                        }))
                    continue

                # Handle register_conversation_image message type
                if data.get('type') == 'register_conversation_image':
                    logger.info(f"Client {client_id} registering conversation image")
                    conversation_id = data.get('conversation_id', '')
                    image_base64 = data.get('image_base64', '')
                    
                    if not conversation_id or not image_base64:
                        logger.error("Missing conversation_id or image_base64")
                        continue
                    
                    try:
                        from PIL import Image
                        import io
                        
                        logger.info(f"[REGISTER_IMG] Received base64 length: {len(image_base64)}")
                        logger.info(f"[REGISTER_IMG] Base64 preview: {image_base64[:100]}")
                        
                        # Frontend already strips the data URI prefix, so we just clean whitespace
                        # Remove any whitespace that might have been introduced during transmission
                        image_base64 = image_base64.strip().replace('\n', '').replace('\r', '').replace(' ', '')
                        logger.info(f"[REGISTER_IMG] After cleanup, length: {len(image_base64)}")
                        logger.info(f"[REGISTER_IMG] After cleanup preview: {image_base64[:100]}")
                        
                        # Decode base64 image
                        image_bytes = base64.b64decode(image_base64)
                        logger.info(f"[REGISTER_IMG] Decoded {len(image_bytes)} bytes")
                        logger.info(f"[REGISTER_IMG] First 20 bytes (hex): {image_bytes[:20].hex()}")
                        logger.info(f"[REGISTER_IMG] First 20 bytes (repr): {repr(image_bytes[:20])}")
                        
                        # Check for JPEG magic bytes (FF D8 FF)
                        if len(image_bytes) >= 3 and image_bytes[:3] == b'\xff\xd8\xff':
                            logger.info(f"[REGISTER_IMG] Valid JPEG magic bytes detected")
                        else:
                            logger.error(f"[REGISTER_IMG] Invalid JPEG magic bytes! Expected FF D8 FF, got: {image_bytes[:3].hex()}")
                        
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        
                        # Store in conversation_images dictionary
                        conversation_images[conversation_id] = pil_image
                        logger.info(f"CONVERSATION: Registered image for conversation {conversation_id}, size: {pil_image.size}")
                        
                        # Send confirmation
                        await websocket.send(json.dumps({
                            'type': 'conversation_image_registered',
                            'conversation_id': conversation_id,
                            'status': 'success'
                        }))
                    except Exception as e:
                        logger.error(f"Error registering conversation image: {e}")
                        logger.error(f"[REGISTER_IMG] Exception type: {type(e).__name__}")
                        logger.error(f"[REGISTER_IMG] Exception details: {str(e)}")
                        import traceback
                        logger.error(f"[REGISTER_IMG] Traceback: {traceback.format_exc()}")
                        await websocket.send(json.dumps({
                            'type': 'conversation_image_registered',
                            'conversation_id': conversation_id,
                            'status': 'error',
                            'error': str(e)
                        }))
                    continue

                # Handle follow_up_question message type (for chat conversations)
                if data.get('type') == 'follow_up_question':
                    logger.info(f"Client {client_id} sent follow-up question")
                    question = data.get('question', '')
                    conversation_id = data.get('conversation_id', '')
                    
                    if not question.strip():
                        await websocket.send(json.dumps({
                            'type': 'follow_up_response',
                            'status': 'error',
                            'error': 'No question provided',
                            'timestamp': datetime.now().isoformat()
                        }))
                        continue
                    
                    try:
                        # Get the stored image for this conversation
                        pil_image = conversation_images.get(conversation_id)
                        
                        if pil_image is not None:
                            logger.info(f"FOLLOW-UP: Found stored image for conversation {conversation_id}, size: {pil_image.size}")
                            
                            # Create prompt for follow-up question
                            prompt = f"You are analyzing this image. The user is asking a follow-up question: {question}\n\nPlease provide a helpful and concise answer based on what you can see in the image."
                            
                            # Send to Gemini using gemini-2.5-flash
                            try:
                                model = genai.GenerativeModel('gemini-2.5-flash')
                                response = model.generate_content([prompt, pil_image])
                                answer = response.text.strip()
                                
                                logger.info(f"FOLLOW-UP: Generated answer length: {len(answer)}")
                                logger.info(f"FOLLOW-UP: Successfully used gemini-2.5-flash for image+text")
                                
                                await websocket.send(json.dumps({
                                    'type': 'follow_up_response',
                                    'status': 'success',
                                    'response': answer,
                                    'timestamp': datetime.now().isoformat()
                                }))
                                
                            except Exception as e:
                                logger.error(f"Error getting Gemini response for follow-up: {e}")
                                await websocket.send(json.dumps({
                                    'type': 'follow_up_response',
                                    'status': 'error',
                                    'error': f'Error generating response: {str(e)}',
                                    'timestamp': datetime.now().isoformat()
                                }))
                        else:
                            # No image found for this conversation
                            logger.warning(f"FOLLOW-UP: No image found for conversation {conversation_id}")
                            answer = "I cannot find the image for this conversation. It may have expired. Please start a new conversation with a fresh image."
                            
                            await websocket.send(json.dumps({
                                'type': 'follow_up_response',
                                'status': 'success',
                                'response': answer,
                                'timestamp': datetime.now().isoformat()
                            }))
                            
                    except Exception as e:
                        logger.error(f"Error handling follow-up question: {e}")
                        await websocket.send(json.dumps({
                            'type': 'follow_up_response',
                            'status': 'error',
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }))
                    continue

                # If neither image nor text were found, warn
                if not image_present and not text_payload:
                    logger.warning(f"Unknown or empty message received from {client_id}: {data}")
                    continue

                # Send combined acknowledgment back to client
                await websocket.send(json.dumps({
                    'type': 'ack',
                    'frame_number': data.get('frameNumber'),
                    'results': combined_results
                }))

            except json.JSONDecodeError as e:
                # Log full error and offending message to session log and main logger
                tb = traceback.format_exc()
                if client_id in active_session_loggers:
                    active_session_loggers[client_id].log('ERROR', f'Invalid JSON: {e}\nMessage: {message}')
                    active_session_loggers[client_id].log('ERROR', tb)
                logger.error(f"Invalid JSON: {e} - Message: {message}")
                logger.error(tb)
            except Exception as e:
                tb = traceback.format_exc()
                if client_id in active_session_loggers:
                    active_session_loggers[client_id].log('ERROR', f'Error handling message: {e}\nMessage: {message}')
                    active_session_loggers[client_id].log('ERROR', tb)
                logger.error(f"Error handling message: {e}")
                logger.error(tb)

    except Exception as e:
        # Import websockets module to avoid shadowing by parameter name
        import websockets as ws_module
        if isinstance(e, ws_module.exceptions.ConnectionClosed):
            logger.info(f"Client disconnected: {client_id} (code: {e.code}, reason: {e.reason})")
            if client_id in active_session_loggers:
                active_session_loggers[client_id].log("INFO", f"Disconnected (code: {e.code}, reason: {e.reason})")
        else:
            logger.error(f"Error with client {client_id}: {e}", exc_info=True)
            if client_id in active_session_loggers:
                active_session_loggers[client_id].log("ERROR", f"Connection error: {e}")
    finally:
        connected_clients.discard(websocket)
        # Clean up streaming tools for this client
        if client_id in active_streaming_tools:
            logger.info(f"Stopping streaming tool for disconnected client {client_id}")
            # Clean up Gemini Live session if active
            if active_streaming_tools[client_id].get('gemini_live') and gemini_live_manager:
                await gemini_live_manager.stop_session(client_id)
            del active_streaming_tools[client_id]
        # Clean up any active Copilot streams for this client
        if client_id in active_copilot_streams:
            logger.info(f"Stopping Copilot stream for disconnected client {client_id}")
            await stop_copilot_session_stream(client_id)
        # Detach websocket from pending Copilot polls (don't kill them — the
        # poll keeps running and stores results in the DB so a reconnecting
        # client can pick them up)
        for issue_num, info in pending_copilot_issues.items():
            if info.get('client_id') == client_id:
                info['websocket'] = None  # detach, don't delete
                info['client_id'] = None
                logger.info(f"Detached client from pending Copilot poll for issue #{issue_num} (client disconnected)")
        # Close and clean up session logger
        if client_id in active_session_loggers:
            active_session_loggers[client_id].close()
            del active_session_loggers[client_id]
        logger.info(f"Cleaned up client: {client_id}")


async def broadcast_stats():
    """
    Periodically broadcast server statistics to all clients
    """
    try:
        while True:
            await asyncio.sleep(5)
            
            if connected_clients and stats['start_time']:
                elapsed = (datetime.now() - stats['start_time']).total_seconds()
                fps = stats['total_frames'] / elapsed if elapsed > 0 else 0
                
                message = json.dumps({
                    'type': 'stats',
                    'total_frames': stats['total_frames'],
                    'total_texts': stats.get('total_texts', 0),
                    'fps': round(fps, 2),
                    'clients': len(connected_clients),
                    'uptime': round(elapsed, 1)
                })
                
                # Send to all connected clients
                websockets.broadcast(connected_clients, message)
    except Exception as e:
        logger.error(f"Error in broadcast_stats: {e}", exc_info=True)


# Global tracking for background-monitored sessions
monitored_sessions = set()  # Track session IDs we're already monitoring
monitored_prs = set()  # Track PR numbers we've checked


async def monitor_copilot_sessions():
    """
    Background task to continuously monitor for new Copilot sessions and capture their logs.
    Checks:
    1. All open PRs for Copilot agent tasks
    2. Recent @copilot comments that spawn sessions
    3. Any active sessions we haven't captured yet
    
    This ensures we have historical logs for all sessions, even if no one explicitly requests them.
    """
    global monitored_sessions, monitored_prs
    
    if not GITHUB_TOKEN:
        logger.warning("GitHub token not available, skipping Copilot session monitoring")
        return
    
    logger.info("Starting background Copilot session monitoring")
    check_interval = 60  # Check every minute
    
    # Create clean environment for gh CLI
    env = os.environ.copy()
    env.pop('GH_TOKEN', None)
    env.pop('GITHUB_TOKEN', None)
    
    while True:
        try:
            await asyncio.sleep(check_interval)
            
            g = Github(GITHUB_TOKEN)
            repo = g.get_repo(GITHUB_REPO)
            
            # 1. Check all open PRs for Copilot sessions
            pulls = repo.get_pulls(state='open', sort='updated', direction='desc')
            for pr in list(pulls)[:20]:  # Check 20 most recently updated PRs
                if pr.number in monitored_prs:
                    continue  # Already checked this PR
                
                try:
                    # Check if PR has Copilot agent activity (created by copilot or mentions copilot)
                    is_copilot_pr = (
                        pr.user.login == 'github-actions[bot]' or
                        'copilot' in (pr.title + (pr.body or '')).lower() or
                        any('copilot' in comment.body.lower() for comment in pr.get_issue_comments()[:10])
                    )
                    
                    if is_copilot_pr:
                        logger.info(f"Found potential Copilot PR #{pr.number}: {pr.title}")
                        
                        # Try to find session ID for this PR
                        proc = await asyncio.create_subprocess_exec(
                            'gh', 'agent-task', 'view',
                            str(pr.number),
                            '-R', GITHUB_REPO,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=env
                        )
                        stdout, stderr = await proc.communicate()
                        output = stdout.decode('utf-8', errors='replace')
                        
                        # Extract session ID from output
                        import re
                        uuid_pattern = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
                        matches = re.findall(uuid_pattern, output)
                        
                        if matches:
                            session_id = matches[0]
                            if session_id not in monitored_sessions:
                                logger.info(f"Starting background capture for session {session_id} (PR #{pr.number})")
                                monitored_sessions.add(session_id)
                                monitored_prs.add(pr.number)
                                
                                # Start background log capture (no websocket needed)
                                asyncio.create_task(
                                    capture_session_logs_background(session_id, pr.number)
                                )
                        else:
                            # Mark as checked even if no session found
                            monitored_prs.add(pr.number)
                            
                except Exception as e:
                    logger.warning(f"Error checking PR #{pr.number} for Copilot session: {e}")
            
            # 2. Check for recent @copilot comments
            # Get recent issue comments (issues include PRs in GitHub API)
            issues = repo.get_issues(state='open', sort='updated', direction='desc')
            for issue in list(issues)[:30]:  # Check 30 most recent
                try:
                    comments = issue.get_comments()
                    for comment in list(comments)[-5:]:  # Last 5 comments
                        if '@copilot' in comment.body.lower():
                            # This might have spawned a session
                            pr_number = issue.number if issue.pull_request else None
                            if pr_number and pr_number not in monitored_prs:
                                logger.info(f"Found @copilot comment in PR #{pr_number}, checking for session...")
                                monitored_prs.add(pr_number)
                                
                                # Check for session (same logic as above)
                                proc = await asyncio.create_subprocess_exec(
                                    'gh', 'agent-task', 'view',
                                    str(pr_number),
                                    '-R', GITHUB_REPO,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE,
                                    env=env
                                )
                                stdout, stderr = await proc.communicate()
                                output = stdout.decode('utf-8', errors='replace')
                                
                                matches = re.findall(uuid_pattern, output)
                                if matches:
                                    session_id = matches[0]
                                    if session_id not in monitored_sessions:
                                        logger.info(f"Starting background capture for @copilot session {session_id} (PR #{pr_number})")
                                        monitored_sessions.add(session_id)
                                        asyncio.create_task(
                                            capture_session_logs_background(session_id, pr_number)
                                        )
                except Exception as e:
                    logger.warning(f"Error checking issue #{issue.number} for @copilot comments: {e}")
            
            logger.debug(f"Background monitor: Tracking {len(monitored_sessions)} sessions, {len(monitored_prs)} PRs")
            
        except Exception as e:
            logger.error(f"Error in monitor_copilot_sessions: {e}", exc_info=True)
            await asyncio.sleep(check_interval)


async def capture_session_logs_background(session_id: str, pr_number: int):
    """
    Capture logs for a Copilot session in the background (no WebSocket client needed).
    Stores everything in the database for later retrieval.
    
    Args:
        session_id: Copilot session ID
        pr_number: Associated PR number
    """
    global monitored_sessions
    
    logger.info(f"Background log capture started for session {session_id} (PR #{pr_number})")
    
    try:
        # Create session in database
        copilot_db.create_session(session_id, pr_number)
        
        # Fetch and store historical logs
        await fetch_and_store_session_logs(session_id, pr_number)
        
        logger.info(f"Background log capture completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error in background log capture for session {session_id}: {e}")
    finally:
        # Keep session in monitored_sessions so we don't try to capture it again
        pass


async def main():
    """
    Start the WebSocket server
    """
    logger.info(f"Starting WebSocket server on {HOST}:{PORT}")
    logger.info(f"Frame saving: {'enabled' if SAVE_FRAMES else 'disabled'}")
    logger.info(f"GitHub issue creation: {'enabled' if GITHUB_TOKEN else 'disabled'}")
    
    # Start server with increased message size limit (20MB to handle large base64 images)
    # Default is 1MB which may be too small for high-quality photos
    async with websockets.serve(
        handle_client, 
        HOST, 
        PORT,
        max_size=20 * 1024 * 1024,  # 20MB max message size
        ping_interval=20,  # Send ping every 20 seconds
        ping_timeout=10    # Wait 10 seconds for pong response
    ):
        logger.info("Server started successfully")
        logger.info(f"Clients can connect to: ws://34.144.178.116:{PORT}")
        logger.info(f"Max message size: 20MB")
        
        # Start background tasks independently for resilience
        asyncio.create_task(broadcast_stats())
        asyncio.create_task(monitor_text_pause())
        # Background Copilot monitoring disabled - we poll specific PRs when user comments with @copilot
        
        # Keep server running
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise