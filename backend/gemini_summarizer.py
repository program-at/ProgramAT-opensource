"""
Gemini-based summarization for Copilot session logs.
"""
import os
import logging
import google.generativeai as genai
from typing import List, Dict

logger = logging.getLogger(__name__)

# Lazy initialization - don't configure until first use
_model = None
_model_initialized = False


def _get_model():
    """Lazy initialization of Gemini model."""
    global _model, _model_initialized
    
    if _model_initialized:
        return _model
    
    _model_initialized = True
    
    # Get configuration at runtime (after .env is loaded)
    api_key = os.environ.get('GEMINI_API_KEY')
    model_name = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash-lite')  # Match stream_server.py default
    
    if not api_key:
        logger.warning("GEMINI_API_KEY not found, summarization will be disabled")
        return None
    
    try:
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(model_name)
        logger.info(f"Gemini model initialized: {model_name}")
        return _model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        return None


async def summarize_entries(entries: List[Dict]) -> str:
    """
    Summarize a batch of Copilot log entries using Gemini.
    
    Args:
        entries: List of entry dictionaries with 'text' and 'is_code' fields
        
    Returns:
        A concise summary suitable for text-to-speech
    """
    model = _get_model()
    if not model:
        logger.warning("Gemini model not available, returning placeholder summary")
        return "Copilot is processing..."
    
    if not entries:
        return "Copilot is processing..."
    
    # Combine ALL entry texts (including code)
    combined_text = "\n\n".join([e['text'] for e in entries])
    
    # Create prompt optimized for screen reader users
    prompt = f"""You are summarizing GitHub Copilot agent activity logs for a blind developer using a screen reader.

The logs describe what the Copilot agent is doing while working on their code. They may include both explanatory text and code snippets.

Your task: Summarize in 1-3 sentences (maximum 50 words) what Copilot accomplished:
- Describe SPECIFICALLY what was done (e.g., "Added error handling to the login function" NOT "Generated code")
- If code was written, explain what functionality it implements
- If files were modified, mention which ones and why
- If errors were fixed, explain what was corrected
- Use clear, natural language suitable for being read aloud
- Focus on what the developer needs to know about the progress

NEVER just say "generated code" or "wrote code" without explaining what it does.

Log entries:
{combined_text}

Summary (1-3 sentences describing what was accomplished):"""
    
    try:
        response = model.generate_content(prompt)
        summary = response.text.strip()
        
        # Remove any quotes or extra formatting
        summary = summary.strip('"').strip("'").strip()
        
        # Ensure it's not too long (but allow more space for detail)
        if len(summary) > 350:
            # Truncate at last complete sentence or word
            sentences = summary[:350].split('. ')
            if len(sentences) > 1:
                summary = '. '.join(sentences[:-1]) + '.'
            else:
                summary = summary[:347] + "..."
        
        logger.info(f"Generated summary: {summary}")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary with Gemini: {e}")
        # Fallback: use first entry text truncated
        fallback = entries[0]['text'][:150]
        return f"{fallback}..." if len(entries[0]['text']) > 150 else fallback


def summarize_entries_sync(entries: List[Dict]) -> str:
    """
    Synchronous version of summarize_entries for non-async contexts.
    
    Args:
        entries: List of entry dictionaries with 'text' and 'is_code' fields
        
    Returns:
        A concise summary suitable for text-to-speech
    """
    model = _get_model()
    if not model:
        logger.warning("Gemini model not available, returning placeholder summary")
        return "Copilot is processing..."
    
    # Filter out code entries
    non_code_entries = [e for e in entries if not e.get('is_code', False)]
    
    if not non_code_entries:
        return "Copilot generated code."
    
    # Combine entry texts
    combined_text = "\n\n".join([e['text'] for e in non_code_entries])
    
    # Create prompt optimized for screen reader users
    prompt = f"""You are summarizing GitHub Copilot agent activity logs for a blind developer using a screen reader.

The logs describe what the Copilot agent is doing while working on their code.

Your task: Summarize the following log entries in ONE concise sentence (maximum 15 words) that:
- Is natural and clear when read aloud
- Focuses on the main action/progress
- Uses present tense
- Avoids technical jargon when possible

Log entries:
{combined_text}

Summary (one sentence only):"""
    
    try:
        response = model.generate_content(prompt)
        summary = response.text.strip()
        
        # Remove any quotes or extra formatting
        summary = summary.strip('"').strip("'").strip()
        
        # Ensure it's not too long
        if len(summary) > 200:
            # Truncate at last complete word
            summary = summary[:197] + "..."
        
        logger.info(f"Generated summary: {summary}")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary with Gemini: {e}")
        # Fallback: use first entry text truncated
        fallback = non_code_entries[0]['text'][:100]
        return f"{fallback}..." if len(non_code_entries[0]['text']) > 100 else fallback
