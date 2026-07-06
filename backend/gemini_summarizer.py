"""
Gemini-based summarization for Copilot session logs.
"""
import logging
from typing import List, Dict
from litellm_utils import extract_text
from model_router import system_llm_call

logger = logging.getLogger(__name__)


async def summarize_entries(entries: List[Dict]) -> str:
    """
    Summarize a batch of Copilot log entries using Gemini.
    
    Args:
        entries: List of entry dictionaries with 'text' and 'is_code' fields
        
    Returns:
        A concise summary suitable for text-to-speech
    """
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
        response = system_llm_call(
            messages=[{'role': 'user', 'content': prompt}],
        )
        summary = extract_text(response)
        
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
        logger.error(f"Error generating summary with LiteLLM: {e}")
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
        response = system_llm_call(
            messages=[{'role': 'user', 'content': prompt}],
        )
        summary = extract_text(response)
        
        # Remove any quotes or extra formatting
        summary = summary.strip('"').strip("'").strip()
        
        # Ensure it's not too long
        if len(summary) > 200:
            # Truncate at last complete word
            summary = summary[:197] + "..."
        
        logger.info(f"Generated summary: {summary}")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary with LiteLLM: {e}")
        # Fallback: use first entry text truncated
        fallback = non_code_entries[0]['text'][:100]
        return f"{fallback}..." if len(non_code_entries[0]['text']) > 100 else fallback
