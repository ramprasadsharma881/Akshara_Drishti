"""
AksharaDrishti - AI Spell Correction with Gemini via Helicone
==============================================================
Uses Gemini AI through Helicone Gateway to fix OCR spelling errors
in Kannada, Telugu, and Sanskrit Vedantic texts.

Features:
- Conservative correction (only fix obvious OCR errors)
- Preserves technical/religious terms
- Context-aware corrections
- Graceful error handling
- Uses Helicone for caching and rate limit management
"""

import os
import httpx
import json
import re
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - HELICONE PROXY FOR GEMINI
# ============================================================================

# Helicone Gateway endpoint (proxies to Gemini)
HELICONE_GATEWAY = "https://gateway.helicone.ai"

# Target Gemini API base URL (Helicone needs this to know where to forward requests)
GEMINI_TARGET_URL = "https://generativelanguage.googleapis.com"

# Full endpoint for Gemini 2.0 Flash (latest model)
# Note: Model name must include version suffix
GEMINI_MODEL_PATH = "/v1beta/models/gemini-2.0-flash:generateContent"

# Timeout for API calls (seconds)
API_TIMEOUT = 60  # Increased for reliability

# Maximum text length to process at once (to avoid token limits)
MAX_TEXT_LENGTH = 3000

# Enable/disable spell checking (can be controlled via env var)
SPELL_CHECK_ENABLED = os.environ.get("SPELL_CHECK_ENABLED", "true").lower() == "true"

# Rate limiting settings (Gemini free tier: ~15 RPM)
# We'll be very conservative to avoid 429 errors
REQUESTS_PER_MINUTE = 10  # Stay under the limit
REQUEST_DELAY_SECONDS = 60 / REQUESTS_PER_MINUTE  # ~6 seconds between requests
MAX_RETRIES = 3  # Max retries on rate limit before giving up

# Track last request time for rate limiting
import time
_last_request_time = 0


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

SPELL_CHECK_PROMPT = """You are an expert proofreader for Vedantic and philosophical texts in Kannada, Telugu, and Sanskrit scripts.

TASK: Fix ONLY clear OCR (optical character recognition) spelling errors in the following text. 

IMPORTANT RULES:
1. ONLY fix obvious spelling mistakes caused by OCR errors
2. DO NOT change any Sanskrit technical terms, mantras, or proper nouns
3. DO NOT rephrase, rewrite, or add any new content
4. DO NOT change sentence structure or grammar
5. PRESERVE all punctuation including । and ॥ (Sanskrit dandas)
6. If you're unsure about a word, LEAVE IT UNCHANGED
7. Return ONLY the corrected text, nothing else - no explanations

Common OCR errors to fix:
- Similar-looking characters confused (e.g., ನ/ಸ, క/ఖ, म/भ)
- Missing or extra vowel marks (matras)
- Broken consonant clusters
- Misread conjunct consonants

TEXT TO CORRECT:
{text}

CORRECTED TEXT:"""


# ============================================================================
# API FUNCTIONS
# ============================================================================

def get_helicone_key() -> Optional[str]:
    """
    Get Helicone API key from environment variable.
    
    Note: If you've configured your Gemini API key in the Helicone dashboard,
    you don't need a separate GEMINI_API_KEY - Helicone will use the stored one.
    """
    return os.environ.get("HELICONE_API_KEY")


def _wait_for_rate_limit():
    """Wait if needed to respect rate limits."""
    global _last_request_time
    
    if _last_request_time > 0:
        elapsed = time.time() - _last_request_time
        if elapsed < REQUEST_DELAY_SECONDS:
            wait_time = REQUEST_DELAY_SECONDS - elapsed
            logger.info(f"Rate limiting: waiting {wait_time:.1f}s before next request...")
            time.sleep(wait_time)
    
    _last_request_time = time.time()


def call_gemini_via_helicone(text: str, helicone_key: str) -> Optional[str]:
    """
    Call Gemini API via Helicone Gateway.
    
    Since the Gemini API key is configured in Helicone dashboard,
    we only need the Helicone key - Helicone handles the Gemini auth.
    
    Args:
        text: Text to spell-check
        helicone_key: Helicone API key
    
    Returns:
        Corrected text or None if failed
    """
    # Wait for rate limit
    _wait_for_rate_limit()
    
    prompt = SPELL_CHECK_PROMPT.format(text=text)
    
    # Request payload (Gemini native format)
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,  # Low temperature for consistent corrections
            "topK": 1,
            "topP": 0.8,
            "maxOutputTokens": 4096
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
    }
    
    # Headers for Helicone Gateway
    # Since Gemini key is stored in Helicone dashboard, we don't need x-goog-api-key
    # Helicone will use the provider key from your dashboard
    headers = {
        "Content-Type": "application/json",
        "Helicone-Auth": f"Bearer {helicone_key}",
        "Helicone-Target-Url": GEMINI_TARGET_URL,
        "Helicone-Cache-Enabled": "true",
    }
    
    # Full endpoint URL
    endpoint = f"{HELICONE_GATEWAY}{GEMINI_MODEL_PATH}"
    
    # Retry logic with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=API_TIMEOUT) as client:
                response = client.post(
                    endpoint,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract text from Gemini response
                    if "candidates" in result and len(result["candidates"]) > 0:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            parts = candidate["content"]["parts"]
                            if len(parts) > 0 and "text" in parts[0]:
                                logger.info("Gemini spell check successful")
                                return parts[0]["text"].strip()
                    logger.warning(f"Unexpected Gemini response structure")
                    return None
                    
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = (attempt + 1) * 15  # 15s, 30s, 45s
                    logger.warning(f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    logger.error(f"Gemini API error: {response.status_code} - {response.text[:200]}")
                    return None
                    
        except httpx.TimeoutException:
            logger.error("Gemini API timeout")
            return None
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return None
    
    logger.error("Max retries exceeded for Gemini API")
    return None


# ============================================================================
# MAIN SPELL CHECK FUNCTION
# ============================================================================

def spell_check_text(text: str) -> Tuple[str, bool]:
    """
    Main spell-checking function. Uses Gemini via Helicone Gateway.
    
    Since you've configured your Gemini API key in Helicone dashboard,
    we only need the HELICONE_API_KEY - no separate GEMINI_API_KEY needed!
    
    Args:
        text: Text to spell-check
    
    Returns:
        Tuple of (corrected_text, was_corrected)
        - If correction succeeds: (corrected_text, True)
        - If correction fails or disabled: (original_text, False)
    """
    if not text or len(text.strip()) < 10:
        return text, False
    
    if not SPELL_CHECK_ENABLED:
        logger.info("Spell checking disabled via environment variable")
        return text, False
    
    helicone_key = get_helicone_key()
    
    if not helicone_key:
        logger.warning("HELICONE_API_KEY not found. Spell check disabled.")
        return text, False
    
    # Split long text into chunks
    if len(text) > MAX_TEXT_LENGTH:
        chunks = split_text_into_chunks(text, MAX_TEXT_LENGTH)
        corrected_chunks = []
        any_corrected = False
        
        for chunk in chunks:
            corrected_chunk, was_corrected = spell_check_single_chunk(chunk, helicone_key)
            corrected_chunks.append(corrected_chunk)
            if was_corrected:
                any_corrected = True
        
        return '\n'.join(corrected_chunks), any_corrected
    else:
        return spell_check_single_chunk(text, helicone_key)


def spell_check_single_chunk(text: str, helicone_key: str) -> Tuple[str, bool]:
    """
    Spell-check a single chunk of text using Helicone Gateway.
    """
    logger.info("Calling Gemini via Helicone Gateway...")
    corrected = call_gemini_via_helicone(text, helicone_key)
    
    if corrected:
        # Validate correction (basic sanity check)
        if validate_correction(text, corrected):
            return corrected, True
        else:
            logger.warning("Correction failed validation, using original text")
            return text, False
    
    return text, False


def split_text_into_chunks(text: str, max_length: int) -> list:
    """
    Split text into chunks at paragraph boundaries.
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        if current_length + para_length > max_length and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length + 2  # +2 for \n\n
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def validate_correction(original: str, corrected: str) -> bool:
    """
    Validate that the correction is reasonable.
    
    Checks:
    1. Not too different from original (< 30% change)
    2. Not empty
    3. Contains similar script characters
    """
    if not corrected or len(corrected.strip()) == 0:
        return False
    
    # Check length ratio (shouldn't be too different)
    length_ratio = len(corrected) / len(original) if len(original) > 0 else 0
    if length_ratio < 0.5 or length_ratio > 2.0:
        logger.warning(f"Length ratio {length_ratio} is suspicious")
        return False
    
    # Check that key script characters are preserved
    # Count Indic script characters in both
    original_indic = len(re.findall(r'[\u0900-\u097F\u0C00-\u0C7F\u0C80-\u0CFF]', original))
    corrected_indic = len(re.findall(r'[\u0900-\u097F\u0C00-\u0C7F\u0C80-\u0CFF]', corrected))
    
    if original_indic > 10:  # Only check if original has significant Indic text
        indic_ratio = corrected_indic / original_indic
        if indic_ratio < 0.7:
            logger.warning(f"Indic character ratio {indic_ratio} is suspicious")
            return False
    
    return True


# ============================================================================
# BATCH PROCESSING FOR PAGES
# ============================================================================

def spell_check_page_text(text: str) -> Tuple[str, bool, str]:
    """
    Spell-check text for a single page.
    
    Args:
        text: Page text to spell-check
    
    Returns:
        Tuple of (corrected_text, was_corrected, status_message)
    """
    if not text or len(text.strip()) < 20:
        return text, False, "Text too short, skipping spell check"
    
    try:
        corrected, was_corrected = spell_check_text(text)
        
        if was_corrected:
            return corrected, True, "AI spell check applied"
        else:
            return text, False, "No corrections needed or API unavailable"
            
    except Exception as e:
        logger.error(f"Spell check error: {str(e)}")
        return text, False, f"Spell check error: {str(e)}"


# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test_spell_check():
    """Test the spell check functionality."""
    
    test_text = """
    ದಕ್ಷಿಣಾಮೂರ್ತಿಸ್ನೋತ್ರ
    
    ಓಂ ನಮಃ ಪ್ರಣವಾರ್ಥಾಯ ಶುದ್ಧಜ್ಞಾನೈಕಮೂರ್ತಯೇ
    ನಿರ್ಮಲಾಯ ಪ್ರಶಾಂತಾಯ ದಕ್ಷಿಣಾಮೂರ್ತಯೇ ನಮಃ ॥
    """
    
    print("Testing spell check...")
    print(f"Original text:\n{test_text}")
    print("-" * 50)
    
    corrected, was_corrected, status = spell_check_page_text(test_text)
    
    print(f"Status: {status}")
    print(f"Was corrected: {was_corrected}")
    print(f"Corrected text:\n{corrected}")


if __name__ == "__main__":
    test_spell_check()
