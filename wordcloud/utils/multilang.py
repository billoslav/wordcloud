"""
Multi-language support utilities for wordcloud generation.

This module provides language detection, language-specific tokenization,
and support for RTL languages and CJK characters.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Optional, Dict, List, Set, Tuple

from .logging_config import get_logger

logger = get_logger(__name__)

# Try to import optional dependencies
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.debug("langdetect not available, language detection disabled")

try:
    import stopwords
    STOPWORDS_AVAILABLE = True
except ImportError:
    STOPWORDS_AVAILABLE = False
    logger.debug("stopwords package not available")

# CJK tokenization libraries
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.debug("jieba not available, Chinese tokenization disabled")

try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False
    logger.debug("mecab-python3 not available, Japanese tokenization disabled")

try:
    from konlpy.tag import Okt, Kkma
    KONLPY_AVAILABLE = True
except ImportError:
    KONLPY_AVAILABLE = False
    logger.debug("konlpy not available, Korean tokenization disabled")


# RTL language codes
RTL_LANGUAGES = {'ar', 'he', 'fa', 'ur', 'yi'}

# CJK language codes
CJK_LANGUAGES = {'zh', 'ja', 'ko'}


def detect_language(text: str, fallback: str = 'en') -> str:
    """
    Detect the language of the input text.
    
    Args:
        text: Text to analyze
        fallback: Language code to return if detection fails (default: 'en')
        
    Returns:
        ISO 639-1 language code (e.g., 'en', 'fr', 'zh', 'ar')
    """
    if not LANGDETECT_AVAILABLE:
        logger.warning("langdetect not available, using fallback language")
        return fallback
    
    if not text or len(text.strip()) < 10:
        logger.debug(f"Text too short for language detection, using fallback: {fallback}")
        return fallback
    
    try:
        lang = detect(text)
        logger.debug(f"Detected language: {lang}")
        return lang
    except LangDetectException as e:
        logger.warning(f"Language detection failed: {e}, using fallback: {fallback}")
        return fallback


def is_rtl_language(lang_code: str) -> bool:
    """
    Check if a language code represents an RTL language.
    
    Args:
        lang_code: ISO 639-1 language code
        
    Returns:
        True if the language is RTL (Right-to-Left)
    """
    return lang_code in RTL_LANGUAGES


def is_cjk_language(lang_code: str) -> bool:
    """
    Check if a language code represents a CJK language.
    
    Args:
        lang_code: ISO 639-1 language code
        
    Returns:
        True if the language is CJK (Chinese/Japanese/Korean)
    """
    return lang_code in CJK_LANGUAGES


def get_language_stopwords(lang_code: str) -> Set[str]:
    """
    Get language-specific stopwords.
    
    Args:
        lang_code: ISO 639-1 language code
        
    Returns:
        Set of stopwords for the language, empty set if not available
    """
    if not STOPWORDS_AVAILABLE:
        logger.debug("stopwords package not available")
        return set()
    
    try:
        # Map language codes to stopwords package language names
        lang_map = {
            'en': 'english',
            'es': 'spanish',
            'fr': 'french',
            'de': 'german',
            'it': 'italian',
            'pt': 'portuguese',
            'ru': 'russian',
            'ar': 'arabic',
            'zh': 'chinese',
            'ja': 'japanese',
            'ko': 'korean',
            'hi': 'hindi',
            'nl': 'dutch',
            'sv': 'swedish',
            'tr': 'turkish',
        }
        
        lang_name = lang_map.get(lang_code)
        if lang_name:
            sw = stopwords.get_stopwords(lang_name)
            logger.debug(f"Loaded {len(sw)} stopwords for language: {lang_code}")
            return set(sw)
        else:
            logger.debug(f"No stopwords available for language: {lang_code}")
            return set()
    except Exception as e:
        logger.warning(f"Failed to load stopwords for {lang_code}: {e}")
        return set()


def tokenize_cjk(text: str, lang_code: str) -> List[str]:
    """
    Tokenize CJK (Chinese/Japanese/Korean) text.
    
    Args:
        text: Text to tokenize
        lang_code: Language code ('zh', 'ja', or 'ko')
        
    Returns:
        List of tokens
    """
    if lang_code == 'zh' and JIEBA_AVAILABLE:
        tokens = jieba.cut(text, cut_all=False)
        return [t.strip() for t in tokens if t.strip()]
    elif lang_code == 'ja' and MECAB_AVAILABLE:
        try:
            mecab = MeCab.Tagger("-Owakati")
            result = mecab.parse(text)
            return [t.strip() for t in result.split() if t.strip()]
        except Exception as e:
            logger.warning(f"MeCab tokenization failed: {e}, falling back to character-based")
            return list(text)
    elif lang_code == 'ko' and KONLPY_AVAILABLE:
        try:
            okt = Okt()
            tokens = okt.morphs(text)
            return [t.strip() for t in tokens if t.strip()]
        except Exception as e:
            logger.warning(f"KoNLPy tokenization failed: {e}, falling back to character-based")
            return list(text)
    else:
        # Fallback: character-based tokenization for CJK
        logger.debug(f"CJK tokenization library not available for {lang_code}, using character-based")
        return [c for c in text if c.strip()]


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text (NFD to NFC conversion).
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    return unicodedata.normalize('NFC', text)


def contains_cjk_characters(text: str) -> bool:
    """
    Check if text contains CJK characters.
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains CJK characters
    """
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # Chinese
            return True
        if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff':  # Japanese
            return True
        if '\uac00' <= char <= '\ud7a3':  # Korean
            return True
    return False


def contains_rtl_characters(text: str) -> bool:
    """
    Check if text contains RTL characters.
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains RTL characters
    """
    for char in text:
        # Arabic, Hebrew, and other RTL scripts
        if '\u0590' <= char <= '\u05ff':  # Hebrew
            return True
        if '\u0600' <= char <= '\u06ff':  # Arabic
            return True
        if '\u0700' <= char <= '\u074f':  # Syriac
            return True
        if '\u0750' <= char <= '\u077f':  # Arabic Supplement
            return True
        if '\u08a0' <= char <= '\u08ff':  # Arabic Extended-A
            return True
        if '\ufb50' <= char <= '\ufdff':  # Arabic Presentation Forms-A
            return True
        if '\ufe70' <= char <= '\ufeff':  # Arabic Presentation Forms-B
            return True
    return False


def get_text_direction(text: str, lang_code: Optional[str] = None) -> str:
    """
    Determine text direction (LTR or RTL).
    
    Args:
        text: Text to analyze
        lang_code: Optional language code (if None, will detect from text)
        
    Returns:
        'rtl' or 'ltr'
    """
    if lang_code and is_rtl_language(lang_code):
        return 'rtl'
    
    if contains_rtl_characters(text):
        return 'rtl'
    
    return 'ltr'

