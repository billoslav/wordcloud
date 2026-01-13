from operator import itemgetter
from typing import Optional, Dict, List, Tuple, Union, Callable
from collections import Counter
import re

from .logging_config import get_logger

logger = get_logger(__name__)

# Try to import optional NLP libraries
try:
    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.corpus import stopwords as nltk_stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.debug("NLTK not available, stemming/lemmatization disabled")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.debug("spaCy not available, advanced NLP disabled")

class TextProcessor:
    def __init__(
        self,
        min_word_length: int = 3,
        max_words: int = 200,
        stopwords = [],
        language: Optional[str] = None,
        enable_stemming: bool = False,
        enable_lemmatization: bool = False,
        n_gram_range: Optional[Tuple[int, int]] = None,
        text_processor_options: Optional[Dict] = None,
    ):
        """
        Initialize the text processor.
        
        Args:
            min_word_length: Minimum length for words to be included
            max_words: Maximum number of words to return
            stopwords: List of words to exclude
            language: Language code (e.g., 'en', 'fr', 'zh'). If None, will auto-detect.
            enable_stemming: Whether to apply stemming (requires NLTK)
            enable_lemmatization: Whether to apply lemmatization (requires NLTK or spaCy)
            n_gram_range: Tuple (min_n, max_n) for n-gram extraction (e.g., (1, 2) for unigrams+bigrams)
            text_processor_options: Additional options dict for advanced configuration
        """
        self.min_word_length = min_word_length
        self.max_words = max_words
        self.stopwords = set(stopwords or [])
        self.language = language
        self.enable_stemming = enable_stemming
        self.enable_lemmatization = enable_lemmatization
        self.n_gram_range = n_gram_range
        self.text_processor_options = text_processor_options or {}
        
        # Initialize NLP tools if available
        self._stemmer = None
        self._lemmatizer = None
        self._spacy_nlp = None
        
        if enable_stemming and NLTK_AVAILABLE:
            try:
                self._stemmer = PorterStemmer()
                logger.debug("PorterStemmer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize stemmer: {e}")
        
        if enable_lemmatization:
            if SPACY_AVAILABLE:
                try:
                    lang_code = language or 'en'
                    spacy_model = self.text_processor_options.get('spacy_model', f'{lang_code}_core_web_sm')
                    self._spacy_nlp = spacy.load(spacy_model)
                    logger.debug(f"spaCy model loaded: {spacy_model}")
                except Exception as e:
                    logger.warning(f"Failed to load spaCy model: {e}, trying NLTK")
                    if NLTK_AVAILABLE:
                        try:
                            self._lemmatizer = WordNetLemmatizer()
                            logger.debug("WordNetLemmatizer initialized")
                        except Exception as e2:
                            logger.warning(f"Failed to initialize NLTK lemmatizer: {e2}")
            elif NLTK_AVAILABLE:
                try:
                    self._lemmatizer = WordNetLemmatizer()
                    logger.debug("WordNetLemmatizer initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize lemmatizer: {e}")
        
        logger.info(f"TextProcessor initialized: min_word_length={min_word_length}, max_words={max_words}, "
                   f"language={language}, stemming={enable_stemming}, lemmatization={enable_lemmatization}, "
                   f"n_gram_range={n_gram_range}")
        
    def _stem_word(self, word: str) -> str:
        """Apply stemming to a word."""
        if self._stemmer:
            return self._stemmer.stem(word)
        return word
    
    def _lemmatize_word(self, word: str, pos: str = 'n') -> str:
        """Apply lemmatization to a word."""
        if self._spacy_nlp:
            doc = self._spacy_nlp(word)
            if doc:
                return doc[0].lemma_
        elif self._lemmatizer:
            return self._lemmatizer.lemmatize(word, pos=pos)
        return word
    
    def _extract_ngrams(self, tokens: List[str], min_n: int, max_n: int) -> List[str]:
        """Extract n-grams from tokens."""
        ngrams = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)
        return ngrams
    
    def split_text(self, text_to_split, stopwords=None, min_word_length=None):
        """
        Split and preprocess text, counting word frequencies.
        
        This method takes a string of text, splits it into words, and counts
        the frequency of each word. It also handles case normalization,
        stopword removal, filtering by minimum word length, stemming/lemmatization,
        and n-gram extraction.
        
        Args:
            text_to_split (str): Text to analyze
            stopwords (list, optional): List of words to exclude. If None, use the instance's stopwords.
            min_word_length (int, optional): Minimum word length to include. If None, use the instance's value.
            
        Returns:
            dict: Dictionary of {word: frequency} pairs
            
        Example:
            >>> tp = TextProcessor(stopwords=["and", "the"])
            >>> tp.split_text("The quick and the dead")
            {'quick': 1, 'dead': 1}
        """
        from .multilang import (
            detect_language, get_language_stopwords, tokenize_cjk,
            is_cjk_language, normalize_unicode, contains_cjk_characters
        )
        
        logger.debug(f"Splitting text (length: {len(text_to_split)} characters)")
        
        # Normalize Unicode
        text_to_split = normalize_unicode(text_to_split)
        
        # Detect language if not specified
        detected_lang = self.language
        if detected_lang is None:
            detected_lang = detect_language(text_to_split)
            logger.debug(f"Auto-detected language: {detected_lang}")
        
        # Get language-specific stopwords
        lang_stopwords = get_language_stopwords(detected_lang)
        combined_stopwords = self.stopwords.union(lang_stopwords)
        if stopwords:
            combined_stopwords = combined_stopwords.union(set(stopwords))
        
        min_len = min_word_length if min_word_length is not None else self.min_word_length
        
        # Tokenize based on language
        if is_cjk_language(detected_lang) or contains_cjk_characters(text_to_split):
            tokens = tokenize_cjk(text_to_split, detected_lang)
        else:
            # Standard tokenization for non-CJK languages
            clean_text = text_to_split.replace("-", " ")
            clean_text = re.sub(r"[^\w\s]", " ", clean_text, flags=re.UNICODE)
            tokens = clean_text.split()
        
        # Process tokens
        processed_tokens = []
        for token in tokens:
            # Filter non-alphabetic characters (for non-CJK)
            if not is_cjk_language(detected_lang) and not contains_cjk_characters(token):
                token = ''.join([i for i in token if i.isalnum()])
            
            if not token or len(token) < min_len:
                continue
            
            # Case normalization (for non-CJK)
            if not is_cjk_language(detected_lang) and not contains_cjk_characters(token):
                token = token.lower()
            
            # Stopword filtering
            if token in combined_stopwords:
                continue
            
            # Apply stemming or lemmatization
            if self.enable_stemming:
                token = self._stem_word(token)
            elif self.enable_lemmatization:
                token = self._lemmatize_word(token)
            
            if token:
                processed_tokens.append(token)
        
        # Extract n-grams if specified
        if self.n_gram_range:
            min_n, max_n = self.n_gram_range
            ngrams = self._extract_ngrams(processed_tokens, min_n, max_n)
            # Filter n-grams
            filtered_ngrams = []
            for ngram in ngrams:
                if len(ngram.replace(' ', '')) >= min_len and ngram not in combined_stopwords:
                    filtered_ngrams.append(ngram)
            processed_tokens.extend(filtered_ngrams)
        
        # Count frequencies
        res = dict(Counter(processed_tokens))
        logger.debug(f"Split text into {len(res)} unique words/phrases")
        return res 
    
    def sort_normalize(self, input_words):
        """
        Sort words by frequency and normalize frequencies.
        
        This method sorts words by their frequency (descending) and normalizes
        the frequencies relative to the most frequent word. This ensures that
        the most frequent word will have a normalized frequency of 1.0, and all
        other words will have normalized frequencies between 0.0 and 1.0.
        
        Args:
            input_words (dict): Dictionary of {word: frequency} pairs
            
        Returns:
            list: List of (word, normalized_frequency, original_frequency) tuples
            
        Raises:
            ValueError: If input_words is empty
            
        Example:
            >>> wc = Wordcloud()
            >>> wc.sort_normalize({'apple': 5, 'banana': 3, 'cherry': 1})
            [('apple', 1.0, 5), ('banana', 0.6, 3), ('cherry', 0.2, 1)]
        """
        if not input_words:  # Check for empty input
            logger.error("No words to process in sort_normalize")
            raise ValueError("No words to process.")

        logger.debug(f"Sorting and normalizing {len(input_words)} words")
        frequencies = sorted(input_words.items(), key=itemgetter(1), reverse=True)
        max_frequency = float(frequencies[0][1])
        result = [(word, freq / max_frequency, freq) for word, freq in frequencies]
        if result:
            logger.debug(f"Most frequent word: '{result[0][0]}' with frequency {result[0][2]}")
        return result
    
    def prepare_text(self, to_split):
        """
        Prepare text for wordcloud generation.
        
        This is a convenience method that combines text splitting, frequency
        counting, and normalization in a single call.
        
        Args:
            to_split (str): Text to analyze
            stopwords (list, optional): List of words to exclude
            min_word_length (int, optional): Minimum word length to include
            
        Returns:
            list: List of (word, normalized_frequency, original_frequency) tuples
            
        Example:
            >>> wc = Wordcloud()
            >>> wc.prepare_text("apple apple banana banana banana cherry")
            [('banana', 1.0, 3), ('apple', 0.6666666666666666, 2), ('cherry', 0.3333333333333333, 1)]
        """
        logger.info(f"Preparing text (length: {len(to_split)} characters)")
        splitted = self.split_text(to_split)
        result = self.sort_normalize(splitted)
        logger.info(f"Prepared {len(result)} words for wordcloud")
        return result