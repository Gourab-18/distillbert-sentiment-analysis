"""
Day 22: NLP Basics - Text Preprocessing
Topics: Tokenization, Stemming, Lemmatization
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from collections import Counter
import pandas as pd
import numpy as np

# Download required NLTK data
def download_nltk_resources():
    """Download necessary NLTK resources"""
    resources = ['punkt', 'averaged_perceptron_tagger', 'wordnet',
                 'stopwords', 'omw-1.4', 'punkt_tab', 'averaged_perceptron_tagger_eng']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

download_nltk_resources()


class TextPreprocessor:
    """
    Comprehensive text preprocessing class for NLP tasks
    """

    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))

        # Initialize stemmers
        self.porter_stemmer = PorterStemmer()
        self.lancaster_stemmer = LancasterStemmer()
        self.snowball_stemmer = SnowballStemmer(language)

        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # Regex tokenizer (removes punctuation)
        self.regex_tokenizer = RegexpTokenizer(r'\w+')

    def basic_clean(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def remove_punctuation(self, text):
        """Remove punctuation from text"""
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_numbers(self, text):
        """Remove numbers from text"""
        return re.sub(r'\d+', '', text)

    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        return [token for token in tokens if token.lower() not in self.stop_words]

    # ==================== TOKENIZATION ====================

    def word_tokenize(self, text):
        """
        Word tokenization using NLTK
        Splits text into individual words
        """
        return word_tokenize(text)

    def sentence_tokenize(self, text):
        """
        Sentence tokenization
        Splits text into sentences
        """
        return sent_tokenize(text)

    def regex_tokenize(self, text):
        """
        Regex-based tokenization (removes punctuation automatically)
        """
        return self.regex_tokenizer.tokenize(text)

    def custom_tokenize(self, text, pattern=r'\w+'):
        """
        Custom tokenization with user-defined pattern
        """
        return re.findall(pattern, text)

    # ==================== STEMMING ====================

    def porter_stem(self, tokens):
        """
        Porter Stemmer - Most common, gentle stemming
        Example: 'running' -> 'run', 'cats' -> 'cat'
        """
        return [self.porter_stemmer.stem(token) for token in tokens]

    def lancaster_stem(self, tokens):
        """
        Lancaster Stemmer - More aggressive stemming
        Example: 'maximum' -> 'maxim'
        """
        return [self.lancaster_stemmer.stem(token) for token in tokens]

    def snowball_stem(self, tokens):
        """
        Snowball Stemmer (Porter2) - Improved Porter
        Supports multiple languages
        """
        return [self.snowball_stemmer.stem(token) for token in tokens]

    # ==================== LEMMATIZATION ====================

    def get_wordnet_pos(self, treebank_tag):
        """Convert Penn Treebank POS tags to WordNet POS tags"""
        from nltk.corpus import wordnet

        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun

    def lemmatize(self, tokens, use_pos=True):
        """
        Lemmatization - Reduces words to their dictionary form
        Example: 'better' -> 'good', 'running' -> 'run'

        Args:
            tokens: List of tokens
            use_pos: Whether to use POS tagging for better results
        """
        if use_pos:
            pos_tags = pos_tag(tokens)
            return [
                self.lemmatizer.lemmatize(token, self.get_wordnet_pos(pos))
                for token, pos in pos_tags
            ]
        else:
            return [self.lemmatizer.lemmatize(token) for token in tokens]

    # ==================== FULL PIPELINE ====================

    def preprocess(self, text,
                   lowercase=True,
                   remove_punct=True,
                   remove_nums=False,
                   remove_stops=True,
                   stem=False,
                   lemmatize=True,
                   stemmer='porter'):
        """
        Complete preprocessing pipeline

        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_punct: Remove punctuation
            remove_nums: Remove numbers
            remove_stops: Remove stopwords
            stem: Apply stemming
            lemmatize: Apply lemmatization
            stemmer: Which stemmer to use ('porter', 'lancaster', 'snowball')

        Returns:
            List of processed tokens
        """
        # Basic cleaning
        text = self.basic_clean(text)

        if lowercase:
            text = text.lower()

        if remove_nums:
            text = self.remove_numbers(text)

        # Tokenize
        if remove_punct:
            tokens = self.regex_tokenize(text)
        else:
            tokens = self.word_tokenize(text)

        # Remove stopwords
        if remove_stops:
            tokens = self.remove_stopwords(tokens)

        # Stemming or Lemmatization (not both typically)
        if stem and not lemmatize:
            if stemmer == 'porter':
                tokens = self.porter_stem(tokens)
            elif stemmer == 'lancaster':
                tokens = self.lancaster_stem(tokens)
            else:
                tokens = self.snowball_stem(tokens)
        elif lemmatize:
            tokens = self.lemmatize(tokens)

        return tokens

    def preprocess_batch(self, texts, **kwargs):
        """Preprocess a batch of texts"""
        return [self.preprocess(text, **kwargs) for text in texts]


def compare_stemming_methods(word_list):
    """
    Compare different stemming methods
    """
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()

    results = []
    for word in word_list:
        results.append({
            'Original': word,
            'Porter': porter.stem(word),
            'Lancaster': lancaster.stem(word),
            'Snowball': snowball.stem(word),
            'Lemmatized': lemmatizer.lemmatize(word)
        })

    return pd.DataFrame(results)


def demonstrate_preprocessing():
    """
    Demonstrate all preprocessing techniques
    """
    # Sample text
    sample_text = """
    Natural Language Processing (NLP) is a subfield of linguistics, computer science,
    and artificial intelligence. NLP is concerned with the interactions between computers
    and human language. The researchers are studying various algorithms for processing
    textual data. They have been running experiments since 2020.
    Visit https://example.com for more information! Contact: nlp@research.org
    """

    print("=" * 60)
    print("TEXT PREPROCESSING DEMONSTRATION")
    print("=" * 60)

    preprocessor = TextPreprocessor()

    # Original text
    print("\n1. ORIGINAL TEXT:")
    print("-" * 40)
    print(sample_text)

    # Basic cleaning
    print("\n2. AFTER BASIC CLEANING:")
    print("-" * 40)
    cleaned = preprocessor.basic_clean(sample_text)
    print(cleaned)

    # Word tokenization
    print("\n3. WORD TOKENIZATION:")
    print("-" * 40)
    tokens = preprocessor.word_tokenize(cleaned)
    print(f"Tokens ({len(tokens)}): {tokens}")

    # Sentence tokenization
    print("\n4. SENTENCE TOKENIZATION:")
    print("-" * 40)
    sentences = preprocessor.sentence_tokenize(cleaned)
    for i, sent in enumerate(sentences, 1):
        print(f"  Sentence {i}: {sent}")

    # Remove stopwords
    print("\n5. AFTER REMOVING STOPWORDS:")
    print("-" * 40)
    tokens_no_stop = preprocessor.remove_stopwords(tokens)
    print(f"Tokens ({len(tokens_no_stop)}): {tokens_no_stop}")

    # Compare stemming methods
    print("\n6. STEMMING COMPARISON:")
    print("-" * 40)
    test_words = ['running', 'runs', 'ran', 'studies', 'studying',
                  'better', 'processing', 'computers', 'algorithms']
    comparison = compare_stemming_methods(test_words)
    print(comparison.to_string(index=False))

    # Full pipeline
    print("\n7. FULL PREPROCESSING PIPELINE (with lemmatization):")
    print("-" * 40)
    processed = preprocessor.preprocess(sample_text)
    print(f"Processed tokens ({len(processed)}): {processed}")

    # Full pipeline with stemming
    print("\n8. FULL PREPROCESSING PIPELINE (with stemming):")
    print("-" * 40)
    processed_stem = preprocessor.preprocess(sample_text, stem=True, lemmatize=False)
    print(f"Processed tokens ({len(processed_stem)}): {processed_stem}")

    return preprocessor


if __name__ == "__main__":
    preprocessor = demonstrate_preprocessing()

    # Additional examples
    print("\n" + "=" * 60)
    print("ADDITIONAL EXAMPLES")
    print("=" * 60)

    # POS tagging example
    text = "The quick brown fox jumps over the lazy dog"
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    print(f"\nPOS Tagging: {pos_tags}")

    # Batch processing
    texts = [
        "Machine learning is transforming industries",
        "Deep learning models require large datasets",
        "NLP enables computers to understand human language"
    ]

    print("\nBatch Processing:")
    processed_batch = preprocessor.preprocess_batch(texts)
    for original, processed in zip(texts, processed_batch):
        print(f"  Original: {original}")
        print(f"  Processed: {processed}")
        print()
