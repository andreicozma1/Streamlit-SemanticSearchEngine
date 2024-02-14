from enum import Enum
from string import punctuation

import spacy
import streamlit as st
from spacy.lang.en import stop_words
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.cli import download

from src.utils import pprint


def setup_spacy(name="en_core_web_md"):
    if not spacy.util.is_package(name):
        download(name)

    return spacy.load(name)


nlp = setup_spacy()


class Highlighter(Enum):
    YAKE = "yake"
    RAKE_NLTK = "rake_nltk"
    MULTI_RAKE = "multi_rake"
    TEXTRANK = "textrank"

    def __str__(self):
        return self.value


@st.cache_data
def get_highlights(text: str, method: Highlighter, **kwargs):
    highlights = []
    match method:
        case Highlighter.YAKE:
            import yake

            # - `lan`: The language of the text from which keywords will be extracted. It defaults to "en" (English). The language is used to determine the appropriate stopword list.
            # - `n`: The size of the sliding window for generating candidate keyphrases. It defaults to 3.
            # - `dedupLim`: The deduplication limit. If the similarity between two candidate keyphrases is greater than this limit, the second keyphrase is considered a duplicate and is not added to the result set. It defaults to 0.9.
            # - `dedupFunc`: The function used to calculate the similarity between two candidate keyphrases for deduplication. It can be 'jaro_winkler', 'jaro', 'sequencematcher', 'seqm', or any other function that takes two strings and returns a similarity score. It defaults to 'seqm'.
            # - `windowsSize`: The size of the window for the `DataCore` class, which is used to generate candidate keyphrases. It defaults to 1.
            # - `top`: The number of top-scoring keyphrases to return. It defaults to 20.

            # lan="en", n=3, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=20,
            kwargs.setdefault("lan", "en")

            pprint(kwargs)
            kwargs.setdefault("stopwords", set(stop_words.STOP_WORDS))

            custom_kw_extractor = yake.KeywordExtractor(**kwargs)
            result = custom_kw_extractor.extract_keywords(text)
            print(result)

            if result:
                highlights, scores = zip(*result)

        case Highlighter.RAKE_NLTK:
            from rake_nltk import Rake

            # language: str = 'english',
            # ranking_metric: Metric = Metric.DEGREE_TO_FREQUENCY_RATIO,
            # DEGREE_TO_FREQUENCY_RATIO = 0  # Uses d(w)/f(w) as the metric
            # WORD_DEGREE = 1  # Uses d(w) alone as the metric
            # WORD_FREQUENCY = 2  # Uses f(w) alone as the metric
            # max_length: int = 100000,
            # min_length: int = 1,
            kwargs.setdefault("language", "english")
            kwargs.setdefault("punctuations", set(list(punctuation)))
            kwargs.setdefault("min_length", 2)
            kwargs.setdefault("include_repeated_phrases", False)

            pprint(kwargs)
            kwargs.setdefault("stopwords", set(stop_words.STOP_WORDS))

            r = Rake(**kwargs)
            r.extract_keywords_from_text(text)
            result = r.get_ranked_phrases_with_scores()
            print(result)

            if result:
                scores, highlights = zip(*result)

        case Highlighter.MULTI_RAKE:
            from multi_rake import Rake

            # min_chars: The minimum number of characters a word must have to be considered as a keyword. Defaults to 3.
            # max_words: The maximum number of words an extracted keyword can have. Defaults to 3.
            # min_freq: The minimum frequency a candidate keyword needs to have to be considered as a keyword. Defaults to 1.
            # language_code: The language code for the language of the text. If not provided, the language will be detected automatically.
            # stopwords: A set of stopwords. If not provided, the stopwords for the detected or provided language will be used.
            # lang_detect_threshold: The minimum number of characters in the text for language detection to be applied. If the text has fewer characters, the language is considered unknown. Defaults to 50.
            # max_words_unknown_lang: The maximum number of words an extracted keyword can have if the language is unknown. Defaults to 2.
            # generated_stopwords_percentile: The percentile of word frequencies to use as the upper bound when generating stopwords from the text. Defaults to 80.
            # generated_stopwords_max_len: The maximum length of a word for it to be considered as a stopword when generating stopwords from the text. Defaults to 3.
            # generated_stopwords_min_freq: The minimum frequency a word needs to have to be considered as a stopword when generating stopwords from the text. Defaults to 2.
            kwargs.setdefault("language_code", "en")
            kwargs.setdefault("min_freq", 2)

            pprint(kwargs)
            kwargs.setdefault("stopwords", set(stop_words.STOP_WORDS))

            r = Rake(**kwargs)
            result = r.apply(text)
            print(result)

            if result:
                highlights, scores = zip(*result)

        case Highlighter.TEXTRANK:
            from summa import keywords as summa_keywords

            # ratio=0.2, words=None, language="english", split=False, scores=False, deaccent=False
            kwargs.setdefault("language", "english")
            kwargs.setdefault("split", False)
            kwargs.setdefault("min_length", 3)

            if kwargs.get("ratio", None) == 0:
                kwargs.pop("ratio")
            if kwargs.get("words", None) == 0:
                kwargs.pop("words")

            pprint(kwargs)

            min_length = kwargs.pop("min_length")

            result = summa_keywords.keywords(text, scores=True, **kwargs)
            print(result)

            result = list(filter(lambda x: len(x[0]) >= min_length, result))

            if result:
                highlights, scores = zip(*result)

        case _:
            raise ValueError(f"Invalid method: {method}")

    highlights = list(set(highlights))
    print("keywords:", highlights)

    return highlights


def merge_spans(spans):
    """
    Merge overlapping or adjacent spans to ensure each part of the text is processed once.
    """
    if not spans:
        return []

    # Sort spans by their start position
    sorted_spans = sorted(spans, key=lambda span: span.start)

    # Initialize merged spans list with the first span
    merged = [sorted_spans[0]]

    for current_span in sorted_spans[1:]:
        prev_span = merged[-1]

        # If current span overlaps or is adjacent to the previous span, merge them
        if current_span.start <= prev_span.end:
            merged[-1] = Span(
                prev_span.doc,
                prev_span.start,
                max(prev_span.end, current_span.end),
                label=prev_span.label,
            )
        else:
            merged.append(current_span)

    return merged


@st.cache_data
def apply_highlight(text: str, highlights: list[str] | str, color: str):
    assert color in [
        "blue",
        "green",
        "orange",
        "red",
        "violet",
        "gray",
        "grey",
        "rainbow",
    ]

    # Function to preprocess text and highlights for matching
    def preprocess(doc):
        return " ".join([token.lemma_.lower() for token in doc])

    text_doc = nlp(text)
    text_doc_lemmatized = nlp(preprocess(text_doc))

    if not isinstance(highlights, list):
        highlights = [highlights]

    # sort highlights by length to match longer phrases first
    # highlights = sorted(highlights, key=len, reverse=True)

    highlight_docs = list(nlp.pipe(highlights))
    highlight_docs_lemmatized = list(
        nlp.pipe([preprocess(doc) for doc in highlight_docs])
    )

    # Matcher for lowercase matching
    matcher_lower = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher_lower.add("QUERY_LOWER", highlight_docs)

    # Matcher for lemma matching
    matcher_lemma = PhraseMatcher(nlp.vocab, attr="LEMMA")
    matcher_lemma.add("QUERY_LEMMA", highlight_docs_lemmatized)

    # Match using both matchers
    matches_lower = matcher_lower(text_doc)
    matches_lemma = matcher_lemma(text_doc_lemmatized)

    # Combine matches
    matches_combined = list(set(matches_lower + matches_lemma))

    matched_spans = [text_doc[start:end] for match_id, start, end in matches_combined]

    # Deduplicate and sort spans (assume merge_spans is defined)
    unique_spans = sorted(set(matched_spans), key=lambda span: span.start_char)

    unique_spans = merge_spans(unique_spans)

    highlighted_text = ""
    last_idx = 0
    for span in unique_spans:
        start, end = span.start_char, span.end_char
        highlighted_text += text[last_idx:start]
        highlighted_text += f":{color}[**{text[start:end]}**]"
        last_idx = end
    highlighted_text += text[last_idx:]

    num_matches = len(unique_spans)

    return highlighted_text, num_matches
