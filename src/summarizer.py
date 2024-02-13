from enum import Enum
from typing import Optional

import streamlit as st
from spacy.lang.en import stop_words

from src.utils import pprint


class Summarizer(Enum):
    LSA = "lsa"
    LUHN = "luhn"
    EDMUNDSON = "edmundson"
    TEXT_RANK = "text-rank"
    LEX_RANK = "lex-rank"
    KL_SUM = "kl-sum"
    RANDOM = "random"
    REDUCTION = "reduction"

    def __str__(self):
        return self.value


@st.cache_data
def get_summary(text: str, method: Summarizer, **kwargs):
    match method:
        case "summa":
            import summa

            # ratio=0.2, words=None, language="english", split=False,
            kwargs.setdefault("language", "english")
            kwargs.setdefault("split", False)
            if kwargs.get("ratio", None) == 0:
                kwargs.pop("ratio")
            if kwargs.get("words", None) == 0:
                kwargs.pop("words")

            pprint(kwargs)

            summary = summa.summarizer.summarize(text, **kwargs)
        case "sumy":
            from sumy.nlp.stemmers import Stemmer
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.parsers.plaintext import PlaintextParser

            kwargs.setdefault("language", "english")
            kwargs.setdefault("sentence_count", 6)

            pprint(kwargs)

            algorithm = kwargs.pop("algorithm")

            match algorithm:
                case Summarizer.LSA:
                    from sumy.summarizers.lsa import LsaSummarizer as Sum
                case Summarizer.LUHN:
                    from sumy.summarizers.luhn import LuhnSummarizer as Sum
                case Summarizer.EDMUNDSON:
                    from sumy.summarizers.edmundson import EdmundsonSummarizer as Sum
                case Summarizer.TEXT_RANK:
                    from sumy.summarizers.text_rank import TextRankSummarizer as Sum
                case Summarizer.LEX_RANK:
                    from sumy.summarizers.lex_rank import LexRankSummarizer as Sum
                case Summarizer.KL_SUM:
                    from sumy.summarizers.kl import KLSummarizer as Sum
                case Summarizer.RANDOM:
                    from sumy.summarizers.random import RandomSummarizer as Sum
                case Summarizer.REDUCTION:
                    from sumy.summarizers.reduction import ReductionSummarizer as Sum
                case _:
                    raise ValueError(f"Invalid algorithm: {kwargs['algorithm']}")

            language = kwargs.pop("language")
            sentence_count = kwargs.pop("sentence_count")

            parser = PlaintextParser.from_string(text, Tokenizer(language))

            stemmer = Stemmer(language)
            summarizer = Sum(stemmer)

            if algorithm == Summarizer.EDMUNDSON:
                summarizer.null_words = stop_words.STOP_WORDS
                summarizer.bonus_words = parser.significant_words
                summarizer.stigma_words = parser.stigma_words

                print("null_words:", summarizer.null_words)
                print("bonus_words:", summarizer.bonus_words)
                print("stigma_words:", summarizer.stigma_words)
            else:
                summarizer.stop_words = stop_words.STOP_WORDS

            summary_sentences = summarizer(parser.document, sentence_count)
            summary = " ".join([str(sentence) for sentence in summary_sentences])

        case "transformers":
            raise NotImplementedError("Transformers summarization not implemented")
        case "ChatGPT":
            raise NotImplementedError("ChatGPT summarization not implemented")
        case _:
            raise ValueError(f"Invalid summary method: {text}")

    assert isinstance(summary, str), "Summary must be a string"

    return summary.strip()
