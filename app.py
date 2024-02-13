import json
import os
from enum import Enum

import matplotlib.pyplot as plt
import pyperclip
import streamlit as st
from langchain_community.vectorstores import Chroma
from wordcloud import WordCloud

from src import extractors, highlighter, resources, summarizer, utils

st.set_page_config(layout="wide")


@st.cache_resource
def setup_db(docs_path="./docs/", cache_path="./cache/"):
    cached_embedder, local_store = resources.setup_embedder(cache_path)
    docs = resources.load_docs(docs_path)
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=cached_embedder,
    )

    return vector_db


db = setup_db()


def run_search(query, search_type, **kwargs):
    print("=" * 80)
    print(locals())
    retriever = db.as_retriever(
        search_type=search_type,
        search_kwargs=kwargs,
    )
    results = retriever.get_relevant_documents(query=query)
    print(f"Found {len(results)} results")
    return results


def st_copy_to_clipboard_btn(text, **kwargs):
    if st.button("Copy", **kwargs):
        # NOTE: will only be able to copy on the host machine
        pyperclip.copy(text)
        # st.info("Copied to Clipboard")
        st.toast("Copied to Clipboard", icon="ℹ️")


@st.cache_data
def generate_wordcloud(text):
    wordcloud = WordCloud(
        background_color="white",
        width=1280,
        height=720,
        min_word_length=3,
    ).generate(text)
    return wordcloud


def display_wordcloud(text):
    # https://pypi.org/project/wordcloud/
    wordcloud = generate_wordcloud(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=False)


class SearchResult:
    def __init__(self, doc):
        self.text: str = doc.page_content
        self.text_clean: str = utils.clean_text(self.text)

        self.metadata: dict = doc.metadata

        self.code: list = extractors.extract_code(self.text)
        self.math: list = extractors.extract_math(self.text)


def display_math(math_list: list[str], key: str):
    # with st.container(border=True):
    with st.expander(f"Math ({len(math_list)})", expanded=False):
        # st.subheader("Math Blocks")
        # dropdowns to sort by various options and then another dropdown for ascending or descending
        _, col_sort, col_order, _ = st.columns([2, 2, 2, 2])

        with col_sort:
            sort_by = st.selectbox(
                "Sort By",
                [None, "Length"],
                index=0,
                key=f"math_{key}_sort_by",
            )
        with col_order:
            order = st.selectbox(
                "Order",
                ["Ascending", "Descending"],
                index=1,
                key=f"math_{key}_sort_order",
            )
        reverse = order == "Descending"

        match sort_by:
            case None:
                pass
            case "Length":
                math_list = sorted(math_list, key=lambda x: len(x), reverse=reverse)
            case _:
                raise ValueError(f"Invalid sort_by: {sort_by}")

        for i, math in enumerate(math_list):
            col1, col2, col3 = st.columns([1, 10, 1])
            with col1:
                st.write(f"{i+1}.")
            with col2:
                st.latex(math)
            with col3:
                st_copy_to_clipboard_btn(math, key=f"math_{key}+{i}_copy")
        else:
            st.write("No math blocks found.")


def display_code(code_list: list[tuple[str, str]], key: str):
    # with st.container(border=True):
    with st.expander(f"Code ({len(code_list)})", expanded=False):
        # dropdowns to sort by various options and then another dropdown for ascending or descending
        _, col_sort, col_order, _ = st.columns([2, 2, 2, 2])
        with col_sort:
            sort_by = st.selectbox(
                "Sort By",
                [None, "Language", "Length"],
                index=0,
                key=f"code_{key}_sort_by",
            )
        with col_order:
            order = st.selectbox(
                "Order",
                ["Ascending", "Descending"],
                index=1,
                key=f"code_{key}_sort_order",
            )
        reverse = order == "Descending"

        match sort_by:
            case None:
                pass
            case "Language":
                code_list = sorted(
                    code_list, key=lambda x: x[0].lower(), reverse=reverse
                )
            case "Length":
                code_list = sorted(code_list, key=lambda x: len(x[1]), reverse=reverse)
            case _:
                raise ValueError(f"Invalid sort_by: {sort_by}")

        # st.subheader("Code Blocks")
        for i, (lang, code) in enumerate(code_list):
            col1, col2, col3 = st.columns([1, 10, 1])
            with col1:
                st.write(f"{i+1}.")
            with col2:
                st.code(code, language=lang, line_numbers=True)
            with col3:
                st_copy_to_clipboard_btn(code, key=f"code_{key}+{i}_copy")
        else:
            st.write("No code blocks found.")


def display_search_results(
    search_key: tuple,
    show_summary=True,
    summary_kwargs={},
    show_wordcloud=True,
    show_global_highlights=True,
    show_local_highlights=True,
    show_math_blocks=True,
    show_code_blocks=True,
    show_cleaned_text=True,
    show_raw_text=True,
    show_metadata=True,
    highlights_kwargs={},
):
    query, search_kwargs_hash = search_key
    print("=" * 80)
    print(locals())

    results = st.session_state.search_results.get(search_key, None)

    if not results:
        st.error("No results found.")
        return

    print(f"Displaying {len(results)} results")
    # print(results)

    # remove duplicates if page_content is the same
    len_before = len(results)
    results = list({doc.page_content: doc for doc in results}.values())
    len_after = len(results)
    len_removed = len_before - len_after

    processed_results = [SearchResult(result) for result in results]

    all_text = "\n\n".join([doc.text for doc in processed_results]).strip()
    all_text_clean = utils.clean_text(all_text)

    all_math = list(set([math for result in processed_results for math in result.math]))
    all_code = list(set([code for result in processed_results for code in result.code]))

    all_highlights = highlighter.get_highlights(all_text_clean, **highlights_kwargs)
    all_highlights = sorted(all_highlights, key=lambda kw: kw.lower())

    # with st.container(border=True):
    with st.expander(
        "Global Highlights",
        expanded=True,
    ):
        col1, col2 = st.columns(2)

        # Deselecting checkboxes here means that the phrase will not be highlighted in results
        # The choice will be remembered globally across search results
        all_text_kw_sel = {
            kw: st.session_state.global_highlights.get(kw, True)
            for kw in all_highlights
        }
        with col1:
            for j, (kw, sel) in enumerate(list(all_text_kw_sel.items())[::2]):
                all_text_kw_sel[kw] = st.checkbox(kw, value=sel, key=f"kw_{j}_{kw}")
        with col2:
            for j, (kw, sel) in enumerate(list(all_text_kw_sel.items())[1::2]):
                all_text_kw_sel[kw] = st.checkbox(kw, value=sel, key=f"kw_{j}_{kw}")

        # We want the local selections to persist globally across search results
        # This way the user may decide what is worthy of being highlighted or not with each search result
        # and deselected results (those that won't be highlighted anymore) will be remembered
        st.session_state.global_highlights.update(all_text_kw_sel)
        all_highlights = [kw for kw, sel in all_text_kw_sel.items() if sel]

    with st.container(border=True):
        st.subheader(
            f"{len(results)} Results"
            + (f" ({len_removed} duplicates removed)" if len_removed else "")
        )

        if show_summary:
            # with st.container(border=True):
            with st.expander("Summary", expanded=True):
                # st.subheader("Summary")
                summary = summarizer.get_summary(
                    all_text,
                    **summary_kwargs,
                )
                summary_clean = utils.clean_text(summary)

                # summary, num_matches = highlighter.apply_highlight(
                #     summary,
                #     query,
                #     color="red",
                # )

                if show_global_highlights:
                    summary, num_matches = highlighter.apply_highlight(
                        summary,
                        all_highlights,
                        color="violet",
                    )

                if show_local_highlights:
                    highlights_local = highlighter.get_highlights(
                        summary_clean, **highlights_kwargs
                    )
                    summary, num_matches = highlighter.apply_highlight(
                        summary,
                        highlights_local,
                        color="blue",
                    )

                st.write(summary)
                st_copy_to_clipboard_btn(summary, key="copy_all")

                if st.button("Find Similar", key="find_similar_sum"):
                    st.query_params["q"] = summary_clean
                    st.rerun()
                    return

        if show_math_blocks and all_math:
            display_math(all_math, key="all")

        if show_code_blocks and all_code:
            display_code(all_code, key="all")

        if show_wordcloud:
            # with st.container(border=True):
            with st.expander("Word Cloud", expanded=True):
                # st.subheader("Word Cloud")
                display_wordcloud(all_text_clean)

    for i, result in enumerate(processed_results):
        title = result.metadata.get("title", None)
        source = result.metadata.get("source", None)

        if title is None:
            title, _ = os.path.splitext(os.path.basename(source))

        title = f"{i+1}. {title}"

        text = result.text
        text_clean = result.text_clean

        # text, num_matches = highlighter.apply_highlight(
        #     text,
        #     query,
        #     color="red",
        # )

        if show_global_highlights:
            text, num_matches = highlighter.apply_highlight(
                text,
                all_highlights,
                color="violet",
            )

        if show_local_highlights:
            highlights_local = highlighter.get_highlights(
                text_clean, **highlights_kwargs
            )
            text, num_matches = highlighter.apply_highlight(
                text,
                highlights_local,
                color="blue",
            )
        else:
            highlights_local = None

        # with st.expander(title, expanded=True):
        with st.container(border=True):
            st.subheader(title)
            # st.write(doc.page_content)
            # st.markdown(doc.page_content)

            st.markdown(text, unsafe_allow_html=False)

            st_copy_to_clipboard_btn(result.text, key=f"copy_{i}")

            if st.button("Find Similar", key=f"find_similar_{i}"):
                st.query_params["q"] = result.text_clean
                st.rerun()
                return

            if highlights_local:
                with st.expander("Highlights", expanded=False):
                    st.write(highlights_local)
                    st_copy_to_clipboard_btn(
                        "\n".join(highlights_local), key=f"copy_{i}_highlights"
                    )

            if show_math_blocks and result.math:
                display_math(result.math, key=str(i))

            if show_code_blocks and result.code:
                display_code(result.code, key=str(i))

            if show_cleaned_text:
                with st.expander("Cleaned Text", expanded=False):
                    st.text(result.text_clean)
                    st_copy_to_clipboard_btn(result.text_clean, key=f"copy_{i}_clean")

            if show_raw_text:
                with st.expander("Raw Text", expanded=False):
                    # st.write(result.text)
                    st.text(result.text)
                    st_copy_to_clipboard_btn(result.text, key=f"copy_{i}_raw")

            if show_metadata:
                with st.expander("Metadata", expanded=False):
                    st.json(result.metadata, expanded=True)
                    st_copy_to_clipboard_btn(
                        json.dumps(result.metadata, indent=2), key=f"copy_{i}_meta"
                    )


def create_sidebar():
    # Options for Search
    st.sidebar.subheader("Search Options")
    search_type = st.sidebar.selectbox(
        "Search Type",
        [
            "mmr",
            "similarity",
            "similarity_score_threshold",
        ],
        index=0,
    )

    k = st.sidebar.slider("Number of Results (k)", 1, 25, 10)
    search_kwargs = dict(
        k=k,
    )
    match search_type:
        case "similarity_score_threshold":
            score_threshold = st.sidebar.slider("Score Threshold", 0.0, 1.0, 0.5, 0.01)
            search_kwargs["score_threshold"] = score_threshold
        case "mmr":
            fetch_k = st.sidebar.slider("Fetch K", 1, 100, 25)
            search_kwargs["fetch_k"] = fetch_k
            lambda_mult = st.sidebar.slider("Lambda Mult", 0.0, 1.0, 0.5, 0.01)
            search_kwargs["lambda_mult"] = lambda_mult

    # Options for Display
    st.sidebar.divider()
    st.sidebar.subheader("Display Options")

    display_kwargs = dict(
        show_summary=st.sidebar.toggle(
            "Summary",
            value=True,
            help="Display a summary of the entire search results (all documents combined).",
        ),
        show_wordcloud=st.sidebar.toggle(
            "Word Cloud",
            value=True,
            help="Display a word cloud of the entire search results (all documents combined).",
        ),
        show_global_highlights=st.sidebar.toggle(
            "Global Highlights",
            value=True,
            help="Highlight key-words/phrases extracted from the entire search results (all documents combined).",
        ),
        show_local_highlights=st.sidebar.toggle(
            "Local Highlights",
            value=False,
            help="Highlight key-words/phrases extracted from each individual document in the search results. NOTE: May be extremely slow for large number of results depending on the method used.",
        ),
        show_math_blocks=st.sidebar.toggle(
            "Math Blocks",
            value=True,
            help="Extract math blocks and display them in a collapsible section for each search result.",
        ),
        show_code_blocks=st.sidebar.toggle(
            "Code Blocks",
            value=True,
            help="Extract code blocks and display them in a collapsible section for each search result.",
        ),
        show_cleaned_text=st.sidebar.toggle(
            "Cleaned Text",
            value=False,
            help="Display the cleaned text of the search results.",
        ),
        show_raw_text=st.sidebar.toggle(
            "Raw Text",
            value=False,
            help="Display the raw text of the search results.",
        ),
        show_metadata=st.sidebar.toggle(
            "Metadata",
            value=True,
            help="Display the metadata of the search results.",
        ),
    )

    if display_kwargs["show_summary"]:
        st.sidebar.divider()
        st.sidebar.subheader("Summary Options")

        summary_kwargs = dict(
            method=st.sidebar.selectbox(
                "Summary Method",
                [
                    "summa",
                    "sumy",
                    # "transformers",
                    # "transformers (not implemented)",
                    # "openai (not implemented)",
                ],
                index=1,
                help="The method used to generate a summary of the entire search results (all documents combined).",
            ),
        )

        match summary_kwargs["method"]:
            case "summa":
                summary_kwargs["ratio"] = st.sidebar.slider(
                    "Ratio",
                    0.0,
                    1.0,
                    0.2,
                    help="The ratio of the summary to the entire text. Set to 0 to use the words instead.",
                )
                summary_kwargs["words"] = st.sidebar.slider(
                    "Words",
                    0,
                    1000,
                    0,
                    help="The number of words in the summary. Set to 0 to use the ratio instead.",
                )
            case "sumy":
                summary_kwargs["algorithm"] = st.sidebar.selectbox(
                    "Algorithm",
                    summarizer.Summarizer,
                    index=0,
                    help="The algorithm used to generate the summary.",
                )
                summary_kwargs["sentences_count"] = st.sidebar.slider(
                    "Sentences Count",
                    1,
                    25,
                    6,
                    help="The number of sentences in the summary.",
                )
            case "transformers":
                pass
            #     summary_kwargs["model"] = st.sidebar.selectbox(
            #         "Model",
            #         ["t5-small", "t5-base", "t5-large"],
            #         index=1,
            #         help="The model used to generate the summary.",
            #     )
            case "ChatGPT":
                pass
            case _:
                raise ValueError(f"Invalid summary method: {summary_kwargs['method']}")
        display_kwargs.update(
            summary_kwargs=summary_kwargs,
        )

    # if show_global_highlights or show_local_highlights:
    if (
        display_kwargs["show_global_highlights"]
        or display_kwargs["show_local_highlights"]
    ):
        st.sidebar.divider()
        st.sidebar.subheader("Highlights Options")

        highlights_kwargs = dict(
            method=st.sidebar.selectbox(
                "Highlights Extraction Method",
                highlighter.Highlighter,
                index=0,
                help="The method used to extract key-words/phrases which will be shown in a different color to emphasize their importance.",
            )
        )

        match highlights_kwargs["method"]:
            case highlighter.Highlighter.YAKE:
                highlights_kwargs["n"] = st.sidebar.slider(
                    "n",
                    1,
                    10,
                    3,
                    1,
                    help="The size of the sliding window for generating candidate keyphrases.",
                )
                highlights_kwargs["dedupLim"] = st.sidebar.slider(
                    "Deduplication Limit",
                    0.0,
                    0.99,
                    0.9,
                    0.01,
                    help="The deduplication limit. If the similarity between two candidate keyphrases is greater than this limit, the second keyphrase is considered a duplicate and is not added to the result set.",
                )
                highlights_kwargs["dedupFunc"] = st.sidebar.selectbox(
                    "Deduplication Function",
                    ["seqm", "jaro", "levs"],
                    index=0,
                    help="The function used to calculate the similarity between two candidate keyphrases for deduplication.",
                )
                highlights_kwargs["windowsSize"] = st.sidebar.slider(
                    "Window Size",
                    1,
                    10,
                    1,
                    help="The size of the window for the `DataCore` class, which is used to generate candidate keyphrases.",
                )
                highlights_kwargs["top"] = st.sidebar.slider(
                    "Top",
                    1,
                    50,
                    20,
                    help="The number of top-scoring keyphrases to return.",
                )
            case highlighter.Highlighter.RAKE_NLTK:

                class Metric(Enum):
                    """Different metrics that can be used for ranking."""

                    DEGREE_TO_FREQUENCY_RATIO = 0  # Uses d(w)/f(w) as the metric
                    WORD_DEGREE = 1  # Uses d(w) alone as the metric
                    WORD_FREQUENCY = 2  # Uses f(w) alone as the metric

                highlights_kwargs["ranking_metric"] = st.sidebar.selectbox(
                    "Ranking Metric",
                    Metric,
                    index=0,
                    help="The ranking metric to use when extracting key-words/phrases.",
                )
                highlights_kwargs["min_length"] = st.sidebar.slider(
                    "Min Length",
                    1,
                    10,
                    2,
                    help="Minimum limit on the number of words in a phrase.",
                )
                highlights_kwargs["max_length"] = st.sidebar.slider(
                    # default is 100000
                    "Max Length",
                    1,
                    100000,
                    100,
                    help="Maximum limit on the number of words in a phrase.",
                )
            case highlighter.Highlighter.MULTI_RAKE:
                highlights_kwargs["min_chars"] = st.sidebar.slider(
                    "Min Chars",
                    1,
                    10,
                    3,
                    help="The minimum number of characters a word must have to be considered as a keyword.",
                )
                highlights_kwargs["max_words"] = st.sidebar.slider(
                    "Max Words",
                    1,
                    10,
                    3,
                    help="The maximum number of words an extracted keyword can have.",
                )
                highlights_kwargs["min_freq"] = st.sidebar.slider(
                    "Min Frequency",
                    2,
                    10,
                    2,
                    help="The minimum frequency a candidate keyword needs to have to be considered as a keyword.",
                )

            case highlighter.Highlighter.TEXTRANK:
                highlights_kwargs["ratio"] = st.sidebar.slider(
                    "Ratio",
                    0.0,
                    1.0,
                    0.0,
                    help="The ratio of key-words/phrases to the entire text. Set to 0 to use the words instead.",
                )
                highlights_kwargs["words"] = st.sidebar.slider(
                    "Words",
                    0,
                    50,
                    20,
                    help="The number of key-words/phrases. Set to 0 to use the ratio instead.",
                )
                # highlights_kwargs["split"] = st.sidebar.checkbox(
                #     "Split",
                #     value=False,
                #     help="Whether to split the key-words/phrases when extracting.",
                # )
                highlights_kwargs["deaccent"] = st.sidebar.checkbox(
                    "Deaccent",
                    value=True,
                    help="Whether to deaccent the key-words/phrases when extracting.",
                )
            case _:
                raise ValueError(
                    f"Invalid highlights method: {highlights_kwargs['method']}"
                )

        display_kwargs.update(
            highlights_kwargs=highlights_kwargs,
        )

    st.sidebar.divider()
    st.sidebar.subheader("Miscellaneous Options")

    # display_kwargs.update(
    #     min_latex_commands=st.sidebar.slider(
    #         "Min Latex Command Length",
    #         0,
    #         10,
    #         2,
    #         1,
    #         help="The minimum number of LaTeX commands (e.g. \\frac, \\sum, etc.) in a math expression in order to be considered relevant enough to display.",
    #     ),
    # )

    return search_type, search_kwargs, display_kwargs


def main():
    st.title("Semantic Search Engine", anchor="top")

    if "search_results" not in st.session_state:
        st.session_state.search_results = {}

    if "global_highlights" not in st.session_state:
        st.session_state.global_highlights = {}

    search_type, search_kwargs, display_kwargs = create_sidebar()

    search_kwargs_hash = hash(frozenset(search_kwargs.items()))

    col1, col2 = st.columns([5, 1])
    with col1:
        # Create a search bar
        query = st.text_input(
            "Search Query",
            value=st.query_params.get("q", ""),
            on_change=None,
            placeholder="Enter a search query...",
            key="query",
            label_visibility="collapsed",
        )
        st.query_params["q"] = query
    with col2:
        search_button = st.button("Search")

    if search_button or query:
        # if search_button:
        if query:
            search_key = (query, search_kwargs_hash)

            if not st.session_state.search_results.get(search_key, None):
                st.session_state.search_results[search_key] = run_search(
                    query, search_type, **search_kwargs
                )

            display_search_results(
                search_key,
                **display_kwargs,
            )
        else:
            st.error("Please enter a search query.")


if __name__ == "__main__":
    main()
