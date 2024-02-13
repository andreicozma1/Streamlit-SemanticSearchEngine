import logging
import os
from pprint import pformat as _pformat
from pprint import pprint as _pprint

import tiktoken

from . import extractors

logger = logging.getLogger(__name__)


def clean_text(text):
    text = extractors.remove_code(text)
    text = extractors.remove_latex(text)
    # text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.lower()
    return text


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def read_file(path: str, max_len=None) -> str:
    logger.debug(f"Reading file: {path}")

    path = path.strip()
    contents = ""

    if not os.path.isfile(path):
        logger.error(f"Input is not a file: {path}")
        return f"File not found: {path}"

    with open(path, "r", encoding="utf-8") as file:
        contents = file.read().strip()

    if contents == "":
        logger.warning(f"{path} is empty.")

    if max_len is not None and len(contents) > max_len:
        logger.warning(
            f"{path} is too long. Truncating to {max_len} chars. (actual: {len(contents)})"
        )
        contents = contents[:max_len]

    return contents


pprint = lambda x, *args, **kwargs: _pprint(x, *args, **kwargs, sort_dicts=False)

pformat = lambda x, *args, **kwargs: _pformat(x, *args, **kwargs, sort_dicts=False)
