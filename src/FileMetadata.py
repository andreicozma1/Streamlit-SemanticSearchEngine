import hashlib
import os
import re
from datetime import datetime
import mimetypes

from . import utils


class ContentMetadata:
    def __init__(self, source: str):
        self.source = source
        self.mime, _ = mimetypes.guess_type(source)

        self.sha256 = self.get_sha256()

        self.size = os.path.getsize(source)

        # if it's a text file then also add more info
        if self.is_text:
            self.get_meta_text()

    def get_sha256(self):
        with open(self.source, "rb") as file:
            return hashlib.sha256(file.read()).hexdigest()

    def get_meta_text(self):
        meta_text = dict()
        with open(self.source, "r", encoding="utf-8") as file:
            content = file.read()
            # self.n_chars = len(content)
            # self.n_words = len(content.split())
            # self.n_tokens = utils.count_tokens(content)
            # self.n_lines = len(content.splitlines())
            # self.n_sentences = len(re.split(r"[.!?]+", content))
            # self.n_paragraphs = len(re.split(r"\n\s*\n", content))
            meta_text.update(
                dict(
                    n_chars=len(content),
                    n_words=len(content.split()),
                    n_tokens=utils.count_tokens(content),
                    n_lines=len(content.splitlines()),
                    n_sentences=len(re.split(r"[.!?]+", content)),
                    n_paragraphs=len(re.split(r"\n\s*\n", content)),
                )
            )
        return meta_text

    @property
    def mtime(self):
        mtime = os.path.getmtime(self.source)
        mtime = datetime.fromtimestamp(mtime)
        mtime = mtime.strftime("%Y-%m-%d %H:%M:%S")
        return mtime

    @property
    def ctime(self):
        ctime = os.path.getctime(self.source)
        ctime = datetime.fromtimestamp(ctime)
        ctime = ctime.strftime("%Y-%m-%d %H:%M:%S")
        return ctime

    @property
    def is_text(self):
        if self.mime is None:
            return False
        return self.mime.startswith("text/")

    @property
    def content(self):
        with open(self.source, "r", encoding="utf-8") as file:
            return file.read()

    def json(self):
        return dict(
            source=self.source,
            sha256=self.sha256,
            magic=self.mime,
            size=self.size,
            mtime=self.mtime,
            ctime=self.ctime,
            is_text=self.is_text,
            **self.get_meta_text(),
        )

    @classmethod
    def from_path(cls, path: str):
        return cls(source=path)
