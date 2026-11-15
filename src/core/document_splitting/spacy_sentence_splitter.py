import os
from pathlib import Path
from typing import List

import spacy

from core.document_splitting.base_document_splitter import BaseDocumentSplitter


class SpacySentenceSplitter(BaseDocumentSplitter):
    def __init__(self) -> None:
        self.nlp = None

    async def _split(self, text) -> List[str]:
        if not self.nlp:
            self.__initialize()

        return [sent.text for sent in self.nlp(text).sents]

    def __initialize(self):
        cache_dir = Path.home() / ".cache" / "spacy"
        model_path = "en_core_web_sm"
        try:
            print(f"Loading spacy model {model_path}...")
            self.nlp = spacy.load(os.path.join(cache_dir, model_path))
        except OSError:
            print("Failed to load model. Downloading...")
            os.makedirs(cache_dir, exist_ok=True)
            spacy.cli.download(model_path)
            self.nlp = spacy.load(model_path)
            self.nlp.to_disk(os.path.join(cache_dir, model_path))
        print(f"Successfully loaded {model_path}")

        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")
