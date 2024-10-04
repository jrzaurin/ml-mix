import multiprocessing
from typing import Any, List
from concurrent.futures.process import ProcessPoolExecutor

import spacy
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phraser, Phrases
from spacy.lang.en.stop_words import STOP_WORDS
from pytorch_widedeep.utils.fastai_transforms import partition_by_cores

cores = multiprocessing.cpu_count()


class NLTKLemmaTokenizer(object):
    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords
        self.lemmatizer = WordNetLemmatizer()

    def _process_one_doc(self, doc: str) -> List[str]:
        toks = simple_preprocess(doc, min_len=2)
        if self.remove_stopwords:
            toks = [t for t in toks if t not in STOP_WORDS]
        return [self.lemmatizer.lemmatize(t, pos="v") for t in toks]

    def _process_all_one_cpu(self, docs: List[str]) -> List[List[str]]:
        return [self._process_one_doc(str(doc)) for doc in docs]

    def process_all(self, docs: List[str]) -> List[List[str]]:
        with ProcessPoolExecutor(cores) as e:
            return sum(
                e.map(self._process_all_one_cpu, partition_by_cores(docs, cores)), []
            )


class SpacyLemmaTokenizer(object):
    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords
        self.tok = spacy.load("en_core_web_md", disable=["parser", "ner"])

    def condition(self, t, min_len=2):
        cond = (
            t.is_punct | t.is_space | (t.lemma_ != "-PRON-") | len(t)
            <= min_len | t.is_digit
        )

        if self.remove_stopwords:
            cond |= t.is_stop

        return not cond

    def _process_one_doc(self, doc: str) -> List[str]:
        return [t.lemma_.lower() for t in self.tok(doc) if self.condition(t)]

    def _process_all_one_cpu(self, docs: List[str]) -> List[List[str]]:
        return [self._process_one_doc(str(doc)) for doc in docs]

    def process_all(self, docs: List[str]) -> List[List[str]]:
        with ProcessPoolExecutor(cores) as e:
            return sum(
                e.map(self._process_all_one_cpu, partition_by_cores(docs, cores)), []
            )


class BigramTokenizer(object):
    def __init__(self):
        self.phraser = Phraser

    @staticmethod
    def _process_one_doc(doc: List[str], phrases_model: Any) -> List[str]:
        doc += [t for t in phrases_model[doc] if "_" in t]
        return doc

    def _process_all_one_cpu(self, docs: List[List[str]]) -> List[List[str]]:
        phrases = Phrases(docs, min_count=10)
        bigram = self.phraser(phrases)
        return [self._process_one_doc(doc, bigram) for doc in docs]

    def process_all(self, docs: List[List[str]]) -> List[List[str]]:
        with ProcessPoolExecutor(cores) as e:
            return sum(
                e.map(self._process_all_one_cpu, partition_by_cores(docs, cores)), []
            )
