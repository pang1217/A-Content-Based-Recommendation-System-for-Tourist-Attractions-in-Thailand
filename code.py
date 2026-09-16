from __future__ import annotations

import argparse
import contextlib
import html
import re
import os
import sys
import time
import unicodedata

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Word2Vec / Thai2Vec
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

# FastText
from gensim.models import FastText
from gensim.models.keyedvectors import FastTextKeyedVectors

# Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity

# UTF-8 OUTPUT
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# CONSTANTS
COLUMNS = [
    "place_id",
    "name",
    "category",
    "description_th",
    "description_en",
    "location",
    # "rating",
    # "review",
]

COLUMN_MAP = {
    "ATT_ID": "place_id",
    "ATT_NAME_TH": "name",
    "ATT_CATEGORY_LABEL": "category",
    "ATT_DETAIL_TH": "description_th",
    "ATT_DETAIL_EN": "description_en",
    "PROVINCE_NAME_TH": "location",
    # "rating",
    # "review",
}

DEFAULT_CSV_PATH = Path(__file__).with_name(
    "attraction.csv"
)

DEFAULT_SENTENCE_BERT_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

INSTALL_MESSAGE = (
    "Sentence-BERT requires the sentence-transformers package. "
    "Install it with: pip install sentence-transformers"
)

DOMAIN_STOPWORDS = {
    "ที่",
    "มี",
    "และ",
    "หรือ",
    "ใน",
    "ของ",
    "เป็น",
    "กับ",
    "ได้",
    "ให้",
    "จาก",
    "สำหรับ",
    "สามารถ",
    "นักท่องเที่ยว",
    "สถานที่",
    "เหมาะ",
    "เหมาะสำหรับ",
    "การ",
    "ชม",
    "เที่ยว",
}

# FORMAT VECTOR (preview of the first N dimensions of an embedding)
def format_vector(vector: np.ndarray, preview: int = 20) -> str:
    return np.array2string(
        np.asarray(vector)[:preview],
        precision=3,
        separator=", ",
        formatter={"float_kind": lambda value: f"{value:.4f}"},
    )

# CLEAN HTML
def clean_html(text: object) -> str:
    if pd.isna(text):
        return ""
    soup = BeautifulSoup(str(text), "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    return soup.get_text(separator=" ", strip=True)

# DATA CLEANING
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Original dataset shape: {df.shape}")

    # Drop rows with no ATT_DETAIL_TH
    df = df.dropna(subset=["ATT_DETAIL_TH"], how="all")
    print(f"After dropping NaN ATT_DETAIL_TH: {df.shape}")

    # Remove placeholder "-" and "*" values
    df = df[df["ATT_DETAIL_TH"].astype(str).str.strip() != "-"]
    df = df[df["ATT_DETAIL_TH"].astype(str).str.strip() != "*"]
    print(f"After removing '-' and '*' ATT_DETAIL_TH: {df.shape}")

    # Strip HTML markup
    df["ATT_DETAIL_TH"] = df["ATT_DETAIL_TH"].apply(clean_html)
    df["ATT_DETAIL_EN"] = df["ATT_DETAIL_EN"].apply(clean_html)

    # Remove rows with empty ATT_DETAIL_TH after HTML cleaning
    df = df[df["ATT_DETAIL_TH"].astype(str).str.strip() != ""]
    print(f"After removing empty ATT_DETAIL_TH: {df.shape}")

    # Remove duplicates
    before_dup = len(df)
    df = df.drop_duplicates( subset=["ATT_NAME_TH", "ATT_DETAIL_TH"], keep="first")
    after_dup = len(df)

    print(f"Removed duplicates: {before_dup - after_dup}")
    print(f"After duplicate removal: {df.shape}")
    return df

# SUPPRESS GENSIM STDERR
@contextlib.contextmanager
def suppress_native_stderr():
    """
    Silence low-level C/Cython stderr writes
    during Word2Vec/FastText training.
    """
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open( os.devnull, os.O_WRONLY)

    try:
        sys.stderr.flush()
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2( saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)

# TEXT PREPROCESSING
def clean_thai_text(text: object) -> str:
    if pd.isna(text):
        return ""
    value = html.unescape(str(text))
    value = unicodedata.normalize("NFC", value).lower()

    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[0-9]+", " ", value)
    value = re.sub( r"[^\u0E00-\u0E7Fa-zA-Z\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

# TOKENIZER
def get_thai_word_tokenizer(stopwords: Iterable[str]):
    try:
        from pythainlp import word_tokenize
        from pythainlp.corpus.common import thai_stopwords
        stopword_set = (set(thai_stopwords()).union(stopwords))

        def tokenize(text: str) -> list[str]:
            cleaned = clean_thai_text(text)
            tokens = word_tokenize( cleaned, engine="newmm", keep_whitespace=False)
            # print(tokens)
            return [token.strip() for token in tokens if (len(token.strip()) > 1 and token.strip() not in stopword_set)]
        return tokenize

    except ImportError:
        stopword_set = set(stopwords)
        def tokenize_without_pythainlp(text: str) -> list[str]:
            cleaned = clean_thai_text(text)

            return [token for token in cleaned.split() if (len(token) > 1 and token not in stopword_set)]
        return tokenize_without_pythainlp

def get_char_ngram_tokenizer(min_n: int = 3, max_n: int = 5) -> Callable[[str], list[str]]:
    def tokenize(text: str) -> list[str]:
        cleaned = clean_thai_text(text)
        ngrams: list[str] = []
        for word in cleaned.split():
            padded = f" {word} "
            for n in range(min_n, max_n + 1):
                ngrams.extend(padded[i:i + n] for i in range( max( 0, len(padded) - n + 1)))
        return ngrams
    return tokenize

# BASE RECOMMENDER
@dataclass
class BaseRecommender:
    df: pd.DataFrame
    analyzer: str = "word"
    text_columns: list[str] = field( default_factory=lambda: ["category" ,"description_th"])

    column_weights: dict[str, int] = field(default_factory=lambda: {"category": 3, "description_th": 1})
    stopwords: set[str] = field(default_factory=lambda: set(DOMAIN_STOPWORDS))
    fitted_df: pd.DataFrame | None = None

    # COMMON FEATURE TEXT
    def _build_feature_text(self, row: pd.Series) -> str:
        parts: list[str] = []
        for column in self.text_columns:
            value = clean_thai_text( row.get( column, ""))
            weight = max(1, int(self.column_weights.get(column, 1)))
            parts.extend([value] * weight)
        return " ".join(parts)

    # COMMON PLACE ID
    def recommend_by_place_id(self, place_id: int, top_n: int = 5) -> pd.DataFrame:
        self._ensure_fitted()
        matches = (self.fitted_df.index[ self.fitted_df["place_id"] == place_id].tolist())
        
        # if not found place id
        if not matches:
            raise ValueError(f"place_id not found: {place_id}")

        return self._similar_items(matches[0], top_n=top_n)

    # COMMON NAME
    def recommend_by_name(self, name: str, top_n: int = 5) -> pd.DataFrame:
        self._ensure_fitted()
        exact = (self.fitted_df.index[self.fitted_df["name"]== name].tolist())

        if exact:
            return self._similar_items(exact[0],top_n=top_n)

        # in case name not match
        return self._recommend_by_unknown_name(name, top_n)

    # COMMON TEXT
    def recommend_by_text(self, query: str, top_n: int = 5) -> pd.DataFrame:
        self._ensure_fitted()
        return self._recommend_by_text(query, top_n)

    # COMMON VECTOR (used by --show-vector)
    def get_vector(self, text: str) -> np.ndarray:
        self._ensure_fitted()
        return self._vector_for_text(text)

    # ABSTRACT METHODS
    def _vector_for_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def _recommend_by_unknown_name(self, name: str, top_n: int) -> pd.DataFrame:
        raise NotImplementedError

    def _recommend_by_text(self, query: str, top_n: int) -> pd.DataFrame:
        raise NotImplementedError

    def _similar_items(self, row_index: int,top_n: int) -> pd.DataFrame:
        raise NotImplementedError

    def _ensure_fitted(self) -> None:
        raise NotImplementedError

# All Model Implement BaseRecommender Class!
# TF-IDF (Term Frequency - Inverse Document Frequency)
@dataclass
class TourismTFIDFRecommender(BaseRecommender):
    analyzer: str = "char"
    vectorizer: (TfidfVectorizer | None) = None
    tfidf_matrix: object | None = None

    # FIT
    def fit(self) -> "TourismTFIDFRecommender":
        self.fitted_df = (self.df.copy().reset_index(drop=True))

        corpus = (self.fitted_df.apply(self._build_feature_text, axis=1).tolist())

        if self.analyzer == "word":
            self.vectorizer = (
                TfidfVectorizer(
                    tokenizer=
                        get_thai_word_tokenizer(
                            self.stopwords
                        ),
                    token_pattern=None,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.9,
                    sublinear_tf=True,
                    norm="l2",
                )
            )

        elif self.analyzer == "char":
            self.vectorizer = (
                TfidfVectorizer(
                    preprocessor=clean_thai_text,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                    norm="l2",
                )
            )

        else:
            raise ValueError("analyzer must be 'char' or 'word'")

        self.tfidf_matrix = (self.vectorizer.fit_transform(corpus))
        return self

    # UNKNOWN NAME
    def _recommend_by_unknown_name(self, name: str, top_n: int) -> pd.DataFrame:
        name_vectors = (self.vectorizer.transform(self.fitted_df["name"].map(clean_thai_text)))

        query_vector = (self.vectorizer.transform([clean_thai_text(name)]))
        best_index = int(np.argmax(cosine_similarity(query_vector, name_vectors)[0]))

        return self._similar_items(best_index, top_n=top_n)

    # TEXT QUERY
    def _recommend_by_text(self, query: str, top_n: int) -> pd.DataFrame:
        query_vector = (self.vectorizer.transform([clean_thai_text(query)]))

        scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        ranked_indices = (scores.argsort()[::-1][:top_n])
        return self._format_results(ranked_indices, scores)

    # SIMILAR ITEMS
    def _similar_items(self, row_index: int, top_n: int) -> pd.DataFrame:
        scores = cosine_similarity(self.tfidf_matrix[row_index], self.tfidf_matrix)[0]
        ranked_indices = [idx for idx in scores.argsort()[::-1] if idx != row_index][:top_n]

        return self._format_results(ranked_indices, scores, source_index=row_index)

    # FORMAT
    def _format_results(
        self,
        ranked_indices: Iterable[int],
        scores: np.ndarray,
        source_index: int | None = None
    ) -> pd.DataFrame:
        ranked_indices = list(ranked_indices)
        result = (self.fitted_df.iloc[ranked_indices].copy())
        result.insert(0, "similarity_score", [float(scores[idx]) for idx in ranked_indices])

        if source_index is not None:
            source_name = (self.fitted_df.loc[source_index, "name"])
            result.insert(0, "source_name", source_name)

        return result[
            [
                *(
                ["source_name"] if source_index is not None else []),
                "similarity_score",
                "place_id",
                "name",
                "category",
                "location",
                "description_th",
            ]
        ]

    # VECTOR (dense TF-IDF row for a piece of text)
    def _vector_for_text(self, text: str) -> np.ndarray:
        vector = self.vectorizer.transform([clean_thai_text(text)])
        return np.asarray(vector.todense()).ravel()

    # CHECK FIT
    def _ensure_fitted(self) -> None:
        if (self.vectorizer is None or self.tfidf_matrix is None or self.fitted_df is None):
            raise RuntimeError("Call fit() before requesting recommendations.")

# THAI2VEC
@dataclass
class TourismThai2VecRecommender(BaseRecommender):
    analyzer: str = "word"
    thai2vec_model_name: str = ("thai2fit_wv")
    vector_size: int = 100
    window: int = 5
    min_count: int = 1
    epochs: int = 300
    seed: int = 42
    embedding_model: (KeyedVectors | None) = None
    item_matrix: (np.ndarray | None) = None
    tokenizer: (Callable[[str], list[str]] | None) = None
    embedding_source: (str | None) = None

    # FIT
    def fit(self) -> "TourismThai2VecRecommender":
        self.fitted_df = (self.df.copy().reset_index(drop=True))
        corpus = (self.fitted_df.apply(self._build_feature_text, axis=1).tolist())

        if self.analyzer == "word":
            self.tokenizer = (get_thai_word_tokenizer(self.stopwords))
            self.embedding_model = (self._load_pretrained_thai2vec())
        elif self.analyzer == "char":
            self.tokenizer = (get_char_ngram_tokenizer())
            self.embedding_model = None
        else:
            raise ValueError("analyzer must be 'char' or 'word'")

        tokenized_corpus = [self.tokenizer(text) for text in corpus]
        tokenized_names = [self.tokenizer(name) for name in (self.fitted_df["name"].map(clean_thai_text))]
        training_sentences = [tokens for tokens in [*tokenized_corpus, *tokenized_names] if tokens]

        if not training_sentences:
            raise ValueError(
                "No usable tokens were found "
                "for Word2Vec training."
            )

        if self.embedding_model is None:
            self.embedding_model = (self._train_local_word2vec(training_sentences))
        
        self.item_matrix = np.vstack([self._tokens_to_vector(tokens) for tokens in tokenized_corpus])
        return self

    # UNKNOWN NAME
    def _recommend_by_unknown_name(self, name: str, top_n: int) -> pd.DataFrame:
        name_vectors = np.vstack([self._text_to_vector(value)for value in (self.fitted_df["name"])])
        query_vector = (self._text_to_vector(name).reshape(1, -1))

        best_index = int(np.argmax(cosine_similarity(query_vector, name_vectors)[0]))

        return self._similar_items(best_index, top_n=top_n)

    # TEXT QUERY
    def _recommend_by_text(self, query: str, top_n: int) -> pd.DataFrame:
        query_vector = (self._text_to_vector(query).reshape(1, -1))
        scores = cosine_similarity(query_vector, self.item_matrix)[0]

        ranked_indices = (scores.argsort()[::-1][:top_n])
        return self._format_results(ranked_indices, scores)

    # SIMILAR ITEMS
    def _similar_items(self, row_index: int, top_n: int) -> pd.DataFrame:
        source_vector = (self.item_matrix[row_index].reshape(1, -1))

        scores = cosine_similarity(source_vector, self.item_matrix)[0]
        ranked_indices = [ idx for idx in scores.argsort()[::-1] if idx != row_index ][:top_n]

        return self._format_results(ranked_indices, scores, source_index=row_index)

    # LOAD PRETRAINED THAI2VEC
    def _load_pretrained_thai2vec(self) -> KeyedVectors | None:
        try:
            from pythainlp.word_vector import (WordVector)
            self.embedding_source = (self.thai2vec_model_name)
            return (WordVector(self.thai2vec_model_name).get_model())

        except Exception:
            self.embedding_source = ("local_word2vec")
            return None

    # TRAIN LOCAL WORD2VEC
    def _train_local_word2vec(self, sentences: list[list[str]]) -> KeyedVectors:
        model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=os.cpu_count() or 4,
            sg=1,
            epochs=self.epochs,
            seed=self.seed,
        )
        self.embedding_source = ("local_word2vec")

        return model.wv

    # TEXT -> VECTOR
    def _text_to_vector(self, text: object) -> np.ndarray:
        return self._tokens_to_vector(self.tokenizer(str(text)))
 
    # TOKENS -> VECTOR
    def _tokens_to_vector(self, tokens: Iterable[str]) -> np.ndarray:
        vectors = [self.embedding_model[token] for token in tokens if token in self.embedding_model]

        if not vectors:
            return np.zeros(self.embedding_model.vector_size, dtype=np.float32)

        return np.mean(vectors, axis=0)

    # VECTOR (used by --show-vector)
    def _vector_for_text(self, text: str) -> np.ndarray:
        return self._text_to_vector(text)

    # FORMAT
    def _format_results(
        self,
        ranked_indices: Iterable[int],
        scores: np.ndarray,
        source_index: int | None = None
    ) -> pd.DataFrame:
        ranked_indices = list(ranked_indices)
        result = (self.fitted_df.iloc[ranked_indices].copy())
        result.insert(0, "similarity_score", [float(scores[idx]) for idx in ranked_indices])

        if source_index is not None:
            source_name = (self.fitted_df.loc[source_index, "name"])
            result.insert(0, "source_name", source_name)

        return result[
            [
                *(
                ["source_name"] if source_index is not None else []),
                "similarity_score",
                "place_id",
                "name",
                "category",
                "location",
                "description_th",
            ]
        ]

    # CHECK FIT
    def _ensure_fitted(self) -> None:
        if (self.embedding_model is None or self.item_matrix is None or self.fitted_df is None or self.tokenizer is None):
            raise RuntimeError("Call fit() before requesting recommendations.")

# FASTTEXT
@dataclass
class TourismFastTextRecommender(BaseRecommender):
    analyzer: str = "word"
    vector_size: int = 100
    window: int = 5
    min_count: int = 1
    epochs: int = 300
    min_n: int = 3
    max_n: int = 6
    seed: int = 42
    embedding_model: (FastTextKeyedVectors | None) = None
    item_matrix: (np.ndarray | None) = None
    tokenizer: (Callable[[str], list[str]]| None) = None

    # FIT
    def fit(self) -> "TourismFastTextRecommender":
        self.fitted_df = (self.df.copy().reset_index(drop=True))
        corpus = (self.fitted_df.apply(self._build_feature_text, axis=1).tolist())

        if self.analyzer == "word":
            self.tokenizer = (get_thai_word_tokenizer(self.stopwords))
        elif self.analyzer == "char":
            self.tokenizer = (get_char_ngram_tokenizer())
        else:
            raise ValueError("analyzer must be 'char' or 'word'")

        tokenized_corpus = [self.tokenizer(text) for text in corpus]
        
        # print(tokenized_corpus)
        tokenized_names = [self.tokenizer(name) for name in (self.fitted_df["name"].map(clean_thai_text))]
        # print(tokenized_names)

        training_sentences = [ tokens for tokens in [ *tokenized_corpus, *tokenized_names ] if tokens]

        if not training_sentences:
            raise ValueError(
                "No usable tokens were found "
                "for FastText training."
            )

        self.embedding_model = (self._train_fasttext(training_sentences))
        self.item_matrix = np.vstack([self._tokens_to_vector(tokens)for tokens in tokenized_corpus])
        return self

    # UNKNOWN NAME
    def _recommend_by_unknown_name(self, name: str, top_n: int) -> pd.DataFrame:
        name_vectors = np.vstack([self._text_to_vector(value) for value in (self.fitted_df["name"])])

        query_vector = (self._text_to_vector(name).reshape(1, -1))
        best_index = int(np.argmax(cosine_similarity(query_vector, name_vectors)[0]))

        return self._similar_items(best_index, top_n=top_n)

    # TEXT QUERY
    def _recommend_by_text(self, query: str, top_n: int) -> pd.DataFrame:
        query_vector = (self._text_to_vector(query).reshape(1, -1))

        scores = cosine_similarity(query_vector, self.item_matrix)[0]
        ranked_indices = (scores.argsort()[::-1][:top_n])

        return self._format_results(ranked_indices, scores)

    # SIMILAR ITEMS
    def _similar_items(self, row_index: int, top_n: int) -> pd.DataFrame:
        source_vector = (self.item_matrix[row_index].reshape(1, -1))
        scores = cosine_similarity(source_vector, self.item_matrix)[0]
        ranked_indices = [idx for idx in scores.argsort()[::-1] if idx != row_index ][:top_n]

        return self._format_results(ranked_indices, scores, source_index=row_index)

    # TRAIN FASTTEXT
    def _train_fasttext(self, sentences: list[list[str]]) -> FastTextKeyedVectors:
        model = FastText(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=os.cpu_count() or 4,
            sg=1,
            epochs=self.epochs,
            min_n=self.min_n,
            max_n=self.max_n,
            seed=self.seed,
        )
        return model.wv

    # TEXT -> VECTOR
    def _text_to_vector(self, text: object) -> np.ndarray:
        return self._tokens_to_vector(self.tokenizer(str(text)))

    # TOKENS -> VECTOR
    def _tokens_to_vector(self, tokens: Iterable[str]) -> np.ndarray:
        vectors = [self.embedding_model[token] for token in tokens if token]

        if not vectors:
            return np.zeros(self.embedding_model.vector_size, dtype=np.float32)

        return np.mean(vectors, axis=0)

    # VECTOR (used by --show-vector)
    def _vector_for_text(self, text: str) -> np.ndarray:
        return self._text_to_vector(text)

    # FORMAT
    def _format_results(
        self,
        ranked_indices: Iterable[int],
        scores: np.ndarray,
        source_index: int | None = None
    ) -> pd.DataFrame:
        ranked_indices = list(ranked_indices)
        result = (self.fitted_df.iloc[ranked_indices].copy())
        result.insert(0, "similarity_score", [float(scores[idx]) for idx in ranked_indices])

        if source_index is not None:
            source_name = (self.fitted_df.loc[source_index, "name"])
            result.insert(0, "source_name", source_name)

        return result[
            [
                *(
                ["source_name"] if source_index is not None else []),
                "similarity_score",
                "place_id",
                "name",
                "category",
                "location",
                "description_th",
            ]
        ]

    # CHECK FIT
    def _ensure_fitted(self) -> None:
        if (self.embedding_model is None or self.item_matrix is None or self.fitted_df is None or self.tokenizer is None):
            raise RuntimeError("Call fit() before requesting recommendations.")

# SENTENCE-BERT
@dataclass
class TourismSentenceBERTRecommender(BaseRecommender):
    analyzer: str = "sentence"
    text_columns: list[str] = field(default_factory=lambda: ["category","description_th"])

    model_name: str = (DEFAULT_SENTENCE_BERT_MODEL)

    batch_size: int = 32
    sentence_model: (object | None) = None
    sentence_matrix: (np.ndarray | None) = None

    # FIT
    def fit(self) -> "TourismSentenceBERTRecommender":
        if self.analyzer not in {"sentence", "word", "char"}:

            raise ValueError(
                "analyzer must be "
                "'sentence', 'word', or 'char'"
            )

        self.fitted_df = (self.df.copy().reset_index(drop=True))
        corpus = (self.fitted_df.apply(self._build_feature_text, axis=1).tolist())

        self.sentence_model = (self._load_sentence_bert_model())
        self.sentence_matrix = (self._encode_texts(corpus))
        return self

    # BUILD FEATURE TEXT
    def _build_feature_text(self, row: pd.Series) -> str:
        parts: list[str] = []
        for column in self.text_columns:
            value = clean_thai_text(row.get(column, ""))
            if value:
                parts.append(value)

        return " ".join(parts)

    # UNKNOWN NAME
    def _recommend_by_unknown_name(self, name: str, top_n: int) -> pd.DataFrame:
        name_vectors = (self._encode_texts(self.fitted_df["name"].map(clean_thai_text).tolist()))
        query_vector = (self._encode_texts([clean_thai_text(name)]))

        best_index = int(np.argmax(cosine_similarity(query_vector, name_vectors)[0]))
        return self._similar_items(best_index, top_n=top_n)

    # TEXT QUERY
    def _recommend_by_text(self, query: str, top_n: int) -> pd.DataFrame:
        query_vector = (self._encode_texts([clean_thai_text(query)]))

        scores = cosine_similarity(query_vector, self.sentence_matrix)[0]
        ranked_indices = (scores.argsort()[::-1][:top_n])

        return self._format_results(ranked_indices, scores)

    # SIMILAR ITEMS
    def _similar_items(self, row_index: int, top_n: int) -> pd.DataFrame:
        source_vector = (self.sentence_matrix[row_index].reshape(1, -1))
        scores = cosine_similarity(source_vector, self.sentence_matrix)[0]
        ranked_indices = [idx for idx in scores.argsort()[::-1] if idx != row_index] [:top_n]

        return self._format_results(ranked_indices, scores, source_index=row_index)

    # LOAD MODEL
    def _load_sentence_bert_model(self) -> object:
        try:
            from sentence_transformers import (SentenceTransformer)

        except ImportError as exc:
            raise ImportError(INSTALL_MESSAGE) from exc

        try:
            return SentenceTransformer(self.model_name)

        except Exception as exc:
            raise RuntimeError(
                f"Could not load Sentence-BERT "
                f"model {self.model_name!r}. "
                "Check your internet connection, "
                "Hugging Face cache, or pass a "
                "local model path with --model-name."
            ) from exc

    # ENCODE
    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = (
            self.sentence_model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

        return np.asarray(embeddings, dtype=np.float32)

    # VECTOR (used by --show-vector)
    def _vector_for_text(self, text: str) -> np.ndarray:
        return self._encode_texts([clean_thai_text(text)])[0]

    # FORMAT
    def _format_results(
        self,
        ranked_indices: Iterable[int],
        scores: np.ndarray,
        source_index: int | None = None
    ) -> pd.DataFrame:

        ranked_indices = list(ranked_indices)
        result = (self.fitted_df.iloc[ranked_indices].copy())
        result.insert(0, "similarity_score", [float(scores[idx]) for idx in ranked_indices])

        if source_index is not None:
            source_name = (self.fitted_df.loc[source_index, "name"])
            result.insert(0, "source_name", source_name)

        return result[
            [*(
                [
                "source_name"] if source_index is not None else []),
                "similarity_score",
                "place_id",
                "name",
                "category",
                "location",
                "description_th",
            ]
        ]

    # CHECK FIT
    def _ensure_fitted(self) -> None:
        if (self.sentence_model is None or self.sentence_matrix is None or self.fitted_df is None):
            raise RuntimeError("Call fit() before requesting recommendations.")

# LOAD DATA
def load_dataframe(csv_path: str | None = None) -> pd.DataFrame:
    path = (Path(csv_path) if csv_path else DEFAULT_CSV_PATH)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    print(f"Loading dataset -> {path}")

    df = pd.read_csv(path, low_memory=False)

    # CLEAN RAW HTML DATA (before renaming columns)
    df = clean_dataset(df)
    df = df.rename(columns=COLUMN_MAP)

    missing = (set(COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    df = df[COLUMNS].copy()
    df = df.reset_index(drop=True)
    print(f"Final dataset shape: {df.shape}")
    return df

# PRINT RESULTS
def print_results(title: str, results: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("=" * len(title))

    view = results.copy()
    view["description_th"] = (view["description_th"].str.slice(0, 80)+ "...")
    print(view.to_string(index=False))


def print_top_result_description_tokens(
    recommender: BaseRecommender,
    results: pd.DataFrame
) -> None :
    if results.empty:
        print("no results for tokens")
        return
    
    print("\n== top results : description_th tokens ===\n")
    # if recommender.analyzer == "word" :
    tokenizer = get_thai_word_tokenizer(recommender.stopwords)
    for rank, (_, row) in enumerate(results.iterrows(), start=1 ):
        description = row["description_th"]
        tokens = tokenizer(description)
        print(f"Name: {row['name']}")
        print(
            f"Similarity score: "
            f"{row['similarity_score']:.4f}"
        )
        print(f"Description tokens ({len(tokens)}):")
        print(tokens)
        print("\n")


def print_query_tokens(
    recommender: BaseRecommender,
    query: str,
) -> None:
    if query is None:
        return

    # if recommender.analyzer == "char":
    #     tokenizer = get_char_ngram_tokenizer()
    # else:
    tokenizer = get_thai_word_tokenizer(recommender.stopwords)

    tokens = tokenizer(str(query))

    print("\n== description tokens ===\n")
    print(f"description: {query}")
    print(f"Query tokens ({len(tokens)}):")
    print(tokens)
    print("\n")

def print_query_vector(
    recommender: BaseRecommender,
    model_name: str,
    query: str,
) -> None:
    if query is None:
        return

    vector = recommender.get_vector(str(query))

    print("\n== query vector ===\n")
    print(f"model: {model_name}")
    print(f"description: {query}")
    print(f"vector_shape: {vector.shape}")
    print(f"vector (first 20 dims): {format_vector(vector)}")
    print("\n")

def print_top_result_description_vectors(
    recommender: BaseRecommender,
    model_name: str,
    results: pd.DataFrame,
) -> None:
    if results.empty:
        print("no results for vectors")
        return

    print("\n== top results : description_th vectors ===\n")
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        description = row["description_th"]
        vector = recommender.get_vector(str(description))

        print(f"Rank {rank}")
        print(f"Name: {row['name']}")
        print(
            f"Similarity score: "
            f"{row['similarity_score']:.4f}"
        )
        print(f"model: {model_name}")
        print(f"vector_shape: {vector.shape}")
        print(f"vector (first 20 dims): {format_vector(vector)}")
        print("\n")

# BUILD RECOMMENDER
def build_recommender(model: str, df: pd.DataFrame, analyzer: str, show_tokens:bool = False):
    """
    Factory:
    build + fit a recommender by name.

    Returns:
        recommender,
        model_name,
        fit_seconds,
        error
    """

    started = time.perf_counter()
    try:
        with suppress_native_stderr():
            if model == "thai2vec":
                recommender = (TourismThai2VecRecommender(df=df, analyzer=analyzer).fit())
                model_name = "Thai2Vec"

            elif model == "fasttext":
                recommender = (TourismFastTextRecommender(df=df, analyzer=analyzer).fit())
                model_name = "Fasttext"

            elif model == "berta":
                recommender = (TourismSentenceBERTRecommender(df=df, analyzer=analyzer).fit())
                model_name = "Sentence-Berta"

            else:
                recommender = (TourismTFIDFRecommender(df=df, analyzer="char").fit())
                model_name = "TF-IDF"

        elapsed = (time.perf_counter() - started)
        return (recommender, model_name, elapsed, None)

    except Exception as exc:
        elapsed = (time.perf_counter() - started)

        return (None, model, elapsed, exc)

# SIDE-BY-SIDE COMPARISON
def build_side_by_side_comparison(
    results_by_model: dict[str, pd.DataFrame],
    top_n: int
) -> pd.DataFrame:

    order = [
        "TF-IDF",
        "Thai2Vec",
        "Fasttext",
        "Sentence-Berta"
    ]

    labels = {
        "TF-IDF": "TF-IDF",
        "Fasttext": "FastText",
        "Thai2Vec": "Thai2Vec",
        "Sentence-Berta": "BERT"
    }

    data: dict[str, pd.Series] = {}
    for key in order:
        label = labels[key]
        model_df = (results_by_model.get(key))

        if model_df is not None:
            names = (model_df["name"].reset_index(drop=True))
            scores = (model_df["similarity_score"].reset_index(drop=True))

        else:
            names = pd.Series(dtype=object)
            scores = pd.Series(dtype=float)

        data[label] = (names.reindex(range(top_n)))
        data[f"{label} Score"] = (scores.reindex(range(top_n)))

    return pd.DataFrame(data)

# MODEL RANKING
def build_model_ranking(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank models by top1_score
    from highest to lowest.
    """

    ranking = (summary_df.copy())

    ranking = ranking[ranking["top1_score"].notna()].copy()
    ranking = (ranking.sort_values(by="top1_score", ascending=False).reset_index(drop=True))

    ranking.insert(0, "Score Rank", range(1, len(ranking) + 1))
    ranking = ranking[
        [
            "Score Rank",
            "model",
            "top1_name",
            "top1_score",
            "fit_seconds",
            "query_seconds",
            "status",
        ]
    ]
    return ranking

# EXCEL NUMBER FORMAT
def apply_4f_number_format(worksheet, df: pd.DataFrame) -> None:
    float_cols = [i for i, dtype in enumerate(df.dtypes) if pd.api.types.is_float_dtype(dtype)]

    for col_idx in float_cols:
        excel_col = (col_idx + 1)
        for row in range(2, worksheet.max_row + 1):
            worksheet.cell(row=row, column=excel_col).number_format = ("0.0000")

# SAVE RESULTS
def save_results(
    results: pd.DataFrame,
    csv_path: str | None,
    excel_path: str | None
) -> None:
    """
    Save a single model's results
    to CSV and/or Excel.
    """
    if csv_path:
        results.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved CSV -> {csv_path}")

    if excel_path:
        Path(excel_path).parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            results.to_excel(writer, index=False, sheet_name="results")

            apply_4f_number_format(writer.sheets["results"], results)
        print(f"Saved Excel -> {excel_path}")

# SAVE ALL RESULTS To Excel or Csv files
def save_all_results(
    results_by_model: dict[str, pd.DataFrame],
    summary_df: pd.DataFrame,
    csv_path: str | None,
    excel_path: str | None,
    comparison_df: pd.DataFrame | None = None,
    ranking_df: pd.DataFrame | None = None,
) -> None:
    """
    Save every model's results.

    CSV:
        combined result
        comparison CSV

    Excel:
        summary
        model_ranking
        comparison
        individual model sheets
    """
    if csv_path:
        combined = pd.concat([df.assign(model=name) for name, df in results_by_model.items()], ignore_index=True)

        cols = (["model"] + [c for c in combined.columns if c != "model"])
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

        combined[cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved CSV -> {csv_path}")

        if comparison_df is not None:
            csv_path_obj = Path(csv_path)
            comparison_path = (csv_path_obj.with_name(f"{csv_path_obj.stem}_comparison{csv_path_obj.suffix}"))
            comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
            print(f"Saved CSV -> {comparison_path}")

    if excel_path:
        Path(excel_path).parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # SUMMARY
            summary_df.to_excel(writer, index=False, sheet_name="summary")

            # MODEL RANKING
            if ranking_df is not None:
                ranking_df.to_excel(writer, index=False, sheet_name="model_ranking")
                apply_4f_number_format(writer.sheets["model_ranking"], ranking_df)

            # COMPARISON
            if comparison_df is not None:
                comparison_df.to_excel(writer, index=False, sheet_name="comparison")
                apply_4f_number_format(writer.sheets["comparison"], comparison_df)

            # MODEL RESULTS
            for name, df in (results_by_model.items()):
                sheet_name = re.sub(r"[\\/*?:\[\]]", "_", name)[:31]

                df.to_excel(writer, index=False, sheet_name=sheet_name)

                apply_4f_number_format(writer.sheets[sheet_name], df)
        print(f"Saved Excel -> {excel_path}")

# GET PLACE NAME
def get_place_name(df: pd.DataFrame, place_id: int) -> str | None:
    matches = (df.index[df["place_id"]== place_id].tolist())

    if not matches:
        return None
    return df.loc[matches[0], "name"]

# GET PLACE DESCRIPTION (the text actually used for matching)
def get_place_description_by_id(df: pd.DataFrame, place_id: int) -> str | None:
    matches = (df.index[df["place_id"] == place_id].tolist())

    if not matches:
        return None
    return df.loc[matches[0], "description_th"]

def get_place_description_by_name(df: pd.DataFrame, name: str) -> str | None:
    matches = (df.index[df["name"] == name].tolist())

    if not matches:
        return None
    return df.loc[matches[0], "description_th"]

# RUN ALL MODELS
def run_all_models(df: pd.DataFrame, args: argparse.Namespace) -> None:
    model_ids = ["tfidf", "thai2vec", "fasttext", "berta"]

    sample_name = (args.name or "อุทยานแห่งชาติดอยอินทนนท์")
    sample_text = (args.text or "ทะเล น้ำใส ดำน้ำ พักผ่อน")

    summary_rows: list[dict] = []
    results_by_model: dict[str, pd.DataFrame] = {}

    for model_id in model_ids:
        (recommender, model_name, fit_seconds, fit_error) = build_recommender(model_id, df, args.analyzer, args.show_tokens)

        if fit_error is not None:
            print(
                f"\n[{model_name}] "
                f"fit failed after "
                f"{fit_seconds:.4f}s: "
                f"{fit_error}"
            )

            summary_rows.append({
                "model": model_name,
                "fit_seconds": round(fit_seconds, 4),
                "query_seconds": None,
                "top1_name": None,
                "top1_score": None,
                "status": (
                    f"error: {fit_error}"
                ),
            })
            continue

        query_started = (time.perf_counter())

        if args.place_id is not None:
            results = (recommender.recommend_by_place_id(args.place_id, top_n=args.top_n))

            place_name = (get_place_name(df, args.place_id))

            query_label = (
                f"{place_name} "
                f"(place_id={args.place_id})"
                if place_name
                else
                f"place_id={args.place_id}"
            )
            query_kind = "similar to"
            query_token_text = (
                get_place_description_by_id(df, args.place_id)
                or query_label
            )

        elif args.name:
            results = (recommender.recommend_by_name(args.name, top_n=args.top_n))
            query_label = args.name
            query_kind = "similar to"
            query_token_text = (
                get_place_description_by_name(df, args.name)
                or args.name
            )

        elif args.text:
            results = (recommender.recommend_by_text(args.text, top_n=args.top_n))
            query_label = args.text
            query_kind = "for query"
            query_token_text = args.text

        else:
            results = (recommender.recommend_by_name(sample_name, top_n=args.top_n))
            query_label = sample_name
            query_kind = "similar to"
            query_token_text = (
                get_place_description_by_name(df, sample_name)
                or sample_name
            )

        query_seconds = (time.perf_counter() - query_started)

        print_results(
            f"{model_name} "
            f"recommendations "
            f"{query_kind}: "
            f"{query_label}",
            results
        )

        if args.show_tokens:
            print_query_tokens(recommender, query_token_text)
            print_top_result_description_tokens(
                recommender,
                results,
            )

        if args.show_vector:
            print_query_vector(recommender, model_name, query_token_text)
            print_top_result_description_vectors(
                recommender,
                model_name,
                results,
            )

        print(
            f"[{model_name}] "
            f"fit: {fit_seconds:.3f}s | "
            f"query: {query_seconds:.4f}s"
        )

        results_by_model[model_name] = results
        top1 = results.iloc[0]
        summary_rows.append({
            "model": model_name,
            "fit_seconds": round(fit_seconds,4),
            "query_seconds": round(query_seconds,4),
            "top1_name": top1["name"],
            "top1_score": top1["similarity_score"],
            "status": "ok",
            }
        )

    # SIDE-BY-SIDE
    comparison_df = (build_side_by_side_comparison(results_by_model, args.top_n))

    print("\nSide-by-side comparison")
    print("=" * len("Side-by-side comparison"))
    print(comparison_df.to_string(index=False))

    # SUMMARY
    summary_df = pd.DataFrame(summary_rows)

    print("\nModel comparison summary")
    print("=" * len("Model comparison summary"))
    print(summary_df.to_string(index=False))

    # MODEL RANKING
    ranking_df = (build_model_ranking(summary_df))

    print("\nModel Ranking by Top-1 Score")
    print("=" * 35)
    print(ranking_df.to_string(index=False))

    # SAVE
    if (args.save_csv or args.save_excel):
        save_all_results(
            results_by_model,
            summary_df,
            args.save_csv,
            args.save_excel,
            comparison_df,
            ranking_df
        )

# ARGUMENT PARSER
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Content-based Thai tourism "
            "recommender using TF-IDF "
            "Thai2Vec, fastText or BERT."
        )
    )

    parser.add_argument(
        "--csv",
        help=(
            "Optional CSV path. "
            f"If omitted, "
            f"{DEFAULT_CSV_PATH.name} "
            "is used."
        )
    )

    parser.add_argument(
        "--model",
        choices=[ "tfidf", "thai2vec", "fasttext", "berta", "all"],
        default="tfidf",
        help=(
            "Recommendation model: "
            "tfidf, thai2vec, fasttext, "
            "berta, or all "
            "(compare every model)."
        )
    )

    parser.add_argument(
        "--analyzer",
        choices=[
            "char",
            "word"
        ],
        default="word",
        help=(
            "TF-IDF: char or word. "
            "Thai2Vec: word is recommended."
        )
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help=(
            "Number of recommendations "
            "to return."
        )
    )

    query = (
        parser
        .add_mutually_exclusive_group()
    )

    query.add_argument(
        "--place-id",
        type=int,
        help=(
            "Recommend places similar "
            "to this place_id."
        )
    )

    query.add_argument(
        "--name",
        help=(
            "Recommend places similar "
            "to this place name."
        )
    )

    query.add_argument(
        "--text",
        help=(
            "Recommend places from "
            "free-text Thai interest/query."
        )
    )

    parser.add_argument(
        "--save-csv",
        help=(
            "Save results to this CSV path. "
            "With --model all, all models "
            "are combined into one file "
            "with a 'model' column."
        )
    )

    parser.add_argument(
        "--save-excel",
        help=(
            "Save results to this .xlsx path. "
            "With --model all, each model "
            "gets its own sheet plus "
            "a 'summary' sheet."
        )
    )

    parser.add_argument(
        "--show-tokens",
        action="store_true", 
        help=(
            "Show description_th tokens for Top-N results. "
            "word = Thai word tokens; char = character n-grams."
        ),
    )

    parser.add_argument(
        "--show-vector",
        action="store_true",
        help=(
            "Show the embedding vector (first 20 dims) "
            "for the query and each Top-N result. "
            "Works for tfidf, thai2vec, fasttext, and berta."
        ),
    )

    return parser

# MAIN
def main() -> None:
    args = (build_parser().parse_args())
    df = load_dataframe(args.csv)

    # ALL MODELS
    if args.model == "all":
        run_all_models(df, args)
        return

    # SINGLE MODEL
    (recommender, model_name, fit_seconds, fit_error) = build_recommender(args.model, df, args.analyzer, args.show_tokens)

    if fit_error is not None:
        raise fit_error

    print(
        f"[{model_name}] "
        f"fit time: "
        f"{fit_seconds:.3f}s"
    )

    # PLACE ID
    if args.place_id is not None:
        results = (recommender.recommend_by_place_id(args.place_id, top_n=args.top_n))
        place_name = (get_place_name(df, args.place_id))

        title_label = (
            f"{place_name} "
            f"(place_id={args.place_id})"
            if place_name
            else
            f"place_id={args.place_id}"
        )

        print_results(
            f"{model_name} "
            f"recommendations "
            f"similar to "
            f"{title_label}",
            results
        )

        query_text = (
            get_place_description_by_id(df, args.place_id)
            or place_name
            or str(args.place_id)
        )

        if args.show_tokens:
            print_query_tokens(recommender, query_text)
            print_top_result_description_tokens(
                recommender,
                results,
            )

        if args.show_vector:
            print_query_vector(recommender, model_name, query_text)
            print_top_result_description_vectors(
                recommender,
                model_name,
                results,
            )

        save_results(results, args.save_csv, args.save_excel)

    # NAME
    elif args.name:
        results = (recommender.recommend_by_name(args.name, top_n=args.top_n))

        print_results(
            f"{model_name} "
            f"recommendations "
            f"similar to "
            f"{args.name}",
            results
        )

        query_text = (
            get_place_description_by_name(df, args.name)
            or args.name
        )

        if args.show_tokens:
            print_query_tokens(recommender, query_text)
            print_top_result_description_tokens(
                recommender,
                results,
            )

        if args.show_vector:
            print_query_vector(recommender, model_name, query_text)
            print_top_result_description_vectors(
                recommender,
                model_name,
                results,
            )

        save_results(results, args.save_csv, args.save_excel)

    # TEXT
    elif args.text:
        results = (recommender.recommend_by_text(args.text, top_n=args.top_n))

        print_results(
            f"{model_name} "
            f"recommendations "
            f"for query: "
            f"{args.text}", 
            results
        )

        if args.show_tokens:
            print_query_tokens(recommender, args.text)
            print_top_result_description_tokens(
                recommender,
                results,
            )

        if args.show_vector:
            print_query_vector(recommender, model_name, args.text)
            print_top_result_description_vectors(
                recommender,
                model_name,
                results,
            )

        save_results(results, args.save_csv, args.save_excel)

    # DEFAULT EXAMPLES
    else:
        examples = [
            (
                "อุทยานแห่งชาติดอยอินทนนท์",
                (
                    get_place_description_by_name(df, "อุทยานแห่งชาติดอยอินทนนท์")
                    or "อุทยานแห่งชาติดอยอินทนนท์"
                ),
                f"{model_name} recommendations similar to อุทยานแห่งชาติดอยอินทนนท์",
                recommender.recommend_by_name(
                    "อุทยานแห่งชาติดอยอินทนนท์",
                    top_n=args.top_n
                )
            ),
            (
                "ทะเล น้ำใส ดำน้ำ พักผ่อน",
                "ทะเล น้ำใส ดำน้ำ พักผ่อน",
                f"{model_name} recommendations for: ทะเล น้ำใส ดำน้ำ พักผ่อน",
                recommender.recommend_by_text(
                    "ทะเล น้ำใส ดำน้ำ พักผ่อน",
                    top_n=args.top_n
                )
            ),
            (
                "วัดสองพี่น้อง",
                (
                    get_place_description_by_name(df, "วัดสองพี่น้อง")
                    or "วัดสองพี่น้อง"
                ),
                f"{model_name} recommendations similar to วัดสองพี่น้อง",
                recommender.recommend_by_name(
                    "วัดสองพี่น้อง",
                    top_n=args.top_n
                )
            ),
        ]

        for query, token_text, title, results in examples:
            print_results( title, results )

            if args.show_tokens:
                print_query_tokens(recommender, token_text)
                print_top_result_description_tokens(
                    recommender,
                    results,
                )

            if args.show_vector:
                print_query_vector(recommender, model_name, token_text)
                print_top_result_description_vectors(
                    recommender,
                    model_name,
                    results,
                )

        if (args.save_csv or args.save_excel):
            combined = pd.concat( [ df.assign( query=query) for query, _, _, df in examples], ignore_index=True )
            cols = ( ["query"] + [ c for c in combined.columns if c != "query"])
            save_results(combined[cols], args.save_csv, args.save_excel)

# ENTRY POINT
if __name__ == "__main__":
    main()