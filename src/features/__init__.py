"""Stylometric feature extraction subpackage.

The main entry point is :class:`FeatureExtractor` in ``extractor.py``,
which combines features from the individual modules:

* ``lexical``     — vocabulary richness and word-shape features
* ``syntactic``   — POS and dependency-tree based features
* ``readability`` — Flesch, Kincaid, SMOG, Gunning Fog
* ``discourse``   — discourse markers and connectives
* ``punctuation`` — punctuation distribution
* ``function_words`` — Mosteller–Wallace style function-word vector
"""
from .extractor import FeatureExtractor  # noqa: F401
