"""Strategy-neutral tools for deciding whether one bot should replace another."""

from garboid_pocketrocks.promotion.corpus import (
    CorpusPurpose,
    PromotionCase,
    PromotionCorpus,
    PromotionCorpusError,
    PromotionCorpusRecipe,
    corpus_snapshot_payload,
    load_promotion_corpus,
    validate_corpus_separation,
)

__all__ = [
    "CorpusPurpose",
    "PromotionCase",
    "PromotionCorpus",
    "PromotionCorpusError",
    "PromotionCorpusRecipe",
    "corpus_snapshot_payload",
    "load_promotion_corpus",
    "validate_corpus_separation",
]
