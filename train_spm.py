import io
import unicodedata

import sentencepiece as spm
import tensorflow as tf


def _get_nsmc_nfd():
    with open("nsmc/ratings.txt") as f:
        for line in f:
            yield unicodedata.normalize("NFD", line.split("\t")[1])


spm.SentencePieceTrainer.train(
    sentence_iterator=_get_nsmc_nfd(),
    model_prefix="spm",
    vocab_size=5000,
    normalization_rule_name="identity",
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,
)
