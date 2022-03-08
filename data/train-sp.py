# train.py
import sentencepiece as spm

# 学習
spm.SentencePieceTrainer.Train('--input=wiki/original/wiki.train, --model_prefix=sentencepiece --character_coverage=1.0 --vocab_size=32000')
