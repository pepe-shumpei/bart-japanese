#!/bin/bash

TEXT=data/wiki/tokenized
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/wiki.train \
    --validpref $TEXT/wiki.valid \
    --testpref $TEXT/wiki.test \
    --destdir data-bin/bart \
    --workers 60
