#!/usr/bin/env bash

function run {
    python3 opt.py data/single_obj.npz -b 25 --hold-out 39 -e 1 $1 all
    echo
}

run "--guide-input-embed mlp-10-10"
run "--guide-input-embed resnet-10-10"
run "--guide-input-embed cnn"
run "--guide-input-embed id"

run "--guide-w rnn-tanh-10-10"
run "--guide-w rnn-relu-10-10"
run "--guide-w isf-block-mlp-10-10"
run "--guide-w isf-noblock-mlp-10-10"
run "--guide-w isf-noblock-resnet-10-10"

run "--guide-window-embed mlp-10-10"
run "--guide-window-embed resnet-10-10"
run "--guide-window-embed cnn --window-size 24"
run "--guide-window-embed id"

run "--guide-z mlp-10-10"
run "--guide-z resnet-10-10"
