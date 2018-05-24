#!/usr/bin/env bash

function run {
    python3 opt.py data/single_obj.npz -b 25 --hold-out 39 -e 1 $1 all
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

run "--guide-z noaux-mlp-10-10"
run "--guide-z noaux-resnet-10-10"

run "--guide-z aux-mlp-10-10"
run "--guide-z aux-mlp-10-10 --guide-w isf-noblock-mlp-10-10"

run "--w-transition sdparam-mlp-10-10"
run "--w-transition sdstate-mlp-10-10"

run "--z-transition sdparam-mlp-10-10"
run "--z-transition sdstate-mlp-10-10"
