#!/usr/bin/env bash

function run {
    python3 opt.py data/multi_obj.npz -b 50 --hold-out 19 -e 1 -s 1 --model-delta-w -n $1 all --bkg-params data/bkg_params.pytorch
}

# explicit depth (with inf/nan checks enabled)

# this is intended to be a very similar setup to previous run that
# used a depth implementation that encoded depth in z. i'm using aux
# as side input as previous depth implementation passed depth channel
# as part of aux, which caused it to be included as side info since it
# had one more channel than the main input.
run "-c 1e7 --w-transition sdstate-mlp-50 --z-transition sdstate-mlp-50 --decode-obj mlp-500-1000 --guide-w isf-noblock-nobkg-mlp-800-800 --guide-z auxside-mlp-1000-500 --guide-input-embed cnn --guide-window-embed id -g --use-depth"

# ... with grads blocked (since opus0 suggested this works better)
run "-c 1e7 --w-transition sdstate-mlp-50 --z-transition sdstate-mlp-50 --decode-obj mlp-500-1000 --guide-w isf-block-nobkg-mlp-800-800 --guide-z auxside-mlp-1000-500 --guide-input-embed cnn --guide-window-embed id -g --use-depth"

# --------------------

# re-run best from opus0
run "-c 2e6 --guide-window-embed cnn --window-size 24 --guide-z auxignore-mlp-500-250"

# ... with drift model for z
run "-c 2e6 --guide-window-embed cnn --window-size 24 --guide-z auxignore-mlp-500-250 --model-delta-z"

# ... with extra hidden layer in transitions
run "-c 2e6 --guide-window-embed cnn --window-size 24 --guide-z auxignore-mlp-500-250 --w-transition sdparam-mlp-50-50 --z-transition sdparam-mlp-50-50"

# ... with extra hidden layer in transitions & resnet style
run "-c 2e6 --guide-window-embed cnn --window-size 24 --guide-z auxignore-mlp-500-250 --w-transition sdparam-resnet-50-50 --z-transition sdparam-resnet-50-50"
