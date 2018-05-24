#!/usr/bin/env bash

function run {
    python3 opt.py data/multi_obj.npz -b 50 --hold-out 19 -e 1 -s 1 --model-delta-w -g $1 all --bkg-params data/bkg_params.pytorch
}

# current best
run "-c 2e6"

# object rnn variations
# ------------------------------

# relu
run "-c 2e6 --guide-w rnn-relu-200-200"

# cnn embed
run "-c 2e6 --guide-input-embed cnn"

# extra rnn layer
run "-c 2e6 --guide-w rnn-tanh-200-200-200"

# 2x wider embed mlp
run "-c 2e6 --guide-input-embed mlp-1000-400"

# 2x wider everywhere
run "-c 2e6 --guide-input-embed mlp-1000-400 --guide-w rnn-tanh-400-400"

# z guide variations
# ------------------------------

# extra hidden layer in output
# (comparable to current best, isolates extra layer change)
run "-c 2e6 --guide-z noaux-mlp-100-100"

# side input to first layer
# same layer sizes as current best (isolates choice of where to feed in side info)
run "-c 2e6 --guide-window-embed id --guide-z noaux-mlp-100-100-100"
# something more sensible seeming. (still tiny compared to cnn though.)
run "-c 2e6 --guide-window-embed id --guide-z noaux-mlp-500-250"

# cnn embed
# (compare mlp with cnn. though may not realise full potential with
# only single hidden layer)
run "-c 2e6 --guide-window-embed cnn --window-size 24"
# something more sensible seeming
run "-c 2e6 --guide-window-embed cnn --window-size 24 --guide-z noaux-mlp-500-250"

# image so far variations
# ------------------------------

# mlp embed
# this is vaguely similar to rnn with mlp embed
run "-c 5e6 --guide-w isf-noblock-bkg-mlp-200-200"

# this is vaguely similar to rnn with cnn embed
run "-c 1e7 --guide-input-embed cnn --guide-w isf-noblock-bkg-mlp-200-200"
# try blocking grads
run "-c 1e7 --guide-input-embed cnn --guide-w isf-block-bkg-mlp-200-200"

# side input to first layer
# this looks odd, but gives similar layer sizes to mlp embed.
# this way we isolate the choice of where to feed in side info
run "-c 4e6 --guide-input-embed id --guide-w isf-noblock-bkg-mlp-500-200-200-200"

# also try something more sensible seeming?
run "-c 4e6 --guide-input-embed id --guide-w isf-noblock-bkg-mlp-1000-400-400"
