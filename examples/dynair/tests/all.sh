#!/usr/bin/env bash

function run {
    python3 opt.py data/single_obj.npz -b 25 --hold-out 39 -e 1 $1 all
}

run "--use-depth"

run "--guide-input-embed mlp-10-10"
run "--guide-input-embed resnet-10-10"
run "--guide-input-embed cnn"
run "--guide-input-embed id"

run "--guide-w rnn-tanh-10-10"
run "--guide-w rnn-relu-10-10"
run "--guide-w isf-block-nobkg-mlp-10-10"
run "--guide-w isf-noblock-bkg-mlp-10-10"
run "--guide-w isf-noblock-bkg-resnet-10-10"

run "--guide-window-embed mlp-10-10"
run "--guide-window-embed resnet-10-10"
run "--guide-window-embed cnn --window-size 24"
run "--guide-window-embed cnn --window-size 24 --use-depth"
run "--guide-window-embed id"

run "--guide-z auxignore-mlp-10-10"
run "--guide-z auxignore-resnet-10-10"

run "--guide-z auxside-mlp-10-10"

run "--guide-z auxmain-mlp-10-10 --guide-w isf-noblock-bkg-mlp-10-10"
run "--guide-z auxside-mlp-10-10 --guide-w isf-noblock-bkg-mlp-10-10"

# Can't do this at present, as the aux input includes depth channel,
# which means the aux and main don't have the same dimension.
# run "--guide-z auxmain-mlp-10-10 --guide-w isf-noblock-bkg-mlp-10-10 --use-depth"
run "--guide-z auxside-mlp-10-10 --guide-w isf-noblock-bkg-mlp-10-10 --use-depth"

run "--w-transition sdparam-mlp-10-10"
run "--w-transition sdstate-mlp-10-10"
run "--w-transition sdparam-resnet-10-10"
run "--w-transition sdstate-resnet-10-10"

run "--z-transition sdparam-mlp-10-10"
run "--z-transition sdstate-mlp-10-10"
run "--z-transition sdparam-resnet-10-10"
run "--z-transition sdstate-resnet-10-10"

run "--decode-obj mlp-10-10"
run "--decode-obj resnet-10-10"
