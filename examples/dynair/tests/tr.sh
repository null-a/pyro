#!/usr/bin/env bash

function run_full {
    python3 opt.py $2 -b 50 --hold-out 27 -l 20 -e 1 -s 1 --log-elbo 0 --host cpu --w-transition sdstate-mlp-50 --z-transition sdstate-mlp-50 --decode-obj mlp-500-1000 --model-delta-w --guide-w isf-noblock-nobkg-mlp-800-800 --guide-z auxside-mlp-1000-500 --guide-input-embed cnn --guide-window-embed id -n -c 3e6 -g --use-depth $1 all --bkg-params $3
}

function run_alt {
    # diff with full model:
    # -c 2e6 --window-size 24 --guide-window-embed cnn --guide-z auxmain-mlp-500-250
    python3 opt.py $2 -b 50 --hold-out 27 -l 20 -e 1 -s 1 --log-elbo 0 --host cpu --w-transition sdstate-mlp-50 --z-transition sdstate-mlp-50 --decode-obj mlp-500-1000 --model-delta-w --guide-w isf-noblock-nobkg-mlp-800-800 --guide-z auxmain-mlp-500-200 --guide-input-embed cnn --guide-window-embed cnn --window-size 24 -n -c 3e6 -g --use-depth $1 all --bkg-params $3

}

# full model
# ---------------

run_full "--desc obj11_s_s"  "data/tr_obj11_s.npz"  "data/bkg_tr_obj11_s.pytorch"
run_full "--desc obj11_s_tr" "data/tr_obj11_s.npz"  "data/bkg_tr_obj13_tr.pytorch"
run_full "--desc obj11_t"    "data/tr_obj11_t.npz"  "data/bkg_tr_obj13_tr.pytorch"
run_full "--desc obj11_tr"   "data/tr_obj11_tr.npz" "data/bkg_tr_obj13_tr.pytorch"
run_full "--desc obj13_tr"   "data/tr_obj13_tr.npz" "data/bkg_tr_obj13_tr.pytorch"

# with window embed cnn
# ---------------------

run_alt "--desc alt" "data/tr_obj13_tr.npz" "data/bkg_tr_obj13_tr.pytorch"
