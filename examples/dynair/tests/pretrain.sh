#!/usr/bin/env bash
python3 opt.py data/single_obj.npz -b 25 --hold-out 39 -e 1 -s 1 bkg
cd runs/latest >/dev/null
export PARAM_DIR=`pwd -P`
cd - >/dev/null
python3 opt.py data/single_obj.npz -b 25 --hold-out 39 -e 1 all --bkg-params $PARAM_DIR/params-1.pytorch
