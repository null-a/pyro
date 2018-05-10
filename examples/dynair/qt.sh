#!/usr/bin/env bash
python3 opt.py data/single_obj.npz -b 25 --hold-out 39 -e 1 all
echo
python3 opt.py data/single_obj.npz -b 25 --hold-out 39 -e 1 bkg
echo
