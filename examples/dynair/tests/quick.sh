#!/usr/bin/env bash
python3 opt.py data/multi_obj.npz -b 25 --hold-out 39 -e 1 all
python3 opt.py data/multi_obj.npz -b 25 --hold-out 39 -e 1 bkg
