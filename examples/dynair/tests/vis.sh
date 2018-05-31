#!/usr/bin/env bash

function run {
    python3 opt.py data/single_obj.npz -b 25 --hold-out 39 -e 1 $1 -s 1 all
    cd runs/latest >/dev/null
    export PARAM_DIR=`pwd -P`
    cd - >/dev/null
    python3 make_vis.py data/single_obj.npz $PARAM_DIR/module_config.json $PARAM_DIR/params-1.pytorch 0 frames
    python3 make_vis.py data/single_obj.npz $PARAM_DIR/module_config.json $PARAM_DIR/params-1.pytorch 0 movie
}

run ""
run "--use-depth"

# clean up
rm frames_0_input.png
rm frames_0_recon.png
rm frames_0_extra.png
rm movie_0.gif
rm tmp/frame_*.png
rm -r tmp
