#!/usr/bin/env bash

for i in {1..1}
do
    # i'm reducing the baseline learning rate since the scale of the
    # target will be more sensible, given that we're batching
    # manually. this is still bigger than in the paper, so perhaps
    # searching for better values could yield improved performance.

    # batch size is increased relative to the original example

    python main.py -n 1 -b 100 -blr 0.01 --z-pres-prior 0.01 --scale-prior-sd 0.2 --predict-net 200 --bl-predict-net 200 --decoder-output-use-sigmoid --decoder-output-bias -2 --eval-every 1 --cuda
    mv history.json elbo_$i.json

    # using prior of 0.5 here. 0.01 seems to hinder performance,
    # though intermediate values might plausiable yield better
    # performance.

    python main.py -n 1 -b 100 --z-pres-prior 0.5 --scale-prior-sd 0.2 --predict-net 200 --bl-predict-net 200 --decoder-output-use-sigmoid --decoder-output-bias -2 --eval-every 1 --rws --no-baselines --cuda
    mv history.json rws_$i.json
done
