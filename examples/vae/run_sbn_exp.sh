#!/usr/bin/env bash

for i in {1..1}
do
    python3 sbn.py -n 1 --save --loss elbo --baseline none
    mv history.json elbo_$i.json
    python3 sbn.py -n 1 --save --loss elbo --baseline avg
    mv history.json elbo_avg_$i.json
    python3 sbn.py -n 1 --save --loss elbo --baseline net
    mv history.json elbo_net_$i.json
    python3 sbn.py -n 1 --save --loss rws
    mv history.json rws_$i.json
done
