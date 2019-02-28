#!/usr/bin/env bash

for i in {1..1}
do
    python3 sbn.py --cuda -n 1500 -tf 15 --save --loss elbo --baseline none
    mv history.json elbo_$i.json
    python3 sbn.py --cuda -n 1500 -tf 15 --save --loss elbo --baseline avg
    mv history.json elbo_avg_$i.json
    python3 sbn.py --cuda -n 1500 -tf 15 --save --loss elbo --baseline net
    mv history.json elbo_net_$i.json
    python3 sbn.py --cuda -n 500 -tf 5 --save --loss rws
    mv history.json rws_$i.json
done
