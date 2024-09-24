#!/bin/bash
# sh copy_lcc.sh 2022-01-01/17-02-24
for dir in "$@"
do
    scp -i ~/.ssh/id_ed25519 -r /project/msi290_uksr/generative_tod/outputs/$dir adibm@ric.csr.uky.edu:~/data/projects/ZSToD/outputs/$dir
done