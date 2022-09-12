#!/bin/bash
# sh copy_lcc.sh 2022-01-01/17-02-24
scp -i ~/.ssh/id_ed25519 -r /project/msi290_uksr/generative_tod/outputs/$1 adibm@ric.csr.uky.edu:~/projects/generative_tod/outputs/$1
