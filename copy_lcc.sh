#!/bin/bash
scp -i ~/.ssh/id_ed25519 -r /project/msi290_uksr/generative_tod/outputs/$1 adibm@ric.csr.uky.edu:~/projects/generative_tod/outputs/$1
