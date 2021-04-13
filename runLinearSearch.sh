#!/usr/bin/env bash

export HOME=/barrett/scratch/haozewu/
source /barrett/scratch/haozewu/leaderBoard/solvers/Marabou/py37/bin/activate

network=$1
target=$2
ind=$3

summaryFile=/barrett/scratch/haozewu/SOI/nn_robust_attacks/summaries/"$network"_tar"$target"_ind"$ind".txt
summaryFolder=/barrett/scratch/haozewu/SOI/nn_robust_attacks/"$network"_tar"$target"_ind"$ind"

if [ ! -f $summaryFolder ]
then
    mkdir $summaryFolder
fi

python /barrett/scratch/haozewu/SOI/nn_robust_attacks/linear_search.py /barrett/scratch/haozewu/leaderBoard/solvers/SOI/solver/maraboupy/runMarabou-ls.py $summaryFile $summaryFolder /barrett/scratch/haozewu/SOI/nn_robust_attacks/models/"$network".nnet --dataset=mnist --flip-strategy=random --verbosity=2 --work-dir /barrett/scratch/haozewu/leaderBoard/solvers/SOI/solver/ --offset=0.5 --target-label $target --index $ind --split-threshold=10
