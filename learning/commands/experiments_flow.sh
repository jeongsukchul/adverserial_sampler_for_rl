wandb_project=walker-sampler3
task=WalkerWalk
success_threshold=.7

for beta in 10 2 1 0.66 0.5
do
for gamma in 0. .5 1. 2.
do
    for seed in 0 1 2 3 4 5
    do
       python run.py policy=flowppo task=$task wandb_project=$wandb_project beta=$beta gamma=$gamma seed=$seed 
    done
done
done
