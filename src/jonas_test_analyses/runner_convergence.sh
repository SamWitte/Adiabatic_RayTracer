params="--type=1  --numCutoff=1000 --Nts=100 --seed=1769 --type=1 --saveTree=1 --batchSize=1"
command="nice -20 julia Gen_Samples.jl"

ma="2e-5"
num="100"
prob="1e-100"

for gag in {1e-10,1e-11,1e-12,1e-13,1e-14,1e-15}; do
    tag="${gag}_${num}_${prob}_convergence"
    var="--MassA=$ma --Axg=$gag --ftag=$tag --probCutoff=$prob --maxNodes=$num"
    echo "(time $command $var $params) &> ${tag}.log &"
    (time $command $var $params) &> ${tag}.log &
done
