params="--numCutoff=1000 --Nts=100 --seed=1769 --saveMode=2 --MCNodes=50 --maxNodes=100 --numCutoff=1000"
command="nice -20 julia Gen_Samples.jl"

ma="2e-5"
prob="1e-10"

for gag in {5e-10,1e-10,5e-11,1e-11,5e-12,1e-12,5e-13,1e-13,5e-14,1e-14,5e-15,1e-15}; do
    tag="convergence_$gag"
    var=" --MassA=$ma --Axg=$gag --ftag=$tag --probCutoff=$prob "
    echo "(time $command $var $params) &> ${tag}.log &"
    (time $command $var $params) &> ${tag}.log &
done
