params="--probCutoff=1e-10 --numCutoff=5 --Nts=100 --seed=1769 --MassA=2e-5 --Axg=1e-14"
command="nice -20 julia Gen_Samples.jl"

for tag in {1,2,3,4,5}; do
  (time $command $params --ftag=${tag}) &> ${tag}.log &
done
wait

python3 Combine_Files.py results/npy/tree_*.npy
