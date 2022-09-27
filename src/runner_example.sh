params="--probCutoff=1e-10 --numCutoff=50 --MCNodes=10 --maxNodes=100 --Nts=1000 --seed=-1 --MassA=2e-5 --Axg=1e-14 --saveMode=1"
command="nice -20 julia Gen_Samples.jl"

for tag in {1,2,3,4,5,6}; do
  (time $command $params --ftag=${tag}) &> ${tag}.log &
done
wait

python3 Combine_Files.py results/npy/tree_*.npy
