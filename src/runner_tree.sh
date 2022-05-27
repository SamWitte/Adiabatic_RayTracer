params="--type=1 --probCutoff=1e-10 --numCutoff=5 --Nts=100 --seed=1769"
command="nice -20 julia Gen_Samples.jl"

for ma in {1e-5,5e-6,1e-6}; do
 for gag in {1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9}; do
   tag="${ma}_${gag}"
   var="--MassA=$ma --Axg=$gag --ftag=$tag"
   echo "(time $command $var $params) &> ${tag}.log &"
   (time $command $var $params) &> ${tag}.log &
  done
  wait
done
