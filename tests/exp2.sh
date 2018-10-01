echo "Experiment 2"
echo "CPU version"
for i in 1000 2000 4000 8000 16000
do
  ../heatdist $i 1000 0
done

echo "GPU version"
for i in 1000 2000 4000	8000 16000
do
  ../heatdist $i 1000 1
done
