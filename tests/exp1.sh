echo "Experiment 1"
echo "CPU version"
for i in 1000 2000 4000 8000 16000 32000
do
  ./heatdist 1000 $i 0
done

echo "GPU version"
for i in 1000 2000 4000	8000 16000 32000
do
  ./heatdist 1000 $i 1
done
