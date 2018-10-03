CC=nvcc
SM=52
all: heatdist.cu
	$(CC) -o heatdist -arch=sm_$(SM) heatdist.cu 
