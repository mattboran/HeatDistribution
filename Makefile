CC=nvcc
SM=60
all: heatdist.cu
	$(CC) -o heatdist -arch=sm_$(SM) heatdist.cu 
