CC=nvcc
SM=sm_52
all: heatdist.cu
	$(CC) -o heatdist -arch=$(SM) heatdist.cu 