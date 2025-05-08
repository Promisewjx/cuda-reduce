TARGET = reduce_test
NVCC = nvcc
SRCS = main.cu kernels/warp_reduce.cu kernels/block_reduce.cu kernels/grid_reduce.cu
INCLUDES = -I./include
ARCH = -arch=sm_80

$(TARGET): $(SRCS)
	$(NVCC) $(SRCS) $(INCLUDES) $(ARCH) -o $(TARGET)

clean:
	rm -f $(TARGET)
