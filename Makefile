# Makefile for Connected Component Analysis CUDA Project

# Compiler and flags
NVCC = nvcc
CFLAGS = -O3 -arch=sm_50
TARGET = connected_components
SOURCES = main.cu kernel.cu

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SOURCES) kernel.cuh
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCES)

# Clean target
clean:
	rm -f $(TARGET)

# Run target
run: $(TARGET)
	./$(TARGET)

# Debug build
debug: CFLAGS += -g -G
debug: $(TARGET)

.PHONY: all clean run debug