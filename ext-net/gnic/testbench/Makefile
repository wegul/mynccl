# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -fPIC -g -O0

# Directories
NCCL_ROOT = /home/suweigao/mynccl
GNIC_DIR = $(NCCL_ROOT)/ext-net/gnic
BUILD_DIR = build
TARGET = test_program

# Include paths
INCLUDES = -I$(GNIC_DIR) \
           -I$(GNIC_DIR)/nccl

# Library paths and libraries
LIBDIRS = -L$(GNIC_DIR)
LIBS = -lnccl-net-swg

# CUDA paths (if needed)
CUDA_HOME ?= /usr/local/cuda
CUDA_INCLUDES = -I$(CUDA_HOME)/include
CUDA_LIBS = -L$(CUDA_HOME)/lib64 -lcudart

# Source files
SOURCES = mock_gpu.cpp
OBJECTS = $(SOURCES:%.cpp=$(BUILD_DIR)/%.o)

# Dependency files for tracking header changes
DEPFILES = $(SOURCES:%.cpp=$(BUILD_DIR)/%.d)

# Default target
all: $(BUILD_DIR) $(TARGET) ctrl_server

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build the plugin libraryfirst
$(GNIC_DIR)/libnccl-net-swg.so:
	$(MAKE) -C $(GNIC_DIR)

# Build target with proper linking
$(TARGET): $(OBJECTS) $(GNIC_DIR)/libnccl-net-swg.so
	$(CXX) -o $@ $(OBJECTS) $(LIBDIRS) $(LIBS) $(CUDA_LIBS) -Wl,-rpath,$(GNIC_DIR)

# Build control server
ctrl_server: ctrl_server.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# Compile source files and generate dependency files
$(BUILD_DIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDA_INCLUDES) -MMD -MP -c $< -o $@

# Include dependency files (for header changes)
-include $(DEPFILES)

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(TARGET) ctrl_server

# # Run the test
# run: $(TARGET)
# 	LD_LIBRARY_PATH=$(GNIC_DIR):$$LD_LIBRARY_PATH ./$(TARGET)

.PHONY: all clean run
