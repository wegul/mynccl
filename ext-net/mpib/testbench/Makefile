# Skeleton Makefile for mpib testbench
# This is intentionally minimal; fill in actual build rules later.

.DEFAULT_GOAL := build

# NCCL root (assumes this Makefile is under ext-net/mpib/testbench)
ROOT        ?= ../../..
BUILDDIR    ?= $(abspath ./build)
CXX         ?= g++
CXXFLAGS    ?= -std=c++17 -O2 -g -Wall -Wextra
LDFLAGS     ?=

INCLUDES    := -I$(ROOT)/src -I$(ROOT)/ext-net/mpib/nccl
LIBS        := -L$(ROOT)/build/lib -lnccl

# Binaries to build (you will implement these later)
BINARIES := endpoint_mpib ctrl_server_mpib

build: $(BINARIES)

endpoint_mpib: endpoint.cpp
	@echo "[TODO] Compile endpoint_mpib once endpoint.cpp is implemented"
	@mkdir -p $(BUILDDIR)
	@# Example compile line (uncomment and adjust later):
	@# $(CXX) $(CXXFLAGS) $(INCLUDES) -o $(BUILDDIR)/$@ $^ $(LIBS) $(LDFLAGS)

ctrl_server_mpib: ctrl_server.cpp
	@echo "[TODO] Compile ctrl_server_mpib once ctrl_server.cpp is implemented"
	@mkdir -p $(BUILDDIR)
	@# Example compile line (uncomment and adjust later):
	@# $(CXX) $(CXXFLAGS) $(INCLUDES) -o $(BUILDDIR)/$@ $^ $(LIBS) $(LDFLAGS)

clean:
	@rm -rf $(BUILDDIR)
