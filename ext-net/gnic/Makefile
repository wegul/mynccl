#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
.DEFAULT_GOAL: build
include ../../makefiles/common.mk
SRCDIR   ?= $(abspath ../..)
BUILDDIR ?= .
NCCLDIR  := $(SRCDIR)/src/include
NCCLHDR := $(SRCDIR)/build/include
SRC_FILES := $(wildcard *.c) $(wildcard *.cc) $(wildcard rdma_util/*.c)
CUDA_HOME:=/usr/local/cuda

INC = -Inccl -Irdma_util -Irdma_util/jring -I$(CUDA_HOME)/include
LIB = -libverbs -lrdmacm


build: ${BUILDDIR}/libnccl-net-swg.so

${BUILDDIR}/libnccl-net-swg.so: ${SRC_FILES}
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${BUILDDIR}
	$(CXX) $(INC) -fPIC -shared -fvisibility=default -o $@ $^ $(LIB)

clean:
	rm -f ${BUILDDIR}/libnccl-net-swg.so
