#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
.DEFAULT_GOAL: build
include ../../makefiles/common.mk
SRCDIR   ?= $(abspath ../..)
BUILDDIR ?= .
NCCLDIR  := $(BUILDDIR)

SRC_FILES := $(wildcard src/*.cc)

PLUGIN_NAME := mpib
TARGET_LIB  := libnccl-net-$(PLUGIN_NAME).so

build: ${BUILDDIR}/${TARGET_LIB}

${BUILDDIR}/${TARGET_LIB}: ${SRC_FILES}
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${BUILDDIR}
	$(CXX) $(CXXFLAGS) -DNCCL_BUILD_RDMA_CORE=1 -DNCCL_BUILD_MLX5DV=1 -Iinclude -Inccl -Isrc -shared -o $@ $^ $(LDFLAGS) -libverbs -lrdmacm -lpthread -ldl -lmlx5

clean:
	rm -f ${BUILDDIR}/${TARGET_LIB}
