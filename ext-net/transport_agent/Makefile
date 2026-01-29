# MPIB Transport Agent Makefile

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g -pthread
LDFLAGS = -pthread

SRCDIR = src
INCDIR = include
BUILDDIR = build
BINDIR = $(BUILDDIR)
OBJDIR = $(BUILDDIR)/obj

# Source files (modular structure per phase1_plan.md)
SRCS = $(SRCDIR)/main.cc \
       $(SRCDIR)/shm.cc \
       $(SRCDIR)/ipc.cc \
       $(SRCDIR)/policy.cc

OBJS = $(patsubst $(SRCDIR)/%.cc,$(OBJDIR)/%.o,$(SRCS))
TARGET = $(BINDIR)/agent

.PHONY: all clean

all: $(TARGET)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BINDIR): | $(BUILDDIR)
	mkdir -p $(BINDIR)

$(OBJDIR): | $(BUILDDIR)
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@

$(TARGET): $(OBJS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -rf $(BUILDDIR)

# Install target
install: $(TARGET)
	install -d /usr/local/bin
	install -m 755 $(TARGET) /usr/local/bin/

# Run target for testing
run: $(TARGET)
	$(TARGET) --manual

.PHONY: install run
