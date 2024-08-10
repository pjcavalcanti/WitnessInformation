# Compiler
CXX = g++

# MKL root directory
MKLROOT = /opt/intel/oneapi/mkl/2024.2

# Include directory
INCDIR = $(MKLROOT)/include

# Library directory
LIBDIR = $(MKLROOT)/lib/intel64

# Compiler flags
CXXFLAGS = -std=c++11 -O3 -I$(INCDIR)

# Linker flags
LDFLAGS = -L$(LIBDIR) -Wl,-rpath,$(LIBDIR)

# Libraries
LIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -fopenmp

# Default source file
SRC = $(if $(FILE),$(FILE),main.cpp)

# Output executable (always a.out)
TARGET = a.out

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o a.out $(LDFLAGS) $(LIBS)

clean:
	rm -f a.out

.PHONY: all clean
