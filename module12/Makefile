# ============================================================================
# Makefile — Julia Set Fractal Generator (OpenCL)
# ============================================================================
#
# Cross-platform build for Linux, macOS, and Windows.
#
# Usage:
#   Linux / macOS (GNU Make):
#     make              # build the executable
#     make run          # build and run with default settings
#     make list         # build and list available GPUs
#     make clean        # remove build artifacts
#
#   Windows (MSVC nmake from Developer Command Prompt):
#     nmake /F Makefile              # build
#     nmake /F Makefile run          # build and run
#     nmake /F Makefile list         # list GPUs
#     nmake /F Makefile clean        # clean
#
#   Override OpenCL SDK paths on Windows if needed:
#     nmake /F Makefile OPENCL_INC="C:\path\to\include" OPENCL_LIB="C:\path\to\lib"
#
# ============================================================================

# --------------------------------------------------------------------------
# Project settings
# --------------------------------------------------------------------------
TARGET     = juliaSet
SRC        = juliaSet.cpp
KERNEL     = julia_set.cl
HEADER     = helper_functions.h
CXXSTD     = c++17

# Default generation parameters (override on command line if desired)
DEVICE     = 0
WIDTH      = 1920
HEIGHT     = 1080

# --------------------------------------------------------------------------
# Platform detection
# --------------------------------------------------------------------------
# nmake on Windows does not define OS or UNAME, but defines OS=Windows_NT.
# GNU Make on Linux/macOS can detect via uname.
# --------------------------------------------------------------------------

# Windows (nmake)
!IF DEFINED(OS) && "$(OS)" == "Windows_NT"

# Windows paths — override these if your SDK is installed elsewhere
OPENCL_INC = C:\opencl-sdk\include
OPENCL_LIB = C:\opencl-sdk\lib

CXX        = cl
CXXFLAGS   = /std:$(CXXSTD) /EHsc /O2 /I "$(OPENCL_INC)" /nologo
LDFLAGS    = /link "$(OPENCL_LIB)\OpenCL.lib" /out:$(TARGET).exe
OUTFILE    = $(TARGET).exe
RM         = del /Q

$(OUTFILE): $(SRC) $(HEADER) $(KERNEL)
	$(CXX) $(SRC) $(CXXFLAGS) $(LDFLAGS)

run: $(OUTFILE)
	$(OUTFILE) -operation generate -device $(DEVICE) $(WIDTH) $(HEIGHT)

list: $(OUTFILE)
	$(OUTFILE) -operation list_gpu

clean:
	-$(RM) $(TARGET).exe $(TARGET).obj *.png 2>NUL

# Linux / macOS (GNU Make)
!ELSE

# Detect OS via uname
UNAME_S := $(shell uname -s)

CXX      := g++
CXXFLAGS := -std=$(CXXSTD) -O2 -Wall -Wextra

ifeq ($(UNAME_S),Darwin)
    # macOS: OpenCL is a system framework
    LDFLAGS := -framework OpenCL
else
    # Linux: link against libOpenCL
    LDFLAGS := -lOpenCL
endif

OUTFILE := $(TARGET)

.PHONY: all run list clean

all: $(OUTFILE)

$(OUTFILE): $(SRC) $(HEADER) $(KERNEL)
	$(CXX) $(CXXFLAGS) -o $(OUTFILE) $(SRC) $(LDFLAGS)

run: $(OUTFILE)
	./$(OUTFILE) -operation generate -device $(DEVICE) $(WIDTH) $(HEIGHT)

list: $(OUTFILE)
	./$(OUTFILE) -operation list_gpu

clean:
	rm -f $(OUTFILE) *.png

!ENDIF
