# Detect OS
ifeq ($(OS), Windows_NT)
EXE     = assignment.exe
RUN     = .\$(EXE)
RM      = del /Q
else
EXE     = assignment
RUN     = ./$(EXE)
RM      = rm -f
endif

NVCC        = nvcc
NVCC_FLAGS  = -O2 -std=c++14
LIBS        = -lnppc -lnppif -lnppicc -lcurand

# Default target
all: build

# Compile
build:
	$(NVCC) $(NVCC_FLAGS) assignment.cu -o $(EXE) $(LIBS)

# Run
run:
	$(RUN)

# Clean executable
clean:
	$(RM) $(EXE)
