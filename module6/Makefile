# Detect OS
ifeq($(OS), Windows_NT)
EXE = assignment.exe
RUN = .\$(EXE)
RM = del / Q
else
EXE = assignment
RUN = . / $(EXE)
RM = rm - f
endif

NVCC = nvcc
NVCC_FLAGS = -O2 - std = c++14

# Default target
all : build

# Compile
build :
$(NVCC) $(NVCC_FLAGS) assignment.cu - o $(EXE)

# Run default (no arguments)
run:
$(RUN)

# Run with 1024 threads per block
run_threads :
$(RUN) - threads 1024

# Run with 120000 blocks
run_blocks :
$(RUN) - num_blocks 120000

# Run all variations
run_all :
$(RUN)
$(RUN) - threads 1024
$(RUN) - num_blocks 120000

# Clean executable
clean :
$(RM) $(EXE)