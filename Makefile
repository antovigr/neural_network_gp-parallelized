# Compiler and flags
NVCC = nvcc       

# Source and target
SRC = src/main.cu
TARGET = build/main

# Source and target for debug program
DEBUG_SRC = debug.cu
DEBUG_TARGET = build/debug

# Included directories
WORKSPACEFOLDER = "."
LIBRARIES = "libraries/include"

# Config file
CONFIG = "config.json"

# Rule to compile the source into an executable
build: $(SRC)
	$(NVCC) -I $(WORKSPACEFOLDER) -I $(LIBRARIES) $(SRC) -o $(TARGET)

# Clean rule to remove compiled files
clean:
	rm -f $(TARGET)

# Run rule to execute the program with srun
run: $(TARGET)
	srun --gres=shard:1 --cpus-per-task=4 --mem=2GB ./$(TARGET) $(CONFIG)


# Rule to compile the debug source into an executable
debug: $(DEBUG_SRC)
	$(NVCC) -I $(WORKSPACEFOLDER) -I $(LIBRARIES) $(DEBUG_SRC) -o $(DEBUG_TARGET)

# Run debug rule to execute the debug program with srun
rundebug: $(DEBUG_TARGET)
	srun --gres=shard:1 --cpus-per-task=4 --mem=2GB ./$(DEBUG_TARGET) $(CONFIG)

cleandebug:
	rm -f $(DEBUG_TARGET) 