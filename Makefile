# Compiler and flags
NVCC = nvcc       

# Source and target
SRC = src/main.cpp
TARGET = build/main

# Included directories
WORKSPACEFOLDER = "."
LIBRARIES = "libraries/include"

# Config file
CONFIG = "config.json"

# Rule to compile the source into an executable
$(TARGET): $(SRC)
	$(NVCC) -I $(WORKSPACEFOLDER) -I $(LIBRARIES) $(SRC) -o $(TARGET)

# Clean rule to remove compiled files
clean:
	rm -f $(TARGET)

# Run rule to execute the program with srun
run: $(TARGET)
	srun --gres=shard:1 --cpus-per-task=4 --mem=2GB ./$(TARGET) $(CONFIG)
