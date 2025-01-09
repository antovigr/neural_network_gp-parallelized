# Compiler and flags
NVCC = nvcc                  # The CUDA compiler

# Source and target
SRC = debug.cu               # Your CUDA source file
TARGET = debug               # The output executable

# Included directories
WORKSPACEFOLDER = "."
INCLUDEPATH = "src/include"

# Rule to compile the source into an executable
$(TARGET): $(SRC)
	$(NVCC) -I$(WORKSPACEFOLDER) -I$(INCLUDEPATH) $(SRC) -o $(TARGET)

# Clean rule to remove compiled files
clean:
	rm -f $(TARGET)

# Run rule to execute the program with srun
run: $(TARGET)
	srun --gres=shard:1 --cpus-per-task=4 --mem=2GB ./$(TARGET)
