# Julia Search Algorithms

A high-performance implementation of search algorithms in Julia, featuring both CPU and GPU-accelerated pathfinding for the vacuum world problem.

## Overview

This repository implements several search algorithms to solve vacuum world problems, where an agent (vacuum) must navigate through a grid world to collect all dirt particles while avoiding obstacles. The project includes:

- **A* Search**: Classic informed search algorithm
- **K-Best-First Search (KBFS)**: Parallel extension of best-first search
- **CUDA-accelerated KBFS**: GPU-accelerated heuristic computation for improved performance

## Features

- 🚀 **Multi-threaded parallel search** with K-Best-First Search
- 🔥 **CUDA GPU acceleration** for heuristic function computation
- 📊 **Benchmark tools** for performance analysis
- 🗺️ **Flexible problem representation** with support for custom maps
- 🎯 **Multiple search algorithms** for comparison

## Files Structure

### Core Algorithm Files
- `astar.jl` - A* search implementation
- `kbfs.jl` - K-Best-First Search (CPU parallel version)
- `astar_cuda.jl` - Complete CUDA-accelerated search implementation
- `cuda_kbfs.jl` - CUDA kernels and GPU-accelerated KBFS

### Problem Definition
- `problem.jl` - Abstract problem interface
- `vacuum_world.jl` - Vacuum world problem implementation with state representation

### Entry Point
- `main.jl` - Main execution file with benchmarking examples

### Data
- `data/` - Contains test instances and world maps
  - `world.dat`, `world2.dat` - Sample world files
  - `100/` - Hierarchical test data with varying parameters

## Problem Format

World files use the following format:
```
16 16          # Grid dimensions (rows cols)
Map:
################
#####  ### # # #
#     #       V#   # V = Vacuum (start position)
## #  #  #   # #   # # = Wall/obstacle
#         #    #   # * = Dirt (goal)
# #  # #  #  # #   #   = Empty space
#### #  # #  ###
#              #
#  # ####    # #
##### ##  #  # #
## # # # # ### #
#       #      #
##   #  #   #  #
##  #  # #  # ##
#*  #  # #  # ##   
################
```

## Usage

### Basic Example

```julia
include("main.jl")

# Load a world file
file = "data/world.dat"
map, start_state = parse_file(file)
problem = createVWProblem(map)

# Run A* search
solution = astar(start_state, problem)
path = reconstruct_path(solution)

# Run parallel K-Best-First Search
solution_kbfs = kbfs(start_state, problem, 8)  # 8 threads
```

### CUDA-Accelerated Search

```julia
# Run CUDA-accelerated KBFS
solution_cuda = kbfs_cuda(map, start_state, h_cuda, 64, 128)
# 64 CPU threads, 128 CUDA threads for heuristic computation
```

## Algorithm Details

### A* Search
- Classic informed search using f(n) = g(n) + h(n)
- Euclidean distance heuristic to remaining dirt particles
- Optimal solution guarantee

### K-Best-First Search (KBFS)
- Parallel extension of best-first search
- Expands k nodes simultaneously using multiple threads
- Maintains optimality while improving performance

### CUDA Acceleration
- GPU kernels for parallel heuristic computation
- Efficient memory management with pre-allocated buffers
- Significant speedup for large search spaces

## Dependencies

- Julia 1.6+
- CUDA.jl (for GPU acceleration)
- DataStructures.jl
- BenchmarkTools.jl (for performance testing)

## Performance

The CUDA-accelerated version shows significant performance improvements for problems with:
- Large search spaces
- Complex heuristic calculations
- Multiple simultaneous goal positions

Benchmark results can be obtained by running the timing code in `astar_cuda.jl`.

## Installation

1. Clone the repository
2. Install Julia dependencies:
   ```julia
   using Pkg
   Pkg.add("CUDA")
   Pkg.add("DataStructures")
   Pkg.add("BenchmarkTools")
   ```
3. Ensure CUDA toolkit is installed for GPU acceleration

## Contributing

Contributions are welcome! Areas for improvement:
- Additional search algorithms
- More sophisticated heuristics
- Extended problem domains
- Performance optimizations

## License

This project is open source. Please check the repository for license details.