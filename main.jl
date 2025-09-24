include("vacuum_world.jl")
using .VacuumWorld

include("astar.jl")
include("kbfs.jl")
include("cuda_kbfs.jl")

using BenchmarkTools

file = "data/100/0./1/5"
m::Matrix, s = VacuumWorld.parse_file(file)
problem = VacuumWorld.createVWProblem(m)
res = astar(s, problem)
VacuumWorld.reconstruct_path(res)
res = kbfs(s, problem, 8)
VacuumWorld.reconstruct_path(res)
res = kbfs_cuda(s, problem, h_cuda, 64, 256)
VacuumWorld.reconstruct_path(res)