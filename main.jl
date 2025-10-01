include("problem.jl")
using .Problem
include("vacuum_world.jl")
using .VacuumWorld

include("astar.jl")
include("pastar.jl")
include("kbfs.jl")
include("spbfs.jl")
# include("cuda_kbfs.jl")

using BenchmarkTools
using Profile
using StatProfilerHTML

println(Threads.nthreads())

file = "data/100/0./10/5"
m::Matrix, s = VacuumWorld.parse_file(file)
problem = VacuumWorld.createVWProblem(m)

# res = pastar(s, problem, 7)
# println(res)
@btime res = spbfs(s, problem, 8)
# println(res)
# @benchmark res = astar(s, problem)
# @btime res = pastar(s, problem, 7)
# @btime res = kbfs(s, problem, 7)
# path = VacuumWorld.reconstruct_path(res)
# VacuumWorld.verify_path(problem, s, path)
# path = VacuumWorld.reconstruct_path(res)
# VacuumWorld.verify_path(problem, s, path)
# path = VacuumWorld.reconstruct_path(res)
# VacuumWorld.verify_path(problem, s, path)

# res = kbfs_cuda(s, problem, h_cuda, 64, 256)
# VacuumWorld.reconstruct_path(res)
