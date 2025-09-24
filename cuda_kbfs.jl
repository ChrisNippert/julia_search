using CUDA
include("problem.jl")
using .Problem
include("vacuum_world.jl")
using .VacuumWorld
using Base.Enums
using Base.Threads
using DataStructures

function h_cuda_kernel!(x_vec::CuDeviceVector{<:AbstractFloat},
                        y_vec::CuDeviceVector{<:AbstractFloat},
                        gx_vec::CuDeviceVector{<:AbstractFloat},
                        gy_vec::CuDeviceVector{<:AbstractFloat},
                        h_returns::CuDeviceVector{<:AbstractFloat})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if 1 <= i <= length(h_returns)
        if gx_vec[i] == -1
            h_returns[i] = 0
        else
            h_returns[i] = sqrt((gx_vec[i] - x_vec[i])^2 + (gy_vec[i] - y_vec[i])^2)
        end
    end
    return
end

function test_cuda_kernel()
    x_vec = CuArray([0.0])
    y_vec = CuArray([0.0])

    gx_vec = CuArray([3.0])
    gy_vec = CuArray([4.0])
    ret = CUDA.zeros(Float32, length(x_vec))

    N = length(x_vec)
    threads = min(256, N)
    blocks  = cld(N, threads)
    CUDA.@sync @cuda threads=threads blocks=blocks h_cuda_kernel!(x_vec, y_vec, gx_vec, gy_vec, ret)
    @assert Array(ret) == [5]
end

test_cuda_kernel()

# Some premade vectors for storage
max_n = 256 # largest number of states you expect
xs_buf = CUDA.zeros(Float32, max_n)
ys_buf = CUDA.zeros(Float32, max_n)
goalxs_buf = CUDA.zeros(Float32, max_n)
goalys_buf = CUDA.zeros(Float32, max_n)
heuristics_buf = CUDA.zeros(Float32, max_n)

function h_cuda(s_vec::Vector{Any}, threads::Int64)
    # for each game state, extract the x and y and add then to the arrays. For the goals, extract the x,y into their matrices
    # convert to cuda arrays
    xs = []
    ys = []
    goalxs = []
    goalys = []
    for s in s_vec
        push!(xs, s.position[1])
        push!(ys, s.position[2])
        # For now I will assume that each state only has one goal because idk how id manage it with multiple goals
        if isempty(s.goals)
            push!(goalxs, -1)
            push!(goalys, -1)
        else
            g = first(s.goals)
            push!(goalxs, g[1])
            push!(goalys, g[2])
        end
    end
    n = length(xs)
    copyto!(view(xs_buf, 1:n), Float32.(xs))
    copyto!(view(ys_buf, 1:n), Float32.(ys))
    copyto!(view(goalxs_buf, 1:n), Float32.(goalxs))
    copyto!(view(goalys_buf, 1:n), Float32.(goalys))

    t = min(n, threads)
    blocks = cld(n, t)

    CUDA.@sync @cuda threads=t blocks=blocks h_cuda_kernel!(
        view(xs_buf, 1:n), view(ys_buf, 1:n),
        view(goalxs_buf, 1:n), view(goalys_buf, 1:n),
        view(heuristics_buf, 1:n)
    ) # Using views so its fast
    return Array(view(heuristics_buf, 1:n)) # Back to cpu
end

function kbfs_cuda(start, problem, h_cuda, k_threads::Int, cuda_threads::Int)
    open = PriorityQueue{Any, Float64}()
    enqueue!(open, start, start.f)
    closed = Set()

    solution = nothing

    while !isempty(open) && solution === nothing
        succ_lists = [Vector{Any}() for _ in 1:k_threads] # make k result spots
        states = [dequeue!(open) for _ in 1:min(k_threads, length(open))] # pop k or length(open) states
        foreach(s->push!(closed, s), states) # add to closed each state
        @threads for id in eachindex(states) # Have `states` threads expand in parallel
            succ_lists[id] = [i for i in problem.getSuccessors(problem, states[id])]
        end

        succ_list = vcat(succ_lists...)
        heuristics = h_cuda(succ_list, cuda_threads)

        # update the states
        [succ_list[i].h = r for (i, r) in enumerate(heuristics)]
        [s.f = s.g + s.h for s in succ_list]

        successors = succ_list |> filter(state -> !(state in closed) && !(state in keys(open))) # Flattens and Avoids Duplicates
        for suc in successors
            if isempty(suc.goals) # Found a goal node
                if solution === nothing
                    solution = suc
                elseif suc.f < solution.f
                    solution = suc
                end
                break
            else
                push!(open, suc => suc.f) # If not a goal node, add to open
            end
        end
    end
    if solution !== nothing # Meaning the solution has been found
        return solution
    end
    return nothing
end
