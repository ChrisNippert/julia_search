using Base.Enums
using Base.Threads
using CUDA

@enum Direction Up Down Left Right

direction_moves = Dict(
    Up => CuArray[-1.0, 0.0],
    Down => CuArray[1.0, 0.0],
    Left => CuArray[0.0, -1.0],
    Right => CuArray[0.0, 1.0]
)

N = 65536

map::CuArray{Float32}

xStates::CuArray{Float32} = CUDA.zeros(Float32, 65536)
yStates::CuArray{Float32} = CUDA.zeros(Float32, 65536)
gxy::CuArray{Float32} = CUDA.zeros(Float32, 2) # Should just contain 1:2 for x,y

gStates::CuArray{Float32} = CUDA.zeros(Float32, 65536)
hStates::CuArray{Float32} = CUDA.zeros(Float32, 65536)
fStates::CuArray{Float32} = CUDA.zeros(Float32, 65536)

mutable struct GameState
    id::Int # index into the CuArrays
    parent_move::Union{Tuple{GameState, Direction}, Nothing}
end

index = 1

function parse_file(filename::String)

    dim = (-1, -1)
    board_lines = String[]
    line_num = 1
    for line in eachline(filename)
        if line_num == 1
            nums = split(line)
            dim = (parse(Int, nums[1]), parse(Int, nums[2]))
        elseif line_num == 2
            # skip 'Board:'
        else
            push!(board_lines, line)
        end
        line_num += 1
    end
    # Build the matrix
    mat = zeros(Int, dim...)
    for (i, row) in enumerate(board_lines)
        for (j, c) in enumerate(row)
            if c == '#'
                mat[i, j] = 1
            elseif c == '*'
                gxy[1] = Float32(i)
                gxy[2] = Float32(j)
                mat[i, j] = 0
            elseif c == 'V'
                xStates[index] = Float32(i)
                yStates[index] = Float32(j)
                mat[i, j] = 0
            else
                mat[i, j] = 0
            end
        end
    end

    return (mat, GameState(index += 1, nothing))
end

check_blocking(map::Matrix, p::Position) = map[p...] == 1
# check_bounded(map::Matrix, p::Position) = 1 ≤ p[1] ≤ size(map, 1) && 1 ≤ p[2] ≤ size(map, 2)
isValid(map::Matrix, p::Position) = !check_blocking(map,p)# && check_bounded(map,p)

function move(map::Matrix, s::GameState, direction::Direction)::GameState
    move_delta = direction_moves[direction]
    next_position = s.position .+ move_delta
    @assert isValid(map, next_position)
    goals = copy(s.goals)
    delete!(goals, next_position)
    ss = GameState(next_position, goals, (s, direction), s.g+1, -1, -1)
    return ss
end

function get_valid_moves(map::Matrix, s::GameState)
    valid_moves = Direction[]
    for d in instances(Direction)
        move_delta = direction_moves[d]
        next_position = s.position .+ move_delta
        if isValid(map, next_position)
            @inbounds push!(valid_moves, d)
        end
    end
    valid_moves
end

expand(m::Matrix, s::GameState)::Vector{GameState} = [move(m, s, d) for d in get_valid_moves(m, s)]

using DataStructures
h(s::GameState)::Float64 = isempty(s.goals) ? 0.0 : sum(sqrt(sum((s.position .- g).^2)) for g in s.goals) # euclidean distance from 

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

x_vec = CuArray([0.0])
y_vec = CuArray([0.0])

gx_vec = CuArray([3.0])
gy_vec = CuArray([4.0])
ret = CUDA.zeros(Float32, length(x_vec))

N = length(x_vec)
threads = min(256, N)
blocks  = cld(N, threads)
CUDA.@sync @cuda threads=threads blocks=blocks h_cuda_kernel!(x_vec, y_vec, gx_vec, gy_vec, ret)
ret

max_n = 256 # largest number of states you expect
xs_buf = CUDA.zeros(Float32, max_n)
ys_buf = CUDA.zeros(Float32, max_n)
goalxs_buf = CUDA.zeros(Float32, max_n)
goalys_buf = CUDA.zeros(Float32, max_n)
heuristics_buf = CUDA.zeros(Float32, max_n)

function h_cuda(s_vec::Vector{GameState}, threads::Int64)
    # for each game state, extract the x and y and add then to the arrays. For the goals, extract the x,y into their matrices
    # convert to cuda arrays
    xs = []
    ys = []
    goalxs = []
    goalys = []
    for s::GameState in s_vec
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

m1 = [0 0 1;0 1 1;0 0 0]
s_1 = GameState((2,2),Set([(3,3)]),nothing, 0, -1, -1)
s_2 = GameState((1,2),Set([(3,3)]),nothing, 0, -1, -1)

CUDA.@profile h_cuda([s_1, s_2], 4)
@time h(s_1)
@time h(s_2)

function reconstruct_path(s::GameState)
    moves = Vector{Direction}()
    current_state = s
    while(current_state.parent_move !== nothing)
        push!(moves, current_state.parent_move[2])
        current_state = current_state.parent_move[1]
    end
    moves
end

function astar(map::Matrix, start::GameState, h)
    open = PriorityQueue{GameState, Float64}()
    enqueue!(open, start, start.f)
    closed = Set()

    while !isempty(open) && !isempty(peek(open)[1].goals) # if there is stuff left and the goals have not reached 0
        s = dequeue!(open)
        push!(closed, s)
        successors = [i for i in expand(map, s) if !(i in closed) && !(i in keys(open))]
        [s.h = h(s) for s in successors]
        [s.f = s.h+s.g for s in successors]
        foreach(suc -> enqueue!(open, suc, suc.f), successors)
    end
    if !isempty(open)
        return peek(open)[1]
    end
    return nothing
end

function kbfs(map::Matrix, start::GameState, h, k_threads::Int)
    open = PriorityQueue{GameState, Float64}()
    push!(open, start => start.f)
    closed = Set()

    solution = nothing

    i = 0
    while !isempty(open) && solution === nothing
        succ_lists = [Vector{GameState}() for _ in 1:k_threads] # make k result spots
        states = [dequeue!(open) for _ in 1:min(k_threads, length(open))] # pop k or length(open) states
        foreach(s->push!(closed, s), states) # add to closed each state
        @threads for id in 1:length(states) # Have `states` threads expand in parallel
            succ_lists[id] = [i for i in expand(map, states[id])]
            # Then we need to set h and f (because we can cudafy it later)
            for succ in succ_lists[id]
                succ.h = h(succ)
                succ.f = succ.h + succ.g
            end
        end
        successors = vcat(succ_lists...) |> filter(state -> !(state in closed) && !(state in keys(open))) # Flattens and Avoids Duplicates
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
    println("No goal found")
    return nothing
end

function kbfs_cuda(map::Matrix, start::GameState, h_cuda, k_threads::Int, cuda_threads::Int)
    open = PriorityQueue{GameState, Float64}()
    push!(open, start => start.f)
    closed = Set()

    solution = nothing

    while !isempty(open) && solution === nothing
        succ_lists = [Vector{GameState}() for _ in 1:k_threads] # make k result spots
        states = [dequeue!(open) for _ in 1:min(k_threads, length(open))] # pop k or length(open) states
        foreach(s->push!(closed, s), states) # add to closed each state
        @threads for id in 1:length(states) # Have `states` threads expand in parallel
            succ_lists[id] = [i for i in expand(map, states[id])]
        end
        succ_list = vcat(succ_lists...)
        #its cuda time.
        @profile heuristics = h_cuda(succ_list, cuda_threads)
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

using BenchmarkTools

# # file="data/vacuum_instances/maze/100/100/0.1/10/1"
# dir = "data/vacuum_instances/uniform/100/100/0.4/50"
# files = readdir(dir)
# files = files[1:end-1]
# full_paths = [joinpath(dir, f) for f in files]
# results = Dict()
# for f in full_paths
#     s::GameState = parse_file(f)
#     # @time res = astar(s, h)
#     @time res2 = kbfs(s, h, Threads.nthreads())
#     # path = reconstruct_path(res)
#     # println(path)
# end

using Profile

file = "100/0./1/5"
m::Matrix, s::GameState = parse_file(file)
@time kbfs(m, s, h, 16)
CUDA.@profile kbfs_cuda(m, s, h_cuda, 64, 128)
@time astar(m, s, h)

Profile.print()