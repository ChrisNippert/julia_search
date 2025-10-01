module VacuumWorld
using ..Problem

using Base.Threads

@enum Direction Up Down Left Right

const Open = Int8(0)
const Working = Int8(1)
const Done = Int8(2)

const Position = Tuple{Int32,Int32}

direction_moves = Dict(
    Up => (Int32(-1), Int32(0)),
    Down => (Int32(1), Int32(0)),
    Left => (Int32(0), Int32(-1)),
    Right => (Int32(0), Int32(1))
)

# Make IDs and have tertiary sort by ID (FIFO vs FILO)
# Split state from search node.

struct VWState <: State
    position::Position
    goals::Set{Position}
    parent_move::Union{Tuple{VWState, Direction}, Nothing}
    g::Int32
    h::Float32
    f::Float32
    successors::Vector{VWState}
    state::Atomic{Int8}
end

VWState(position, goals, parent_move, g, h, f) = VWState(position, goals, parent_move, g, h, f, VWState[], Atomic{Int8}(Open))

Base.isless(a::VWState, b::VWState) =
    a.f == b.f ? a.g < b.g : a.f < b.f

sizeof(VWState)

function Base.show(io::IO, s::VWState)
    println(io, "{Position: $(s.position)")
    println(io, "Goals: $(s.goals)")
    println(io, "g: $(s.g)")
    println(io, "h: $(s.h)")
    println(io, "f: $(s.f)")
    print("}")
end

Base.hash(s::VWState, h::UInt) = hash((s.position, s.goals), h)
Base.isequal(a::VWState, b::VWState) = a.position == b.position && a.goals == b.goals

function parse_file(filename::String)
    dim = (-1, -1)
    pos::Tuple{Int32, Int32} = (-1, -1)
    goals = Set{Position}()
    board_lines = String[]
    line_num = 1
    for line in eachline(filename)
        if line_num == 1
            nums = split(line)
            dim = (parse(Int32, nums[1]), parse(Int32, nums[2]))
        elseif line_num == 2
            # skip 'Board:'
        else
            push!(board_lines, line)
        end
        line_num += 1
    end
    # Build the matrix
    mat = zeros(Int32, dim...)
    for (i, row) in enumerate(board_lines)
        for (j, c) in enumerate(row)
            if c == '#'
                mat[i, j] = 1
            elseif c == '*'
                mat[i, j] = 3
                push!(goals, (i,j))
            elseif c == 'V'
                mat[i, j] = 2
                pos=(i, j)
            else
                mat[i, j] = 0
            end
        end
    end
    return (mat, VWState(pos, goals, nothing, 0, Inf, 0))
end

check_blocking(map::Matrix, p::Position) = map[p...] == 1
# check_bounded(map::Matrix, p::Position) = 1 ≤ p[1] ≤ size(map, 1) && 1 ≤ p[2] ≤ size(map, 2)
isValid(map::Matrix, p::Position) = !check_blocking(map,p)# && check_bounded(map,p)

function move(map::Matrix, s::VWState, direction::Direction)::VWState
    move_delta = direction_moves[direction]
    next_position = s.position .+ move_delta
    @assert isValid(map, next_position)
    goals = copy(s.goals)
    delete!(goals, next_position)
    g_val = s.g+1
    h_val = h(next_position,goals)
    ss = VWState(next_position, goals, (s, direction), g_val, h_val, g_val+h_val)
    return ss
end

function get_valid_moves(map::Matrix, s::VWState)
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

expand(problem::ProblemInstance, s::VWState)::Vector{VWState} = [move(problem.otherData(), s, d) for d in get_valid_moves(problem.otherData(), s)]

using DataStructures
# h(s::VWState)::Float32 = isempty(s.goals) ? 0.0 : maximum(sqrt(sum((s.position .- g).^2)) for g in s.goals) # euclidean distance from '
h(position, goals)::Float32 = length(goals)
h(s::VWState)::Float32 = length(s.goals)
isFinished(s::VWState)::Bool = isempty(s.goals)

function reconstruct_path(s)
    moves = Vector{Direction}()
    current_state = s
    while(current_state.parent_move !== nothing)
        push!(moves, current_state.parent_move[2])
        current_state = current_state.parent_move[1]
    end
    moves |> reverse
end

function verify_path(problem, s, path::Vector{Direction})
    currState = s
    for (i, dir) in enumerate(path)
        try
            currState = move(problem.otherData(), currState, dir) # This will assert fail inside of move
        catch
            println("Valid until step $i")
            return false
        end
    end
    return isempty(currState.goals) ? true : false
end

createVWProblem(map::Matrix) = ProblemInstance{VWState}(h, expand, ()->1, isFinished, ()->map)
export VWProblem, createVWProblem, parse_file, reconstruct_path, verify_path, Open, Working, Done
end