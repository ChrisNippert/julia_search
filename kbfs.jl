include("problem.jl")
using .Problem
include("vacuum_world.jl")
using .VacuumWorld
using Base.Enums
using Base.Threads
using DataStructures

function kbfs(start, problem, k_threads::Int)
    open = PriorityQueue{Any, Float64}()
    enqueue!(open, start, start.f)
    closed = Set()

    solution = nothing

    i = 0
    while !isempty(open) && solution === nothing
        succ_lists = [Vector{Any}() for _ in 1:k_threads] # make k result spots
        states = [dequeue!(open) for _ in 1:min(k_threads, length(open))] # pop k or length(open) states
        foreach(s->push!(closed, s), states) # add to closed each state
        @threads for id in eachindex(states) # Have `states` threads expand in parallel
            succ_lists[id] = [i for i in problem.getSuccessors(problem, states[id])]
            # Then we need to set h and f (because we can cudafy it later)
            for succ in succ_lists[id]
                succ.h = problem.h(succ)
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
                enqueue!(open, suc, suc.f) # If not a goal node, add to open
            end
        end
    end
    if solution !== nothing # Meaning the solution has been found
        return solution
    end
    println("No goal found")
    return nothing
end