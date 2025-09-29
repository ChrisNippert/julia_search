using ..Problem
using ..VacuumWorld
using Base.Enums
using DataStructures

function astar(start::State, problem)
    open = BinaryMinHeap{State}()
    closed = Dict{State, Float64}()   # best g-cost seen so far
    push!(open, start)
    closed[start] = start.g

    while !isempty(open)
        s = pop!(open)
        # If this path is stale, skip it
        if s.g > closed[s]
            continue
        end
        # Goal test
        if problem.isFinished(s)
            return s
        end

        # Expand successors
        successors = problem.getSuccessors(problem, s)
        for suc in successors
            if !haskey(closed, suc) || suc.g < closed[suc]
                closed[suc] = suc.g
                push!(open, suc)
            end
        end
    end

    return nothing
end
