using ..Problem
using ..VacuumWorld
using Base.Enums
using Base.Threads
using DataStructures

function astar(start, problem)
    open = BinaryMinHeap{Any}()
    best_g = Dict{Any, Float64}()   # best g-cost seen so far
    push!(open, start)
    best_g[start] = start.g

    while !isempty(open)
        s = pop!(open)
        # If this path is stale, skip it
        if s.g > best_g[s]
            continue
        end
        # Goal test
        if problem.isFinished(s)
            return s
        end

        # Expand successors
        successors = problem.getSuccessors(problem, s)
        for suc in successors
            if !haskey(best_g, suc) || suc.g < best_g[suc]
                best_g[suc] = suc.g
                push!(open, suc)
            end
        end
    end

    return nothing
end
