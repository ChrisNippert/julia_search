using ..Problem
using ..VacuumWorld
using Base.Enums
using Base.Threads
using DataStructures

function kbfs(start::S, problem, k_threads::Int) where {S<:State}
    open = BinaryMinHeap{S}()
    push!(open, start)
    best_g = Dict(start => start.g)

    expansions = 0

    while !isempty(open)
        # dequeue k best states
        states = [pop!(open) for _ in 1:min(k_threads, length(open))]

        expansions += 1

        succ_lists = [Vector{Any}() for _ in 1:length(states)]

        @threads for id in eachindex(states)
            succ_lists[id] = problem.getSuccessors(problem, states[id])
        end

        successors = vcat(succ_lists...)

        finished_Candidates = [suc for suc in successors if problem.isFinished(suc)]
        non_finished = filter(suc -> !problem.isFinished(suc), successors)

        # insert non-goal successors
        for suc in non_finished
            if !haskey(best_g, suc) || suc.g < best_g[suc]
                best_g[suc] = suc.g
                push!(open, suc)
            end
        end

        # check if we can safely return a goal
        if !isempty(finished_Candidates)
            best_finished = finished_Candidates[argmin([suc for suc in finished_Candidates])]
            if isempty(open) || best_finished.f <= first(open).f
                println("$expansions")
                return best_finished
            end
        end
    end

    println("$expansions")
    println("No goal found")
    return nothing
end