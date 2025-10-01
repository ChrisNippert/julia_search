using ..Problem
using ..VacuumWorld
using Base.Enums
using Base.Threads
using DataStructures

function pastar(start, problem, thread_count)
    println("Starting")
    l = ReentrantLock()

    open = BinaryMinHeap{State}()
    closed = Dict{State, Float64}()   # best g-cost seen so far
    push!(open, start)
    closed[start] = start.g

    doStop = Atomic{Bool}(false)
    expansions = 0

    best = nothing

    @threads for _ in 1:thread_count-1
        while !doStop[]
            s = lock(l) do # Apparently the "do" is actually an anonymous function
                if isempty(open)
                    return nothing
                end
                expansions += 1
                pop!(open)
            end
            if s === nothing
                continue
            end
            # If this path is stale, skip it
            if s.g > closed[s]
                continue
            end
            # Goal test
            if problem.isFinished(s)
                lock(l) do
                    if best === nothing
                        best = s
                    elseif s.g < best.g
                        best = s
                    end
                    doStop[] = true
                end
            end

            # Expand successors
            successors = problem.getSuccessors(problem, s)
            for suc in successors
                lock(l) do
                    if !haskey(closed, suc) || suc.g < closed[suc]
                        closed[suc] = suc.g
                        push!(open, suc)
                    end
                end
            end
        end
    end

    println("$expansions")
    return best
end
