using ..Problem
using ..VacuumWorld
using Base.Enums
using Base.Threads
using DataStructures

function preview_window(open::BinaryMinHeap, K::Int)
    # DataStructures.BinaryMinHeap <: AbstractHeap
    # It wraps a vector `open.valtree`
    # We can just grab the first K elements
    return [open.valtree[i] for i in 1:min(K, length(open))]
end

stopflag = Threads.Atomic{Bool}(false)
PROXY_CAPACITY = 8
hits = [Atomic{Int}(0) for _ in 1:8]
misses = [Atomic{Int}(0) for _ in 1:8]

# using spawn we can start this function on a bunch of threads if we want
function speculate(proxy_ref::Ref{Vector{State}}, problem, id)
    # each thread grabs from the proxy list and speculates
    while !stopflag[]
        # try to claim a node
        proxy_list = proxy_ref[] # immutable snapshot of the list
        for node in proxy_list
            success = atomic_cas!(node.state, Open, Working) == Open
            if success
                successors = problem.getSuccessors(problem, node)
                append!(node.successors, successors)
                node.state[] = Done
                atomic_add!(hits[id], 1)
            else
                atomic_add!(misses[id], 1)
            end
        end
        yield()
    end
end

function spbfs(start, problem, thread_count)
    stopflag[] = false

    open = BinaryMinHeap{Any}()
    best_g = Dict{Any, Float64}()   # best g-cost seen so far
    push!(open, start)
    best_g[start] = start.g
    speculated = 0
    manual = 0

    proxy_list = State[]
    proxy_ref = Ref(Vector{State}())

    workers = [@spawn speculate(proxy_ref, problem, i) for i in 1:thread_count]

    # println("Initialized $thread_count threads. Beginning Search")

    while !isempty(open)
        yield()
        s = pop!(open)

        # If this path is stale, skip it
        if s.g > best_g[s]
            continue
        end
        # Goal test
        if problem.isFinished(s)
            h = [x[] for x in hits]
            m = [x[] for x in misses]
            println("Hits: $h")
            println("Misses: $m")
            println("Speculated: $speculated")
            println("Manual: $manual")
            stopflag[] = true
            fetch.(workers)   # waits for them to exit
            return s
        end

        #####
        # Expand successors if needed
        success = atomic_cas!(s.state, Open, Working) == Open
        if success # prevents concurrent access to node
            successors = problem.getSuccessors(problem, s)
            append!(s.successors, successors)
            manual += 1
        else 
            # wait for value to be closeds
            i = 0
            while s.state[] != Done
                yield()   # let other tasks run
                # println("Yielding $i")
                # i+=1
            end
            speculated += 1
        end
        #####

        successors = s.successors
        for suc in successors
            if !haskey(best_g, suc) || suc.g < best_g[suc]
                best_g[suc] = suc.g
                push!(open, suc)

                #####
                proxy_ref[] = preview_window(open, PROXY_CAPACITY)
                #####
            end
        end
    end

    stopflag[] = true
    return nothing
end