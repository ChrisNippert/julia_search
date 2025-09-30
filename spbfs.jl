using ..Problem
using ..VacuumWorld
using Base.Enums
using Base.Threads
using DataStructures

const PROXY_CAPACITY = 8

function preview_window(open::BinaryMinHeap, K::Int)
    # It wraps a vector `open.valtree`
    # We can just grab the first K elements
    return [open.valtree[i] for i in 1:min(K, length(open))]
end

function speculate(proxy_ref::Ref{Vector{State}}, problem, stop_signal)
    # each thread grabs from the proxy list and speculates
    while !stop_signal[]
        # println("Thread $(threadid()) doing stuff")
        proxy_list = proxy_ref[] # immutable snapshot of the list
        for node in proxy_list
            success = atomic_cas!(node.state, Open, Working) == Open # Attempt to claim node
            if success
                successors = problem.getSuccessors(problem, node)
                append!(node.successors, successors)
                node.state[] = Done
            end
        end
        yield()
    end
    # println("Thread $(threadid()) stopping")
end

function spbfs(start, problem, thread_count)
    println("Starting")
    open = BinaryMinHeap{State}()
    closed_g = Dict{State, Float64}()   # best g-cost seen so far
    push!(open, start)
    closed_g[start] = start.g
    speculated = 0
    manual = 0

    #####    
    stop_signal = Atomic{Bool}(false) # set atomic to false
    proxy_ref = Ref(Vector{State}()) # Immutable upon construction and able to have concurrent reads, but not concurrent writes
    workers = [@spawn speculate(proxy_ref, problem, stop_signal) for _ in 1:thread_count-1]
    #####

    while !isempty(open)
        yield()
        s = pop!(open)
        s.g > closed_g[s] && continue
        
        # Goal test
        if problem.isFinished(s)
            println("Success!")
            stop_signal[] = true
            wait.(workers)   # waits for them to exit
            println("Speculated: $speculated")
            println("Manual: $manual")
            return s
        end

        #####
        # Expand successors if needed
        success = atomic_cas!(s.state, Open, Working) == Open
        if success # prevents concurrent access to node
            successors = problem.getSuccessors(problem, s)
            append!(s.successors, successors)
            s.state[] = Done
            manual += 1
        else 
            # wait for state to be Done
            while s.state[] != Done
                yield()
            end
            speculated += 1
        end
        #####

        # s should have successors

        successors = s.successors
        for suc in successors
            if !haskey(closed_g, suc) || suc.g < closed_g[suc]
                closed_g[suc] = suc.g
                push!(open, suc)
                #####
                # update the proxy_ref for the threads to view
                proxy_ref[] = preview_window(open, PROXY_CAPACITY)
                #####
            end
        end
    end

    stop_signal[] = true
    return nothing
end