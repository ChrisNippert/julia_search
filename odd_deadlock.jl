using Base.Threads

function doStuff(proxy_ref::Ref{Vector{Int}}, stop_signal)
    while !stop_signal[]
        proxy_list = proxy_ref[] # immutable snapshot of the list
        for _ in proxy_list
            sleep(0.005)
        end
        yield()
    end
end

function algorithm(thread_count)
    stop_signal = Atomic{Bool}(false) # set atomic to false
    proxy_ref = Ref(Vector{Int}()) # Immutable upon construction and able to have concurrent reads, but not concurrent writes
    workers = [@spawn doStuff(proxy_ref, stop_signal) for _ in 1:thread_count-1]

    time = 1
    done = 2.5*10^6

    while true
        yield()
        if time >= done
            println("Success!")
            stop_signal[] = true
            wait.(workers)   # waits for them to exit
            return true
        end
        proxy_ref =  Ref(rand(Int, 25))
        time += 1
    end

    stop_signal[] = true
    wait.(workers)
    return nothing
end

using BenchmarkTools
@btime algorithm(8)