using Base.Threads

function doStuff(stop_signal, proxy_list)
    while !stop_signal[]
        proxy_list = @atomic proxy_list # immutable snapshot of the list
        for _ in 1:25
            sleep(0.005)
        end
        yield()
    end
end

function algorithm(thread_count)
    stop_signal = Atomic{Bool}(false) # set atomic to false
    @atomic proxy_ref = Vector{Int}() # Immutable upon construction and able to have concurrent reads, but not concurrent writes
    workers = [@spawn doStuff(stop_signal) for _ in 1:thread_count-1]

    time = 1
    done = 2.5*10^6

    while true
        yield()
        if time >= done
            println(stderr, "Success!")
            stop_signal[] = true
            wait.(workers)   # waits for them to exit
            return true
        end
        # proxy_ref[] =  rand(Int, 25)
        time += 1
    end

    stop_signal[] = true
    wait.(workers)
    return nothing
end

using BenchmarkTools
@btime algorithm(8)