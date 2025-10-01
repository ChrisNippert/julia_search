using Base.Threads

mutable struct AtomicWrapper{T}
    @atomic x::T
end

read(avec::AtomicWrapper) = @atomic :acquire avec.x
write(avec::AtomicWrapper, val) = @atomic :release avec.x = val

function f2()
    a = AtomicWrapper{Vector{Int}}(Int[])
    write(a, [1, 2, 3])
    stop = Atomic{Bool}(false)              # termination flag

    thread = @spawn begin 
        i = 0
        while !stop[]
            if i % 10 == 0
                write(a, Int[])
            else
                write(a, [read(a)..., i])
            end
            i += 1
        end
    end

    @threads for _ in 1:10000
        println(read(a))
    end
    stop[] = true
    wait(thread)
end

function f()
    a = Ref{Vector{Int}}(Int[]) # Think of this as a pointer that can have concurrent reads but not concurrent writes, to a vector of Ints
    val = [1, 2, 3]
    a[] = val # Setting the value of a to [1,2,3]
    worker = @spawn begin # Writes occurring while reads occurring
        i = 0
        # Since this is the only thread writing, no concurrent writes
        while true
            if i % 10 == 0
                a[] = []
            else
                a[] = [a[]..., i]
            end
            i += 1
        end
    end
    @threads for _ in 1:100
        println(a[]) # Concurrent Reads
    end
    wait(worker)
end

for _ in 1:10
    f2()
end