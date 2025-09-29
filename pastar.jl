include("problem.jl")
using .Problem
include("vacuum_world.jl")
using .VacuumWorld
using Base.Enums
using Base.Threads
using DataStructures

function astar(start, problem)
    lock = ReentrantLock()
    open = PriorityQueue{Any, Float64}()
    enqueue!(open, start, start.f)
    closed = Set()

    while !isempty(open) && !isempty(peek(open)[1].goals) # if there is stuff left and the goals have not reached 0
        s = dequeue!(open)
        push!(closed, s)
        successors = [i for i in problem.getSuccessors(problem, s) if !(i in closed) && !(i in keys(open))]
        (s.h = h(s) for s in successors)
        (s.f = s.h+s.g for s in successors)
        foreach(suc -> enqueue!(open, suc, suc.f), successors)
    end
    if !isempty(open)
        return peek(open)[1]
    end
    return nothing
end