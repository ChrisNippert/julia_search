module Problem

abstract type State end

struct ProblemInstance{S<:State}
    h::Function
    getSuccessors::Function
    getCost::Function
    isFinished::Function
    otherData::Function
end

ProblemInstance(h, getSuccessors, getCost, isFinished) = ProblemInstance(h, getSuccessors, getCost, isFinished,  x -> nothing)

export ProblemInstance, State

end