module Problem

abstract type State end

struct ProblemInstance{S<:State}
    h::Function
    getSuccessors::Function
    getCost::Function
    otherData::Function
end

ProblemInstance(h, getSuccessors, getCost) = ProblemInstance(h, getSuccessors, getCost, x -> nothing)

export ProblemInstance, State

end