using ApparentHorizonFinder
using ForwardDiff
using StaticArrays
using Test

const δ = SMatrix{3,3}(1, 0, 0, 0, 1, 0, 0, 0, 1)

function bl(x, y, z)
    M = 1
    r = sqrt(x^2 + y^2 + z^2)
    ϕ = 1 + M / 2r
    γ = ϕ^4 * δ
    return γ
end

function brill_lindquist_metric(p::SVector{3})
    T = eltype(p)
    x, y, z = p
    γ = bl(x, y, z)
    ∂xγ = ForwardDiff.derivative(x -> bl(x, y, z), x)
    ∂yγ = ForwardDiff.derivative(y -> bl(x, y, z), y)
    ∂zγ = ForwardDiff.derivative(z -> bl(x, y, z), z)
    ∂γ = SArray{Tuple{3,3,3}}(∂xγ..., ∂yγ..., ∂zγ...)
    K = SMatrix{3,3,T}(0, 0, 0, 0, 0, 0, 0, 0, 0)
    admvars = ADMVars(γ, ∂γ, K)
    return admvars
end

@testset "Brill-Lindquist" begin
    x₀ = SVector{3}(0.0, 0.0, 0.1)
    N = 8
    r = 1.0
    atol = 1.0e-8
    maxiters = 100
    find_horizon(brill_lindquist_metric, x₀, N, r, atol, maxiters)
end
