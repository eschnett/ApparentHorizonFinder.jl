using ApparentHorizonFinder
using ForwardDiff
using SpacetimeMetrics
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

function kerr_schild_metric(p::SVector{3})
    x, y, z = p
    M = 1.0
    a = 0.6
    ks = KerrSchild(M, a)
    t = 0
    g, ∂g = dmetric(ks, SVector{4}(t, x, y, z))
    K = ExtrinsicCurvature(ks, SVector{4}(t, x, y, z))
    γ = SMatrix{3,3}(g[i,j] for i in 2:4, j in 2:4)
    ∂γ = SArray{Tuple{3,3,3}}(∂g[i,j,k] for i in 2:4, j in 2:4, k in 2:4)
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

@testset "Kerr-Schild" begin
    x₀ = SVector{3}(0.0, 0.0, 0.1)
    N = 8
    r = 2.0
    atol = 1.0e-8
    maxiters = 200
    find_horizon(kerr_schild_metric, x₀, N, r, atol, maxiters)
end
