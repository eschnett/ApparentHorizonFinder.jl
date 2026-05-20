using ApparentHorizonFinder
using ForwardDiff
using SpacetimeMetrics
using StaticArrays
using Test

const δ = SMatrix{3,3}(1, 0, 0, 0, 1, 0, 0, 0, 1)

################################################################################

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
    # Horizon at r=1/2
    success, iters, origin, hlm = find_horizon(brill_lindquist_metric, x₀, N, r, atol, maxiters)
    @test success
end

################################################################################

function kerr_schild_metric(p::SVector{3})
    x, y, z = p
    M = 1.0
    a = 0.8
    ks = KerrSchild(M, a)
    t = 0
    g, ∂g = dmetric(ks, SVector{4}(t, x, y, z))
    K = ExtrinsicCurvature(ks, SVector{4}(t, x, y, z))
    γ = SMatrix{3,3}(g[i, j] for i in 2:4, j in 2:4)
    ∂γ = SArray{Tuple{3,3,3}}(∂g[i, j, k] for i in 2:4, j in 2:4, k in 2:4)
    admvars = ADMVars(γ, ∂γ, K)
    return admvars
end

@testset "Kerr-Schild horizon expansion" begin
    # Seed the finder at the exact analytic Kerr horizon (oblate spheroid in
    # Kerr-Schild Cartesian) with the BH centre as origin; the expansion H
    # should vanish to within spectral discretisation error.
    M = 1.0
    a = 0.8
    r_plus = M + sqrt(M^2 - a^2)
    N = 16
    θ, φ = horizon_grid(N)
    r_horizon = [sqrt(r_plus^2 * (r_plus^2 + a^2) / (r_plus^2 + a^2 * cos(θ[i])^2))
                 for i in 1:length(θ), j in 1:length(φ)]
    hlm = horizon_shape(r_horizon)
    atol = 1.0e-3
    success, iters, origin, hlm = find_horizon(kerr_schild_metric, SVector{3}(0.0, 0.0, 0.0), hlm, atol, 0)
    @test success
    @test iters == 0
end

@testset "Kerr-Schild" begin
    x₀ = SVector{3}(0.0, 0.0, 0.1)
    N = 8
    r = 2.0
    atol = 1.0e-8
    maxiters = 100
    # Horizon at r=2, or inside this sphere for a>0
    success, iters, origin, hlm = find_horizon(kerr_schild_metric, x₀, N, r, atol, maxiters)
    @test success
end

################################################################################

function harmonic_metric(p::SVector{3})
    x, y, z = p
    M = 1.0
    a = 0.8
    ha = Harmonic(M, a)
    t = 0
    g, ∂g = dmetric(ha, SVector{4}(t, x, y, z))
    K = ExtrinsicCurvature(ha, SVector{4}(t, x, y, z))
    γ = SMatrix{3,3}(g[i, j] for i in 2:4, j in 2:4)
    ∂γ = SArray{Tuple{3,3,3}}(∂g[i, j, k] for i in 2:4, j in 2:4, k in 2:4)
    admvars = ADMVars(γ, ∂γ, K)
    return admvars
end

@testset "Harmonic horizon expansion" begin
    # The horizon is the oblate spheroid
    # (x²+y²)/M² + z²/(M²-a²) = 1, parametrised in (θ,φ) from the BH centre by
    # h(θ)² = M²(M²-a²) / (M² - a² sin²θ).
    M = 1.0
    a = 0.8
    N = 32
    θ, φ = horizon_grid(N)
    r_horizon = [sqrt(M^2 * (M^2 - a^2) / (M^2 - a^2 * sin(θ[i])^2))
                 for i in 1:length(θ), j in 1:length(φ)]
    hlm = horizon_shape(r_horizon)
    atol = 1.0e-3
    success, iters, origin, hlm = find_horizon(harmonic_metric, SVector{3}(0.0, 0.0, 0.0), hlm, atol, 0)
    @test success
    @test iters == 0
end

@testset "Harmonic" begin
    x₀ = SVector{3}(0.0, 0.0, 0.1)
    N = 32
    r = 2.0
    atol = 1.0e-8
    maxiters = 100
    # Horizon at r=1, or inside this sphere for a>0
    success, iters, origin, hlm = find_horizon(harmonic_metric, x₀, N, r, atol, maxiters)
    @test success
end
