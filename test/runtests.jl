using ApparentHorizonFinder
using FastSphericalHarmonics
using ForwardDiff
using SpacetimeMetrics
using StaticArrays
using Test

const δ = SMatrix{3,3}(1, 0, 0, 0, 1, 0, 0, 0, 1)

################################################################################
# FastSphericalHarmonics complex-API sign / phase conventions used by the
# horizon finder. These tests pin the conventions down so that the
# complex spin-0 → spin-1 → spin-2 path used by `expansion` stays correct
# under future FSH releases.
#
# Summary of conventions, derived from FSH's source plus the tests below:
#
#  Spin-0 normalisation
#  ────────────────────
#  `spinsph_evaluate(C, 0)` returns Σ_{lm} C_{lm} Y_{lm}(θ,φ) where the
#  scalar spherical harmonics use the convention WITHOUT Condon-Shortley
#  phase. In particular Y_{1,0}(θ,φ) = √(3/4π) cosθ.
#
#  ð on a real scalar (textbook reference)
#  ───────────────────────────────────────
#  The TEXTBOOK Goldberg-Penrose action on a spin-0 scalar f is
#       (ð f)(θ,φ) = −[∂_θ + (i/sinθ) ∂_φ] f.
#  FSH's complex spin-1 GRID value of `spinsph_evaluate(spinsph_eth(C,0),1)`
#  reproduces this TEXTBOOK action only for modes with m ≥ 0; for m < 0
#  the FSH spin-1 grid value is the textbook value with a SIGN flip.
#  (This is the same flip that the real-array path applies internally; the
#  complex path does NOT apply it, leaving FSH's internal ₁Y_{l,m}
#  representation directly visible.) Because the gradient enters the
#  expansion as a vector (v_θ̂, v_φ̂), the cleanest workaround is to leave
#  the complex spin-1 grid as-is and use the existing real-array
#  ðh helper for the gradient (it already absorbs the flip).
#
#  ð̄ð on spin-0 is the Laplacian on the unit sphere with eigenvalue −l(l+1),
#  so Δh can be computed in spin-0 coefficient space by multiplying h_{lm}
#  by −l(l+1).
#
#  ð² on a real scalar
#  ───────────────────
#  Chaining `spinsph_eth(_, 0)` then `spinsph_eth(_, 1)` on the complex
#  spin-0 coefficients gives spin-2 coefficients whose grid evaluation
#  EQUALS the textbook spin-2 grid quantity
#       (ð² h)(θ,φ) = (H_{θ̂θ̂} − H_{φ̂φ̂}) + 2 i H_{θ̂φ̂}
#  for every m EXCEPT m = −1, where it has the opposite sign. The fix is
#  to multiply column 2 of the spin-2 coefficient array (which stores the
#  m = −1 mode for every l) by −1 before calling `spinsph_evaluate(_, 2)`.
#  (No other m needs a correction; this was checked across l up to 8.)
#
#  Combining the trace identity Δh = H_{θ̂θ̂} + H_{φ̂φ̂} with the two pieces
#  of the corrected ð²h gives the orthonormal-frame Hessian:
#       Hθθ = (Δh + Re ð²h) / 2
#       Hφφ = (Δh − Re ð²h) / 2
#       Hθφ =  Im ð²h / 2

@testset "FSH complex spin-0 normalisation: Y_{1,0}" begin
    N = 16
    θ, φ = sph_points(N)
    C = zeros(ComplexF64, N, 2N - 1)
    C[spinsph_mode(0, 1, 0)] = 1.0
    F = spinsph_evaluate(C, 0)
    expected = [sqrt(3 / (4π)) * cos(θ[i]) + 0im for i in 1:length(θ), j in 1:length(φ)]
    @test maximum(abs.(F .- expected)) < 1e-13
end

@testset "ð on Y_{1,0} (m=0): −∂_θ Y_{1,0} = √(3/4π) sinθ on grid" begin
    # For m = 0 modes the FSH complex spin-1 grid value coincides with the
    # textbook action of ð on the scalar; no sign correction needed.
    N = 16
    θ, φ = sph_points(N)
    C = zeros(ComplexF64, N, 2N - 1)
    C[spinsph_mode(0, 1, 0)] = 1.0
    ðF = spinsph_evaluate(spinsph_eth(C, 0), 1)
    expected = [sqrt(3 / (4π)) * sin(θ[i]) + 0im for i in 1:length(θ), j in 1:length(φ)]
    @test maximum(abs.(ðF .- expected)) < 1e-13
end

@testset "ð̄ð on spin-0 has eigenvalue −l(l+1)" begin
    N = 8
    C = zeros(ComplexF64, N, 2N - 1)
    for (l, m) in ((1, 0), (2, +1), (2, -1), (3, +2), (3, -2), (4, 0))
        C .= 0
        C[spinsph_mode(0, l, m)] = 1.0
        Δlm = spinsph_ethbar(spinsph_eth(C, 0), 1)
        @test isapprox(Δlm[spinsph_mode(0, l, m)], -l * (l + 1); atol=1e-12)
    end
end

@testset "ð² on Y_{2,0}: closed-form match (no m = −1 fix needed)" begin
    # f = √(5/16π) (3cos²θ − 1); m=0 so the (i/sinθ)∂_φ pieces vanish and
    # the spin-2 grid value reduces to ∂²_θ f − cotθ ∂_θ f.
    N = 16
    θ, φ = sph_points(N)
    C = zeros(ComplexF64, N, 2N - 1)
    C[spinsph_mode(0, 2, 0)] = 1.0
    ð2F = spinsph_evaluate(spinsph_eth(spinsph_eth(C, 0), 1), 2)
    f(θ) = sqrt(5 / (16π)) * (3 * cos(θ)^2 - 1)
    ∂θf(θ) = ForwardDiff.derivative(f, θ)
    ∂2θf(θ) = ForwardDiff.derivative(∂θf, θ)
    expected = [(∂2θf(θ[i]) - cot(θ[i]) * ∂θf(θ[i])) + 0im for i in 1:length(θ), j in 1:length(φ)]
    @test maximum(abs.(ð2F .- expected)) < 1e-12
end

@testset "ð² m = −1 fix: column-2 sign flip recovers textbook spin-2 grid" begin
    # For a single Y_{l,-1} input, the eth-chain spin-2 grid value differs
    # from the textbook (Hθθ − Hφφ) + 2i Hθφ by exactly −1. Multiplying
    # column 2 of the spin-2 coefficient array by −1 restores agreement.
    # We verify both the necessity (without the fix it fails) and the
    # sufficiency (with the fix it matches to machine precision) on Y_{2,-1}.
    function Y2m1(θ, φ)
        +sqrt(15 / (8π)) * sin(θ) * cos(θ) * cis(-φ)
    end
    N = 32
    θg, φg = sph_points(N)
    C = zeros(ComplexF64, N, 2N - 1)
    C[spinsph_mode(0, 2, -1)] = 1.0
    Cs2 = spinsph_eth(spinsph_eth(C, 0), 1)
    no_fix = spinsph_evaluate(copy(Cs2), 2)
    Cs2[:, 2] .*= -1
    fixed = spinsph_evaluate(Cs2, 2)
    i, j = 8, 5
    θ, φ = θg[i], φg[j]
    ∂θY  = ForwardDiff.derivative(t -> Y2m1(t, φ), θ)
    ∂φY  = ForwardDiff.derivative(p -> Y2m1(θ, p), φ)
    ∂2θY = ForwardDiff.derivative(t -> ForwardDiff.derivative(t -> Y2m1(t, φ), t), θ)
    ∂2φY = ForwardDiff.derivative(p -> ForwardDiff.derivative(p -> Y2m1(θ, p), p), φ)
    ∂θ∂φY = ForwardDiff.derivative(τ -> ForwardDiff.derivative(p -> Y2m1(τ, p), φ), θ)
    sθ, cθ = sincos(θ)
    Hθθ = ∂2θY
    Hφφ = ∂2φY / sθ^2 + (cθ / sθ) * ∂θY
    Hθφ = ∂θ∂φY / sθ - (cθ / sθ) * (∂φY / sθ)
    textbook = (Hθθ - Hφφ) + 2im * Hθφ
    @test isapprox(no_fix[i, j], -textbook; atol=1e-12)
    @test isapprox(fixed[i, j], textbook; atol=1e-12)
end

@testset "Hessian recovery from ð, ð² (with m=-1 fix): matches ForwardDiff" begin
    # End-to-end check: take a smooth non-axisymmetric scalar h(θ,φ) (the
    # rotated oblate-spheroid radius, which has rich m ≠ 0 content), and
    # confirm that the recovery formulas
    #   Hθθ = (Δh + Re ð²h) / 2,
    #   Hφφ = (Δh − Re ð²h) / 2,
    #   Hθφ =  Im ð²h / 2
    # — with column-2 sign flip on the spin-2 coefficients — match analytic
    # ForwardDiff differentiation everywhere on the grid.
    M, a, α = 1.0, 0.8, π / 6
    r_plus = M + sqrt(M^2 - a^2)
    sα, cα = sincos(α)
    h(θ, φ) = begin
        cθold = sα * sin(θ) * cos(φ) + cα * cos(θ)
        sθold² = 1 - cθold^2
        1 / sqrt(sθold² / (r_plus^2 + a^2) + cθold^2 / r_plus^2)
    end

    N = 64
    θgrid, φgrid = sph_points(N)
    h_grid = [h(θgrid[i], φgrid[j]) + 0im for i in 1:length(θgrid), j in 1:length(φgrid)]
    hlm_c = spinsph_transform(h_grid, 0)

    # Δh via the spin-0 eigenvalue
    Δhlm = copy(hlm_c)
    for l in 0:N-1, m in -l:l
        Δhlm[spinsph_mode(0, l, m)] *= -l * (l + 1)
    end
    Δh_g = real.(spinsph_evaluate(Δhlm, 0))

    # ð²h via the complex eth chain + m = −1 column fix
    ð2h_c = spinsph_eth(spinsph_eth(hlm_c, 0), 1)
    ð2h_c[:, 2] .*= -1
    ð2h_g = spinsph_evaluate(ð2h_c, 2)

    max_err_θθ = 0.0
    max_err_φφ = 0.0
    max_err_θφ = 0.0
    for j in 1:size(h_grid, 2), i in 1:size(h_grid, 1)
        θ, φ = θgrid[i], φgrid[j]
        ∂θh = ForwardDiff.derivative(t -> h(t, φ), θ)
        ∂φh = ForwardDiff.derivative(p -> h(θ, p), φ)
        ∂2θh = ForwardDiff.derivative(t -> ForwardDiff.derivative(t -> h(t, φ), t), θ)
        ∂2φh = ForwardDiff.derivative(p -> ForwardDiff.derivative(p -> h(θ, p), p), φ)
        ∂θ∂φh = ForwardDiff.derivative(τ -> ForwardDiff.derivative(p -> h(τ, p), φ), θ)
        sθ, cθ = sincos(θ)
        Hθθ_an = ∂2θh
        Hφφ_an = ∂2φh / sθ^2 + (cθ / sθ) * ∂θh
        Hθφ_an = ∂θ∂φh / sθ - (cθ / sθ) * (∂φh / sθ)
        Hθθ_sp = (Δh_g[i, j] + real(ð2h_g[i, j])) / 2
        Hφφ_sp = (Δh_g[i, j] - real(ð2h_g[i, j])) / 2
        Hθφ_sp = imag(ð2h_g[i, j]) / 2
        max_err_θθ = max(max_err_θθ, abs(Hθθ_sp - Hθθ_an))
        max_err_φφ = max(max_err_φφ, abs(Hφφ_sp - Hφφ_an))
        max_err_θφ = max(max_err_θφ, abs(Hθφ_sp - Hθφ_an))
    end
    @test max_err_θθ < 1e-9
    @test max_err_φφ < 1e-9
    @test max_err_θφ < 1e-9
end

################################################################################

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

# Kerr-Schild with M=1, a=0.6, spin axis tilted 30° from ẑ via SpacetimeMetrics.rotate.
function rotated_kerr_schild_metric(p::SVector{3})
    x, y, z = p
    M = 1.0
    a = 0.8
    α = π / 6
    ks = rotate(KerrSchild(M, a), 0, α, 0)   # ZYZ Euler angles: pure R_y(α)
    t = 0
    g, ∂g = dmetric(ks, SVector{4}(t, x, y, z))
    K = ExtrinsicCurvature(ks, SVector{4}(t, x, y, z))
    γ = SMatrix{3,3}(g[i, j] for i in 2:4, j in 2:4)
    ∂γ = SArray{Tuple{3,3,3}}(∂g[i, j, k] for i in 2:4, j in 2:4, k in 2:4)
    admvars = ADMVars(γ, ∂γ, K)
    return admvars
end

@testset "Rotated Kerr-Schild horizon expansion" begin
    # The Kerr horizon stays an oblate spheroid centred at the origin; its
    # short axis is tilted 30° from ẑ. Each radial ray from the origin in the
    # rotated frame is mapped back to the un-rotated frame by R_y(-α), and
    # then meets (x²+y²)/(r_+²+a²) + z²/r_+² = 1.
    M = 1.0
    a = 0.8
    α = π / 6
    r_plus = M + sqrt(M^2 - a^2)
    sα, cα = sincos(α)
    N = 16
    θ, φ = horizon_grid(N)
    r_horizon = Matrix{Float64}(undef, length(θ), length(φ))
    for j in 1:length(φ), i in 1:length(θ)
        # cos θ_old = (R_y(-α) r̂)_z = sin α · sin θ cos φ + cos α · cos θ
        cθold = sα * sin(θ[i]) * cos(φ[j]) + cα * cos(θ[i])
        sθold² = 1 - cθold^2
        r_horizon[i, j] = 1 / sqrt(sθold² / (r_plus^2 + a^2) + cθold^2 / r_plus^2)
    end
    hlm = horizon_shape(r_horizon)
    atol = 1.0e-3
    success, iters, origin, hlm = find_horizon(rotated_kerr_schild_metric, SVector{3}(0.0, 0.0, 0.0), hlm, atol, 0)
    @test success
    @test iters == 0
end

@testset "Rotated Kerr-Schild" begin
    # Find the same tilted horizon starting from a sphere outside it.
    x₀ = SVector{3}(0.0, 0.0, 0.1)
    N = 16
    r = 2.5
    atol = 1.0e-8
    maxiters = 200
    success, iters, origin, hlm = find_horizon(rotated_kerr_schild_metric, x₀, N, r, atol, maxiters)
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
