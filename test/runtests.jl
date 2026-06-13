using AbstractSphericalHarmonics
using ApparentHorizonFinder
using ForwardDiff
using LinearAlgebra: dot
using SpacetimeMetrics
using StaticArrays
using Test
import SSHT                     # second backend (DriscollHealyGrid)

const δ = SMatrix{3,3}(1, 0, 0, 0, 1, 0, 0, 0, 1)

################################################################################
# Spherical-harmonic conventions used by the horizon finder, via the
# AbstractSphericalHarmonics interface (canonical/Wikipedia conventions; the
# backend-specific phase differences are reconciled inside ASH's extensions).
# These tests pin down the spin-0 → spin-1 → spin-2 eth chain that
# `expansion` relies on:
#
#   ðf            = −[∂_θ + (i/sinθ) ∂_φ] f                  (spin 0 → 1)
#   ð̄ð f          = Δf, eigenvalue −l(l+1)
#   ð²f           = (H_θ̂θ̂ − H_φ̂φ̂) + 2i H_θ̂φ̂                  (spin 0 → 2)
# with the orthonormal-frame Hessian
#   H_θ̂θ̂ = ∂²_θ f,
#   H_φ̂φ̂ = (1/sin²θ) ∂²_φ f + cotθ ∂_θ f,
#   H_θ̂φ̂ = (1/sinθ) ∂_θ∂_φ f − (cotθ/sinθ) ∂_φ f,
# so that Δf = H_θ̂θ̂ + H_φ̂φ̂ recovers the full Hessian from ðf, ð²f.
# They run on both backends.

grids16 = (EquiangularGrid(15), DriscollHealyGrid(15))

@testset "Y_{1,0} normalisation: $(typeof(grid))" for grid in grids16
    C = zeros(ComplexF64, ash_nmodes(grid))
    C[ash_mode_index(grid, 0, 1, 0)] = 1.0
    F = ash_evaluate(grid, C, 0)
    expected = [sqrt(3 / (4π)) * cos(ash_point_coord(grid, ij)[1]) + 0im for ij in CartesianIndices(ash_grid_size(grid))]
    @test maximum(abs.(F .- expected)) < 1e-13
end

@testset "ð on Y_{1,0}: −∂_θ Y_{1,0} = √(3/4π) sinθ: $(typeof(grid))" for grid in grids16
    C = zeros(ComplexF64, ash_nmodes(grid))
    C[ash_mode_index(grid, 0, 1, 0)] = 1.0
    ðF = ash_evaluate(grid, ash_eth(grid, C, 0), 1)
    expected = [sqrt(3 / (4π)) * sin(ash_point_coord(grid, ij)[1]) + 0im for ij in CartesianIndices(ash_grid_size(grid))]
    @test maximum(abs.(ðF .- expected)) < 1e-13
end

@testset "ð̄ð on spin-0 has eigenvalue −l(l+1)" begin
    grid = EquiangularGrid(7)
    C = zeros(ComplexF64, ash_nmodes(grid))
    for (l, m) in ((1, 0), (2, +1), (2, -1), (3, +2), (3, -2), (4, 0))
        C .= 0
        C[ash_mode_index(grid, 0, l, m)] = 1.0
        Δlm = ash_ethbar(grid, ash_eth(grid, C, 0), 1)
        @test isapprox(Δlm[ash_mode_index(grid, 0, l, m)], -l * (l + 1); atol=1e-12)
    end
end

@testset "ð² on Y_{2,0}: closed-form match: $(typeof(grid))" for grid in grids16
    # f = √(5/16π) (3cos²θ − 1); m=0 so the (i/sinθ)∂_φ pieces vanish and
    # the spin-2 grid value reduces to ∂²_θ f − cotθ ∂_θ f.
    C = zeros(ComplexF64, ash_nmodes(grid))
    C[ash_mode_index(grid, 0, 2, 0)] = 1.0
    ð2F = ash_evaluate(grid, ash_eth(grid, ash_eth(grid, C, 0), 1), 2)
    f(θ) = sqrt(5 / (16π)) * (3 * cos(θ)^2 - 1)
    ∂θf(θ) = ForwardDiff.derivative(f, θ)
    ∂2θf(θ) = ForwardDiff.derivative(∂θf, θ)
    expected = [(∂2θf(θ) - cot(θ) * ∂θf(θ)) + 0im for (θ, _) in (ash_point_coord(grid, ij) for ij in CartesianIndices(ash_grid_size(grid)))]
    expected = reshape(expected, ash_grid_size(grid))
    @test maximum(abs.(ð2F .- expected)) < 1e-12
end

@testset "Hessian recovery from ð, ð²: matches ForwardDiff: $(typeof(grid))" for grid in
                                                                                  (EquiangularGrid(63), DriscollHealyGrid(63))
    # End-to-end check: take a smooth non-axisymmetric scalar h(θ,φ) (the
    # rotated oblate-spheroid radius, which has rich m ≠ 0 content), and
    # confirm that the recovery formulas
    #   Hθθ = (Δh + Re ð²h) / 2,
    #   Hφφ = (Δh − Re ð²h) / 2,
    #   Hθφ =  Im ð²h / 2
    # match analytic ForwardDiff differentiation everywhere on the grid.
    # (No backend-specific sign fixes — ASH's extensions reconcile them.)
    M, a, α = 1.0, 0.8, π / 6
    r_plus = M + sqrt(M^2 - a^2)
    sα, cα = sincos(α)
    h(θ, φ) = begin
        cθold = sα * sin(θ) * cos(φ) + cα * cos(θ)
        sθold² = 1 - cθold^2
        1 / sqrt(sθold² / (r_plus^2 + a^2) + cθold^2 / r_plus^2)
    end

    sz = ash_grid_size(grid)
    h_grid = [h(ash_point_coord(grid, ij)...) + 0im for ij in CartesianIndices(sz)]
    hlm = ash_transform(grid, h_grid, 0)

    # Δh via the spin-0 eigenvalue
    Δhlm = copy(hlm)
    for l in 0:ash_lmax(grid), m in -l:l
        Δhlm[ash_mode_index(grid, 0, l, m)] *= -l * (l + 1)
    end
    Δh_g = real.(ash_evaluate(grid, Δhlm, 0))

    # ð²h via the eth chain
    ð2h_g = ash_evaluate(grid, ash_eth(grid, ash_eth(grid, hlm, 0), 1), 2)

    max_err_θθ = 0.0
    max_err_φφ = 0.0
    max_err_θφ = 0.0
    for ij in CartesianIndices(sz)
        θ, φ = ash_point_coord(grid, ij)
        ∂θh = ForwardDiff.derivative(t -> h(t, φ), θ)
        ∂φh = ForwardDiff.derivative(p -> h(θ, p), φ)
        ∂2θh = ForwardDiff.derivative(t -> ForwardDiff.derivative(t -> h(t, φ), t), θ)
        ∂2φh = ForwardDiff.derivative(p -> ForwardDiff.derivative(p -> h(θ, p), p), φ)
        ∂θ∂φh = ForwardDiff.derivative(τ -> ForwardDiff.derivative(p -> h(τ, p), φ), θ)
        sθ, cθ = sincos(θ)
        Hθθ_an = ∂2θh
        Hφφ_an = ∂2φh / sθ^2 + (cθ / sθ) * ∂θh
        Hθφ_an = ∂θ∂φh / sθ - (cθ / sθ) * (∂φh / sθ)
        Hθθ_sp = (Δh_g[ij] + real(ð2h_g[ij])) / 2
        Hφφ_sp = (Δh_g[ij] - real(ð2h_g[ij])) / 2
        Hθφ_sp = imag(ð2h_g[ij]) / 2
        max_err_θθ = max(max_err_θθ, abs(Hθθ_sp - Hθθ_an))
        max_err_φφ = max(max_err_φφ, abs(Hφφ_sp - Hφφ_an))
        max_err_θφ = max(max_err_θφ, abs(Hθφ_sp - Hθφ_an))
    end
    @test max_err_θθ < 1e-9
    @test max_err_φφ < 1e-9
    @test max_err_θφ < 1e-9
end

@testset "Dipole extraction (recentring convention)" begin
    # h = c + d⃗·r̂: the l=1 coefficients must reproduce d⃗ via
    # d_x = −√(3/2π) Re c₁₁, d_y = √(3/2π) Im c₁₁, d_z = √(3/4π) Re c₁₀.
    grid = EquiangularGrid(7)
    d = SVector(0.3, -0.2, 0.5)
    r = [begin
             θ, φ = ash_point_coord(grid, ij)
             r̂ = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
             2.0 + dot(d, r̂)
         end for ij in CartesianIndices(ash_grid_size(grid))]
    hlm = horizon_shape(r, grid)
    c11 = hlm[ash_mode_index(grid, 0, 1, +1)]
    c10 = hlm[ash_mode_index(grid, 0, 1, 0)]
    @test isapprox(-sqrt(3 / (2π)) * real(c11), d[1]; atol=1e-13)
    @test isapprox(+sqrt(3 / (2π)) * imag(c11), d[2]; atol=1e-13)
    @test isapprox(+sqrt(3 / (4π)) * real(c10), d[3]; atol=1e-13)
end

@testset "horizon_points / horizon_shape round trip and resampling" begin
    grid = EquiangularGrid(11)
    origin = SVector(0.1, -0.2, 0.3)
    r = [2.0 + 0.1 * cos(ash_point_coord(grid, ij)[1])^2 for ij in CartesianIndices(ash_grid_size(grid))]
    hlm = horizon_shape(r, grid)
    pts = horizon_points(origin, grid, hlm)
    for ij in CartesianIndices(ash_grid_size(grid))
        θ, φ = ash_point_coord(grid, ij)
        r̂ = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
        @test isapprox(pts[ij], origin + r[ij] * r̂; atol=1e-12)
    end
    # resampling to a finer grid reproduces the analytic shape
    result = (; success=true, iters=0, origin, hlm, area=0.0, H_norm=0.0, grid)
    grid′ = EquiangularGrid(19)
    pts′ = horizon_points(result, grid′)
    for ij in CartesianIndices(ash_grid_size(grid′))
        θ, φ = ash_point_coord(grid′, ij)
        r̂ = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
        @test isapprox(pts′[ij], origin + (2.0 + 0.1 * cos(θ)^2) * r̂; atol=1e-12)
    end
end

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
    success, iters, origin, hlm, area = find_horizon(brill_lindquist_metric, x₀, N, r, atol, maxiters)
    @test success
    # Schwarzschild with M=1: proper area 16π M²
    @test isapprox(area, 16π; rtol=1.0e-6)
end

@testset "Brill-Lindquist on DriscollHealyGrid (SSHT backend)" begin
    # The finder is backend-agnostic: same physics on a different grid.
    x₀ = SVector{3}(0.0, 0.0, 0.1)
    grid = DriscollHealyGrid(7)
    result = find_horizon(brill_lindquist_metric, x₀, grid, 1.0, 1.0e-8, 100; verbosity=0)
    @test result.success
    @test result.grid == grid
    @test isapprox(result.area, 16π; rtol=1.0e-6)
end

@testset "Verbosity" begin
    x₀ = SVector{3}(0.0, 0.0, 0.1)
    N = 8
    r = 1.0
    atol = 1.0e-8
    maxiters = 100
    for verbosity in 0:2
        output = let pipe = Pipe()
            result = redirect_stdout(pipe) do
                res = find_horizon(brill_lindquist_metric, x₀, N, r, atol, maxiters; verbosity)
                close(Base.pipe_writer(pipe))
                res
            end
            @test result.success
            read(pipe, String)
        end
        nlines = count('\n', output)
        if verbosity == 0
            @test nlines == 0
        elseif verbosity == 1
            @test nlines == 1
        else
            @test nlines > 1
        end
    end
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
    success, iters, origin, hlm, area = find_horizon(kerr_schild_metric, SVector{3}(0.0, 0.0, 0.0), hlm, atol, 0)
    @test success
    @test iters == 0
    # Kerr horizon area 4π (r₊² + a²), independent of coordinates
    @test isapprox(area, 4π * (r_plus^2 + a^2); rtol=1.0e-8)
end

@testset "Kerr-Schild" begin
    x₀ = SVector{3}(0.0, 0.0, 0.1)
    N = 8
    r = 2.0
    atol = 1.0e-8
    maxiters = 100
    # Horizon at r=2, or inside this sphere for a>0
    success, iters, origin, hlm, area = find_horizon(kerr_schild_metric, x₀, N, r, atol, maxiters)
    @test success
    M = 1.0
    a = 0.8
    r_plus = M + sqrt(M^2 - a^2)
    # N = 8 limits the spectral accuracy of the area to ~1e-5
    @test isapprox(area, 4π * (r_plus^2 + a^2); rtol=1.0e-4)
end

@testset "Kerr-Schild to round-off" begin
    # With the default atol = 0 the flow runs to the round-off floor
    # (|H| ~ lmax² eps). The area is then limited only by the spectral
    # truncation of the surface, ~1e-10 at N = 16.
    x₀ = SVector{3}(0.0, 0.0, 0.1)
    N = 16
    r = 2.0
    result = find_horizon(kerr_schild_metric, x₀, N, r)
    @test result.success
    @test result.H_norm < 1.0e-11
    M = 1.0
    a = 0.8
    r_plus = M + sqrt(M^2 - a^2)
    @test isapprox(result.area, 4π * (r_plus^2 + a^2); rtol=1.0e-9)

    # A tolerance below the round-off floor is unachievable: the stall
    # detector must stop the iteration long before maxiters.
    result = find_horizon(kerr_schild_metric, x₀, N, r, 1.0e-15, 1000)
    @test !result.success
    @test result.iters < 200
    @test result.H_norm < 1.0e-11
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
    success, iters, origin, hlm, area = find_horizon(rotated_kerr_schild_metric, SVector{3}(0.0, 0.0, 0.0), hlm, atol, 0)
    @test success
    @test iters == 0
    # The proper area is invariant under the rotation
    @test isapprox(area, 4π * (r_plus^2 + a^2); rtol=1.0e-8)
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
    success, iters, origin, hlm, area = find_horizon(harmonic_metric, SVector{3}(0.0, 0.0, 0.0), hlm, atol, 0)
    @test success
    @test iters == 0
    # Same invariant Kerr horizon area in harmonic coordinates
    r_plus = M + sqrt(M^2 - a^2)
    @test isapprox(area, 4π * (r_plus^2 + a^2); rtol=1.0e-8)
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
