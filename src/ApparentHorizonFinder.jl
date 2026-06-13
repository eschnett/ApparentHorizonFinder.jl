module ApparentHorizonFinder

using AbstractSphericalHarmonics
import FastSphericalHarmonics    # default backend (EquiangularGrid)
using LinearAlgebra
using Printf
using StaticArrays

export ADMVars
"""
    ADMVars(γ, ∂γ, K)

ADM 3+1 Cauchy data at one spatial point, expressed in Cartesian coordinates.

- `γ::SMatrix{3,3}`           — induced spatial metric, `γ[i,j] = γ_{ij}`.
- `∂γ::SArray{Tuple{3,3,3}}`  — partial derivatives, `∂γ[i,j,k] = ∂_k γ_{ij}`.
- `K::SMatrix{3,3}`           — extrinsic curvature with the standard sign
  convention `K_{ij} = −(1/2α)(∂_t γ_{ij} − D_i β_j − D_j β_i)`.

[`find_horizon`](@ref) expects a callable `x::SVector{3,Float64} -> ADMVars`
that supplies these quantities at every queried point.
"""
struct ADMVars{T}
    γ::SMatrix{3,3,T,3^2}
    ∂γ::SArray{Tuple{3,3,3},T,3,3^3}    # ∂γ[i,j,k] = ∂_k γ_{ij}
    K::SMatrix{3,3,T,3^2}
end

"Map-reduce over the (l, m) modes of a spin-`s` coefficient vector"
function mode_mapreduce(op, f, alm::AbstractVector, grid::SphereGrid, s::Int; init)
    r = init
    for l in abs(s):ash_lmax(grid), m in (-l):(+l)
        r = op(r, f(alm[ash_mode_index(grid, s, l, m)]))
    end
    return r
end

export find_horizon
"""
    find_horizon(admvars, origin, grid::SphereGrid, hlm::Vector{ComplexF64},
                 atol=0.0, maxiters=1000; verbosity=1)
    find_horizon(admvars, origin, grid::SphereGrid, r::Real, atol=0.0, maxiters=1000;
                 verbosity=1)
    find_horizon(admvars, origin, N::Int, r::Real, atol=0.0, maxiters=1000;
                 verbosity=1)
    find_horizon(admvars, origin, hlm::Vector{ComplexF64}, atol=0.0, maxiters=1000;
                 verbosity=1)

Locate an apparent horizon (marginally outer trapped surface) by the
pseudo-spectral fast flow of Gundlach, arXiv:gr-qc/9707050.  Returns
`(; success, iters, origin, hlm, area, H_norm, grid)`.

The candidate surface is parameterised as the level set
`F(x) = |x − origin| − h(θ, φ) = 0`, with `(θ, φ)` measured from `origin`. The
shape function `h` is expanded in spherical harmonics on the
AbstractSphericalHarmonics grid `grid` (`hlm` is a coefficient vector in the
canonical layout, see `AbstractSphericalHarmonics.ash_mode_index`; `h` is
real, so the coefficients satisfy the corresponding reality condition). At
each iteration:

1. The expansion `H = D_i s^i + K_{ij} s^i s^j − K` is evaluated on the
   surface, with the principal-symbol weight `ρ` from eq. (28) of the paper
   applied to keep the iteration well-conditioned.
2. `hlm` is updated by the fast-flow rule
   `Δh_{lm} = −A/(1 + B·l(l+1)) · (ρ H)_{lm}`, which acts as `(L²)⁻¹` on the
   high-`l` modes and as a small relaxation on `l = 0`.
3. `origin` is recentred by absorbing the `l = 1` dipole of `h`; this keeps
   the surface close to spherical about `origin`, which the algorithm requires
   for good convergence.

# Arguments
- `admvars`: callable `admvars(x::SVector{3,Float64}) -> ADMVars` returning the
  local ADM data at the Cartesian point `x`.
- `origin::SVector{3,Float64}`: initial parametrisation origin. Must lie
  inside the star-shaped candidate surface.
- `grid::SphereGrid`: the collocation grid (and thereby the transform
  backend); e.g. `EquiangularGrid(lmax)` (backend FastSphericalHarmonics,
  loaded by this package) or `DriscollHealyGrid(lmax)` (backend SSHT, load it
  with `import SSHT`).
- `hlm::Vector{ComplexF64}`: initial spin-0 coefficients of `h(θ, φ)` in the
  canonical layout.  The `r::Real` forms start from a sphere of radius `r > 0`;
  the `N::Int` form uses `EquiangularGrid(N − 1)`; the grid-less `hlm` form
  infers `EquiangularGrid(isqrt(length(hlm)) − 1)`.
- `atol::Float64 = 0.0`: convergence threshold on the L² norm of the
  (`ρ`-weighted) expansion. The default `atol = 0` iterates until the residual
  stops improving, i.e. down to the round-off floor; a positive value stops
  the iteration early once `|H| < atol`.
- `maxiters::Int = 1000`: hard cap on iterations.
- `verbosity::Int = 1` (keyword): `0` prints nothing, `1` prints one summary
  line when the iteration finishes, `2` additionally prints per-iteration
  diagnostics.

# Returns
NamedTuple `(; success, iters, origin, hlm, area, H_norm, grid)`: the
convergence status, number of iterations taken, the final recentred origin,
the converged coefficients of `h`, the proper area of the final surface (see
[`horizon_area`](@ref)), the final residual (the L² norm of the `ρ`-weighted
expansion), and the grid. The Cartesian horizon points recover as
`origin + h(θ,φ) · r̂(θ,φ)`; see [`horizon_points`](@ref).

# Convergence
The fast flow converges linearly and monotonically: the contraction factor per
iteration depends on the spacetime (the deviation of the true linearised
expansion from the flat-space model built into the flow; e.g. ≈ 0.5 for
Schwarzschild, ≈ 0.67 for Kerr `a = 0.8`, ≈ 0.91 for `a = 0.99`) but not on
the resolution. The residual `|H|` bottoms out at a round-off floor of
roughly `lmax² · eps` (≈ 1e-13), at which point the surface itself is accurate
to ≈ 1e-15 provided the resolution resolves it; the floor is detected by
stalled progress (no 1% improvement over three consecutive iterations).
Running to the floor costs at most about twice as many iterations as
`atol = 1e-8`. With `atol = 0` the iteration succeeds when it reaches this
floor; with `atol > 0` reaching the floor first means the requested tolerance
is unachievable at this resolution and `success` is `false`.
"""
function find_horizon(
    admvars,
    origin::SVector{3,Float64},
    grid::SphereGrid,
    hlm::Vector{ComplexF64},
    atol::Float64=0.0,
    maxiters::Int=1000;
    verbosity::Int=1,
)
    @assert length(hlm) == ash_nmodes(grid)[1]

    @assert atol >= 0
    @assert maxiters >= 0
    @assert 0 <= verbosity <= 2

    lmax = ash_lmax(grid)
    @assert lmax >= 1

    hlm = copy(hlm)

    # Fast flow
    α = 1.0
    β = 0.5
    H_best = Inf
    stall = 0
    for iter in 0:maxiters
        Hlm = expansion(admvars, origin, grid, hlm; modification=:ρ)

        h00 = real(hlm[ash_mode_index(grid, 0, 0, 0)])
        h_avg = h00 / sqrt(4π)                     # mean = ∫h dΩ / 4π

        # Quadrupole magnitude
        h2_norm = lmax >= 2 ? sqrt(sum(abs2(hlm[ash_mode_index(grid, 0, 2, m)]) for m in -2:+2)) : 0.0

        h1lm = copy(hlm)
        h1lm[ash_mode_index(grid, 0, 0, 0)] = 0
        h1_norm = sqrt(mode_mapreduce(+, abs2, h1lm, grid, 0; init=0.0))

        H_norm = sqrt(mode_mapreduce(+, abs2, Hlm, grid, 0; init=0.0))

        # Stall detection: the flow converges linearly (typically by ≥ 9% per
        # iteration even for near-extremal Kerr), so three consecutive
        # iterations that fail to improve the best residual by at least 1%
        # signal the round-off (or truncation) floor.
        stall = H_norm < 0.99 * H_best ? 0 : stall + 1
        H_best = min(H_best, H_norm)

        verbosity >= 2 && @printf(
            "iter: %4d   ⟨h⟩: %6.3f   |h-⟨h⟩|: %9.3e   |hₗ₌₂|: %9.3e   |H|: %9.3e   x₀: (%+.3f, %+.3f, %+.3f)\n",
            iter,
            h_avg,
            h1_norm,
            h2_norm,
            H_norm,
            origin[1],
            origin[2],
            origin[3]
        )
        if (atol > 0 && H_norm < atol) || stall >= 3 || iter == maxiters || h_avg < 1.0e-6
            # With atol = 0, stalling at the floor is the goal; with atol > 0
            # it means the requested tolerance is unachievable.
            success = atol > 0 ? H_norm < atol : stall >= 3
            area = horizon_area(admvars, origin, grid, hlm)
            verbosity >= 1 && @printf(
                "find_horizon: %s after %d iterations   ⟨h⟩: %.6f   |H|: %.3e   area: %.6f   x₀: (%+.3f, %+.3f, %+.3f)\n",
                success ? "converged" : "failed to converge",
                iter,
                h_avg,
                H_norm,
                area,
                origin[1],
                origin[2],
                origin[3]
            )
            return (; success, iters=iter, origin, hlm, area, H_norm, grid)
        end

        A = α / (lmax * (lmax + 1)) + β
        B = β / α
        for l in 0:lmax, m in (-l):(+l)
            i = ash_mode_index(grid, 0, l, m)
            hlm[i] = hlm[i] - A / (1 + B * l * (l + 1)) * Hlm[i]
        end

        # Recenter: move origin to absorb the l=1 dipole of h.
        # With the canonical (Wikipedia, complex) spherical harmonics,
        # h ⊃ d⃗·r̂ has the l=1 coefficients
        #     c_{1,0}  = √(4π/3) d_z ,
        #     c_{1,±1} = √(2π/3) (∓d_x + i d_y) ,
        # so the centroid offset recovers as
        #     d_x = −√(3/2π) Re c_{1,1},  d_y = √(3/2π) Im c_{1,1},
        #     d_z =  √(3/4π) Re c_{1,0} .
        c11 = hlm[ash_mode_index(grid, 0, 1, +1)]
        c10 = hlm[ash_mode_index(grid, 0, 1, 0)]
        Δx = -sqrt(3 / (2π)) * real(c11)
        Δy = +sqrt(3 / (2π)) * imag(c11)
        Δz = +sqrt(3 / (4π)) * real(c10)
        origin = origin + SVector{3}(Δx, Δy, Δz)
        # Linear approximation: subtract the dipole component from h.
        hlm[ash_mode_index(grid, 0, 1, -1)] = 0
        hlm[ash_mode_index(grid, 0, 1, 0)] = 0
        hlm[ash_mode_index(grid, 0, 1, +1)] = 0
    end

    @assert false
end

function find_horizon(
    admvars, origin::SVector{3,Float64}, grid::SphereGrid, r::Real, atol::Float64=0.0, maxiters::Int=1000; verbosity::Int=1
)
    @assert r > 0
    hlm = zeros(ComplexF64, ash_nmodes(grid))
    hlm[ash_mode_index(grid, 0, 0, 0)] = sqrt(4π) * r
    return find_horizon(admvars, origin, grid, hlm, atol, maxiters; verbosity)
end

function find_horizon(
    admvars, origin::SVector{3,Float64}, N::Int, r::Real, atol::Float64=0.0, maxiters::Int=1000; verbosity::Int=1
)
    @assert N > 1
    return find_horizon(admvars, origin, EquiangularGrid(N - 1), r, atol, maxiters; verbosity)
end

function find_horizon(
    admvars, origin::SVector{3,Float64}, hlm::Vector{ComplexF64}, atol::Float64=0.0, maxiters::Int=1000; verbosity::Int=1
)
    return find_horizon(admvars, origin, EquiangularGrid(isqrt(length(hlm)) - 1), hlm, atol, maxiters; verbosity)
end

export horizon_points
"""
    horizon_points(origin::SVector{3,Float64}, grid::SphereGrid,
                   hlm::Vector{ComplexF64}) -> Matrix{SVector{3,Float64}}
    horizon_points(result) -> Matrix{SVector{3,Float64}}
    horizon_points(result, grid′::SphereGrid) -> Matrix{SVector{3,Float64}}

Reconstruct the Cartesian points of the horizon surface on the collocation
grid (size `ash_grid_size(grid)`).  Each entry is
`origin + h(θ, φ) · r̂(θ, φ)` at the grid point's `(θ, φ)` from
`AbstractSphericalHarmonics.ash_point_coord`, where
`r̂ = (sin θ cos φ, sin θ sin φ, cos θ)`.

The second form accepts the `NamedTuple` returned by [`find_horizon`](@ref);
the third form resamples the shape onto a different grid `grid′` (spectral
zero-padding or truncation via `AbstractSphericalHarmonics.ash_resample` —
the grids may even belong to different backends).
"""
function horizon_points(origin::SVector{3,Float64}, grid::SphereGrid, hlm::Vector{ComplexF64})
    @assert length(hlm) == ash_nmodes(grid)[1]
    h = real.(ash_evaluate(grid, hlm, 0))
    pts = Matrix{SVector{3,Float64}}(undef, ash_grid_size(grid)...)
    for ij in CartesianIndices(pts)
        θ, φ = ash_point_coord(grid, ij)
        sθ, cθ = sincos(θ)
        sφ, cφ = sincos(φ)
        r̂ = SVector(sθ * cφ, sθ * sφ, cθ)
        pts[ij] = origin + h[ij] * r̂
    end
    return pts
end
horizon_points(result::NamedTuple) = horizon_points(result.origin, result.grid, result.hlm)
function horizon_points(result::NamedTuple, grid′::SphereGrid)
    hlm′ = ash_resample(grid′, result.hlm, result.grid, 0)
    return horizon_points(result.origin, grid′, hlm′)
end

export horizon_grid
"""
    horizon_grid(grid::SphereGrid) -> (θ, φ)
    horizon_grid(N::Int) -> (θ, φ)

Return the θ and φ values of the collocation grid (the vectors from
`AbstractSphericalHarmonics.ash_thetas`/`ash_phis`).  The integer form is a
convenience for `EquiangularGrid(N − 1)`, whose grid layout is `(θ, φ)` with
`length(θ) = N` and `length(φ) = 2N − 1`.  For layout-independent code prefer
iterating `CartesianIndices(ash_grid_size(grid))` and querying
`ash_point_coord(grid, ij)`.

Typical usage to seed [`find_horizon`](@ref) with an explicit shape on an
`EquiangularGrid`:

    θ, φ = horizon_grid(N)
    r = [my_shape(θ[i], φ[j]) for i in 1:length(θ), j in 1:length(φ)]
    hlm = horizon_shape(r)
"""
horizon_grid(grid::SphereGrid) = (ash_thetas(grid), ash_phis(grid))
horizon_grid(N::Int) = horizon_grid(EquiangularGrid(N - 1))

export horizon_shape
"""
    horizon_shape(r::AbstractMatrix{<:Real}, grid::SphereGrid) -> Vector{ComplexF64}
    horizon_shape(r::AbstractMatrix{<:Real}) -> Vector{ComplexF64}

Transform the radial distances `r[ij] = h(θ_ij, φ_ij)` sampled on the
collocation grid into the corresponding spin-0 coefficients (canonical
layout), suitable as the `hlm` argument of [`find_horizon`](@ref). The input
must have size `ash_grid_size(grid)`; the grid-less form infers
`EquiangularGrid(size(r, 1) − 1)`.

This is the inverse of evaluating `h(θ, φ)` from `hlm`; together with
[`horizon_points`](@ref) it provides a round trip
`r → hlm → points = origin + r · r̂`.
"""
function horizon_shape(r::AbstractMatrix{<:Real}, grid::SphereGrid)
    @assert size(r) == ash_grid_size(grid)
    return ash_transform(grid, Matrix{ComplexF64}(r), 0)
end
horizon_shape(r::AbstractMatrix{<:Real}) = horizon_shape(r, EquiangularGrid(size(r, 1) - 1))

export horizon_area
"""
    horizon_area(admvars, origin::SVector{3,Float64}, grid::SphereGrid,
                 hlm::Vector{ComplexF64}) -> Float64
    horizon_area(admvars, result) -> Float64

Compute the proper area `∮ √(det q) d²y` of the surface described by `origin`
and `hlm`, where `q_AB` is the 2-metric induced on the surface by the spatial
metric `γ_ij` supplied by `admvars` (see [`find_horizon`](@ref)).

The surface `X(θ, φ) = origin + h(θ, φ) r̂(θ, φ)` is differentiated spectrally,
the induced metric is sampled on the collocation grid, and the area element is
integrated by projecting onto the `l = 0` spherical-harmonic mode, so the
result converges spectrally with the angular resolution.

The second form accepts the `NamedTuple` returned by [`find_horizon`](@ref).
"""
function horizon_area(admvars, origin::SVector{3,Float64}, grid::SphereGrid, hlm::Vector{ComplexF64})
    @assert length(hlm) == ash_nmodes(grid)[1]

    h = real.(ash_evaluate(grid, hlm, 0))
    # ðh = −(∂_θ + (i/sinθ) ∂_φ) h on the grid (Wikipedia eth convention)
    ðh = ash_evaluate(grid, ash_eth(grid, hlm, 0), 1)

    # Area element per unit solid angle: f = √(det q) / sinθ, built from the
    # orthonormal-frame tangents e_θ = ∂_θ X and e_φ̂ = (1/sinθ) ∂_φ X, which
    # are smooth at the poles.
    f = Matrix{Float64}(undef, ash_grid_size(grid)...)
    for ij in CartesianIndices(f)
        θ, φ = ash_point_coord(grid, ij)
        sθ, cθ = sincos(θ)
        sφ, cφ = sincos(φ)
        r̂ = SVector(sθ * cφ, sθ * sφ, cθ)
        θ̂ = SVector(cθ * cφ, cθ * sφ, -sθ)
        φ̂ = SVector(-sφ, cφ, 0.0)

        vθ = -real(ðh[ij])              # ∂_θ h
        vφ = -imag(ðh[ij])              # (1/sinθ) ∂_φ h
        eθ = vθ * r̂ + h[ij] * θ̂
        eφ = vφ * r̂ + h[ij] * φ̂

        X = origin + h[ij] * r̂
        γ = admvars(X).γ

        qθθ = dot(eθ, γ * eθ)
        qθφ = dot(eθ, γ * eφ)
        qφφ = dot(eφ, γ * eφ)
        f[ij] = sqrt(qθθ * qφφ - qθφ^2)
    end

    # ∫ f dΩ = √(4π) f₀₀
    flm = ash_transform(grid, ComplexF64.(f), 0)
    return sqrt(4π) * real(flm[ash_mode_index(grid, 0, 0, 0)])
end
horizon_area(admvars, result::NamedTuple) = horizon_area(admvars, result.origin, result.grid, result.hlm)

function expansion(admvars, origin::SVector{3}, grid::SphereGrid, hlm::Vector{ComplexF64}; modification=nothing)
    lmax = ash_lmax(grid)

    h = real.(ash_evaluate(grid, hlm, 0))

    # ðh = −(∂_θ + (i/sinθ) ∂_φ) h on the grid:
    #   ∂_θ h = −Re ðh,   (1/sinθ) ∂_φ h = −Im ðh.
    ðh = ash_evaluate(grid, ash_eth(grid, hlm, 0), 1)

    # Δh = ð̄ð h directly in spin-0 coefficient space (eigenvalue −l(l+1)).
    Δhlm = copy(hlm)
    for l in 0:lmax, m in (-l):l
        Δhlm[ash_mode_index(grid, 0, l, m)] *= -l * (l + 1)
    end
    Δh = real.(ash_evaluate(grid, Δhlm, 0))

    # ð²h: with the canonical (Wikipedia) eth conventions, the spin-2 grid
    # value is exactly the textbook orthonormal-frame combination
    #   ð²h = (H_θ̂θ̂ − H_φ̂φ̂) + 2i H_θ̂φ̂ ,
    # where H_θ̂θ̂ = ∂²_θ h, H_φ̂φ̂ = (1/sin²θ) ∂²_φ h + cotθ ∂_θ h,
    # H_θ̂φ̂ = (1/sinθ) ∂_θ∂_φ h − (cotθ/sinθ) ∂_φ h.  Combined with the trace
    # Δh = H_θ̂θ̂ + H_φ̂φ̂ this yields the full Hessian.
    ð2h = ash_evaluate(grid, ash_eth(grid, ash_eth(grid, hlm, 0), 1), 2)

    H = Matrix{Float64}(undef, ash_grid_size(grid)...)
    for ij in CartesianIndices(H)
        θ, φ = ash_point_coord(grid, ij)
        vθ = -real(ðh[ij])                  # ∂_θ h
        vφ = -imag(ðh[ij])                  # (1/sinθ) ∂_φ h
        Δhij = Δh[ij]
        re_ð2 = real(ð2h[ij])
        im_ð2 = imag(ð2h[ij])
        Hθθ = (Δhij + re_ð2) / 2            # ∂²_θ h
        Hφφ = (Δhij - re_ð2) / 2            # (1/sin²θ) ∂²_φ h + cotθ ∂_θ h
        Hθφ = im_ð2 / 2                     # (1/sinθ) ∂_θ ∂_φ h − cotθ v_φ̂
        H[ij] = expansion_at_point(admvars, origin, θ, φ, h[ij], vθ, vφ, Hθθ, Hφφ, Hθφ; modification)
    end

    Hlm = ash_transform(grid, ComplexF64.(H), 0)
    return Hlm::Vector{ComplexF64}
end

# Build ∂_i F and ∂_i ∂_j F in Cartesian components at one collocation point, then
# evaluate the expansion using the ADM data at that point.
function expansion_at_point(admvars, origin::SVector{3}, θ, φ, h, vθ, vφ, Hθθ, Hφφ, Hθφ; modification=nothing)
    sθ, cθ = sincos(θ)
    sφ, cφ = sincos(φ)

    # Orthonormal spherical basis at this angular position, Cartesian components
    r̂ = SVector(sθ * cφ, sθ * sφ, cθ)
    θ̂ = SVector(cθ * cφ, cθ * sφ, -sθ)
    φ̂ = SVector(-sφ, cφ, zero(θ))

    # Cartesian position of the surface point (F = 0)
    X = origin + h * r̂

    # ∇F and ∇²F in the orthonormal spherical basis (smooth at poles)
    ∇F_sph = SVector(1.0, -vθ / h, -vφ / h)
    HF_sph = SMatrix{3,3}(0.0, vθ / h^2, vφ / h^2, vθ / h^2, 1 / h - Hθθ / h^2, -Hθφ / h^2, vφ / h^2, -Hθφ / h^2, 1 / h - Hφφ / h^2)

    # Rotate to Cartesian: columns of R are r̂, θ̂, φ̂
    R = hcat(r̂, θ̂, φ̂)
    ∇F = R * ∇F_sph
    ∇²F = R * HF_sph * R'

    # H = D_i s^i + K_{ij} s^i s^j - K  with  s^i = γ^{ij} ∂_j F / |∇F|_γ.
    adm = admvars(X)
    γ = adm.γ
    ∂γ = adm.∂γ                 # ∂γ[i,j,k] = ∂_k γ_{ij}
    K = adm.K

    γinv = inv(γ)

    # ∂_k γ^{ij} = -γ^{ia} γ^{jb} ∂_k γ_{ab}
    ∂γinv = SArray{Tuple{3,3,3}}(
        -sum(γinv[i, a] * γinv[j, b] * ∂γ[a, b, k] for a in 1:3, b in 1:3) for i in 1:3, j in 1:3, k in 1:3
    )

    N² = dot(∇F, γinv * ∇F)
    Nnorm = sqrt(N²)
    s = (γinv * ∇F) / Nnorm

    # ∂_i N = [½ (∂_i γ^{kl}) F_k F_l + γ^{kl} F_{ik} F_l] / N
    ∂N = SVector{3}(
        (
            sum(∂γinv[k, l, i] * ∇F[k] * ∇F[l] for k in 1:3, l in 1:3) / 2 +
            sum(γinv[k, l] * ∇²F[i, k] * ∇F[l] for k in 1:3, l in 1:3)
        ) / Nnorm for i in 1:3
    )

    # ∂_i s^i = (∂_i γ^{ij}) F_j / N + γ^{ij} F_{ij} / N - s^i ∂_i N / N
    div_s =
        sum(∂γinv[i, j, i] * ∇F[j] for i in 1:3, j in 1:3) / Nnorm + sum(γinv[i, j] * ∇²F[i, j] for i in 1:3, j in 1:3) / Nnorm -
        dot(s, ∂N) / Nnorm

    # Γ^i_{ij} = ½ γ^{kl} ∂_j γ_{kl}
    trΓ = SVector{3}(sum(γinv[k, l] * ∂γ[k, l, j] for k in 1:3, l in 1:3) / 2 for j in 1:3)

    D_s = div_s + dot(trΓ, s)

    Ktr = sum(γinv[i, j] * K[i, j] for i in 1:3, j in 1:3)
    Kss = dot(s, K * s)

    H = D_s + Kss - Ktr

    if modification === :ρ
        # (28): ρ = 2 r² |∇F| / [(γ^{ab} - s^a s^b)(ḡ_{ab} - ∇̄_a r ∇̄_b r)]
        # In Cartesian with origin at O: ḡ_{ab} = δ_{ab}, ∇̄_a r = r̂_a, r = h on the surface.
        denom = tr(γinv) - dot(r̂, γinv * r̂) - dot(s, s) + dot(s, r̂)^2
        ρ = 2 * h^2 * Nnorm / denom
        H = ρ * H
    end

    return H
end

end
