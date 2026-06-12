module ApparentHorizonFinder

using FastSphericalHarmonics
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

function spinsph_mapreduce(op, f, alm::AbstractMatrix, s::Int; init)
    N, M = size(alm)
    lmax = sph_lmax(N)
    r = init
    for l in abs(s):lmax, m in (-l):(+l)
        r = op(r, f(alm[spinsph_mode(s, l, m)]))
    end
    return r
end

export find_horizon
"""
    find_horizon(admvars, origin, hlm, atol=0.0, maxiters=1000;
                 verbosity=1)     -> (; success, iters, origin, hlm, area, H_norm)
    find_horizon(admvars, origin, N::Int, r::Float64, atol=0.0, maxiters=1000;
                 verbosity=1)     -> (; success, iters, origin, hlm, area, H_norm)

Locate an apparent horizon (marginally outer trapped surface) by the
pseudo-spectral fast flow of Gundlach, arXiv:gr-qc/9707050.

The candidate surface is parameterised as the level set
`F(x) = |x − origin| − h(θ, φ) = 0`, with `(θ, φ)` measured from `origin`. The
shape function `h` is expanded in real spin-0 spherical harmonics (`hlm` is
stored as `Matrix{Float64}` in FastSphericalHarmonics' real-array layout). At
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
- `hlm::Matrix{Float64}`, size `(N, 2N-1)`: initial spin-0 real-array spinsph
  coefficients of `h(θ, φ)`. The second signature builds this from a sphere
  of radius `r > 0` at angular resolution `N`.
- `atol::Float64 = 0.0`: convergence threshold on the L² norm of the
  (`ρ`-weighted) expansion. The default `atol = 0` iterates until the residual
  stops improving, i.e. down to the round-off floor; a positive value stops
  the iteration early once `|H| < atol`.
- `maxiters::Int = 1000`: hard cap on iterations.
- `verbosity::Int = 1` (keyword): `0` prints nothing, `1` prints one summary
  line when the iteration finishes, `2` additionally prints per-iteration
  diagnostics.

# Returns
NamedTuple `(; success, iters, origin, hlm, area, H_norm)`: the convergence
status, number of iterations taken, the final recentred origin, the converged
spin-0 spinsph coefficients of `h`, the proper area of the final surface (see
[`horizon_area`](@ref)), and the final residual (the L² norm of the
`ρ`-weighted expansion). The Cartesian horizon points recover as
`origin + h(θ,φ) · r̂(θ,φ)`.

# Convergence
The fast flow converges linearly and monotonically: the contraction factor per
iteration depends on the spacetime (the deviation of the true linearised
expansion from the flat-space model built into the flow; e.g. ≈ 0.5 for
Schwarzschild, ≈ 0.67 for Kerr `a = 0.8`, ≈ 0.91 for `a = 0.99`) but not on
the resolution `N`. The residual `|H|` bottoms out at a round-off floor of
roughly `lmax² · eps` (≈ 1e-13), at which point the surface itself is accurate
to ≈ 1e-15 provided `N` resolves it; the floor is detected by stalled progress
(no 1% improvement over three consecutive iterations). Running to the floor
costs at most about twice as many iterations as `atol = 1e-8`. With `atol = 0`
the iteration succeeds when it reaches this floor; with `atol > 0` reaching
the floor first means the requested tolerance is unachievable at this
resolution and `success` is `false`.
"""
function find_horizon(
    admvars, origin::SVector{3,Float64}, hlm::Matrix{Float64}, atol::Float64=0.0, maxiters::Int=1000; verbosity::Int=1
)
    N, M = size(hlm)
    @assert M == 2N - 1

    @assert atol >= 0
    @assert maxiters >= 0
    @assert 0 <= verbosity <= 2

    lmax = sph_lmax(N)          # N-1
    @assert lmax >= 1

    # Fast flow
    α = 1.0
    β = 0.5
    H_best = Inf
    stall = 0
    for iter in 0:maxiters
        Hlm = expansion(admvars, origin, hlm; modification=:ρ)

        h00 = hlm[spinsph_mode(0, 0, 0)]
        h_avg = h00 / sqrt(4π)                     # mean = ∫h dΩ / 4π

        # Quadrupole magnitude
        h2_norm = sqrt(sum(abs2(hlm[spinsph_mode(0, 2, m)]) for m in -2:+2))

        h1lm = copy(hlm)
        h1lm[spinsph_mode(0, 0, 0)] = 0
        h1_norm = sqrt(spinsph_mapreduce(+, abs2, h1lm, 0; init=0.0))

        H_norm = sqrt(spinsph_mapreduce(+, abs2, Hlm, 0; init=0.0))

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
            area = horizon_area(admvars, origin, hlm)
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
            return (; success, iters=iter, origin, hlm, area, H_norm)
        end

        A = α / (lmax * (lmax+1)) + β
        B = β / α
        for l in 0:lmax, m in (-l):(+l)
            i = spinsph_mode(0, l, m)
            hlm[i] = hlm[i] - A / (1 + B * l * (l+1)) * Hlm[i]
        end

        # Recenter: move origin to absorb the l=1 dipole of h.
        # In FSH's real-array (real-SH) storage with Y_{l,m} normalised so that
        # ⟨r̂_x⟩ = √(4π/3) Y_{1,+1}, ⟨r̂_y⟩ = √(4π/3) Y_{1,-1}, ⟨r̂_z⟩ = √(4π/3) Y_{1,0},
        # the centroid offset is d_i = 3 ⟨h r̂_i⟩ = √(3/(4π)) · h_{1,*}.
        h1m1 = hlm[spinsph_mode(0, 1, -1)]
        h10 = hlm[spinsph_mode(0, 1, 0)]
        h1p1 = hlm[spinsph_mode(0, 1, +1)]
        Δx = h1p1 * sqrt(3 / (4π))
        Δy = h1m1 * sqrt(3 / (4π))
        Δz = h10 * sqrt(3 / (4π))
        origin = origin + SVector{3}(Δx, Δy, Δz)
        # Linear approximation: subtract the dipole component from h.
        hlm[spinsph_mode(0, 1, -1)] = 0
        hlm[spinsph_mode(0, 1, 0)] = 0
        hlm[spinsph_mode(0, 1, +1)] = 0
    end

    @assert false
end

function find_horizon(admvars, origin::SVector{3,Float64}, N::Int, r::Float64, atol::Float64=0.0, maxiters::Int=1000; verbosity::Int=1)
    @assert N > 0
    @assert r > 0
    M = 2N - 1
    h = fill(float(r), N, M)
    hlm = spinsph_transform(h, 0)
    return find_horizon(admvars, origin, hlm, atol, maxiters; verbosity)
end

export horizon_points
"""
    horizon_points(origin::SVector{3,Float64},
                   hlm::Matrix{Float64}) -> Matrix{SVector{3,Float64}}
    horizon_points(result) -> Matrix{SVector{3,Float64}}

Reconstruct the Cartesian points of the horizon surface on its `(θ, φ)`
collocation grid (size `(N, 2N-1)`, matching `sph_points(N)` from
`FastSphericalHarmonics`). Each entry is `origin + h(θ_i, φ_j) · r̂(θ_i, φ_j)`,
where `h` is evaluated from `hlm` and
`r̂ = (sin θ cos φ, sin θ sin φ, cos θ)`.

The second form accepts the `NamedTuple` returned by [`find_horizon`](@ref).
"""
function horizon_points(origin::SVector{3,Float64}, hlm::Matrix{Float64})
    N, M = size(hlm)
    @assert M == 2N - 1
    h = spinsph_evaluate(hlm, 0)
    θgrid, φgrid = sph_points(N)
    pts = Matrix{SVector{3,Float64}}(undef, N, M)
    for j in 1:M, i in 1:N
        sθ, cθ = sincos(θgrid[i])
        sφ, cφ = sincos(φgrid[j])
        r̂ = SVector(sθ * cφ, sθ * sφ, cθ)
        pts[i, j] = origin + h[i, j] * r̂
    end
    return pts
end
horizon_points(result::NamedTuple) = horizon_points(result.origin, result.hlm)

export horizon_grid
"""
    horizon_grid(N::Int) -> (θ, φ)

Return the `(θ_i, φ_j)` collocation grid that [`find_horizon`](@ref),
[`horizon_points`](@ref), and [`horizon_shape`](@ref) operate on at angular
resolution `N`. The result is the tuple returned by
`FastSphericalHarmonics.sph_points(N)`: `θ` has length `N`, `φ` has length
`2N − 1`.

Typical usage to seed [`find_horizon`](@ref) with an explicit shape:

    θ, φ = horizon_grid(N)
    r = [my_shape(θ[i], φ[j]) for i in 1:length(θ), j in 1:length(φ)]
    hlm = horizon_shape(r)
"""
horizon_grid(N::Int) = sph_points(N)

export horizon_shape
"""
    horizon_shape(r::AbstractMatrix{<:Real}) -> Matrix{Float64}

Transform the radial distances `r[i, j] = h(θ_i, φ_j)` sampled on the
[`horizon_grid`](@ref) into the corresponding spin-0 real-spinsph coefficients,
suitable as the `hlm` argument of [`find_horizon`](@ref). The input must have
size `(N, 2N − 1)` matching `horizon_grid(N)`.

This is the inverse of evaluating `h(θ, φ)` from `hlm`; together with
[`horizon_points`](@ref) it provides a round trip
`r → hlm → points = origin + r · r̂`.
"""
function horizon_shape(r::AbstractMatrix{<:Real})
    N, M = size(r)
    @assert M == 2N - 1
    return spinsph_transform(Matrix{Float64}(r), 0)
end

export horizon_area
"""
    horizon_area(admvars, origin::SVector{3,Float64},
                 hlm::Matrix{Float64}) -> Float64
    horizon_area(admvars, result) -> Float64

Compute the proper area `∮ √(det q) d²y` of the surface described by `origin`
and `hlm`, where `q_AB` is the 2-metric induced on the surface by the spatial
metric `γ_ij` supplied by `admvars` (see [`find_horizon`](@ref)).

The surface `X(θ, φ) = origin + h(θ, φ) r̂(θ, φ)` is differentiated spectrally,
the induced metric is sampled on the collocation grid, and the area element is
integrated by projecting onto the `l = 0` spherical-harmonic mode, so the
result converges spectrally with the angular resolution `N`.

The second form accepts the `NamedTuple` returned by [`find_horizon`](@ref).
"""
function horizon_area(admvars, origin::SVector{3,Float64}, hlm::Matrix{Float64})
    N, M = size(hlm)
    @assert M == 2N - 1

    h = spinsph_evaluate(hlm, 0)
    # ðh: grid values are (−∂_θ h, −(1/sinθ) ∂_φ h); see `expansion`.
    ðh_real = spinsph_evaluate(spinsph_eth(hlm, 0), 1)

    θgrid, φgrid = sph_points(N)
    # Area element per unit solid angle: f = √(det q) / sinθ, built from the
    # orthonormal-frame tangents e_θ = ∂_θ X and e_φ̂ = (1/sinθ) ∂_φ X, which
    # are smooth at the poles.
    f = Matrix{Float64}(undef, N, M)
    for j in 1:M, i in 1:N
        sθ, cθ = sincos(θgrid[i])
        sφ, cφ = sincos(φgrid[j])
        r̂ = SVector(sθ * cφ, sθ * sφ, cθ)
        θ̂ = SVector(cθ * cφ, cθ * sφ, -sθ)
        φ̂ = SVector(-sφ, cφ, 0.0)

        vθ = -ðh_real[i, j][1]          # ∂_θ h
        vφ = -ðh_real[i, j][2]          # (1/sinθ) ∂_φ h
        eθ = vθ * r̂ + h[i, j] * θ̂
        eφ = vφ * r̂ + h[i, j] * φ̂

        X = origin + h[i, j] * r̂
        γ = admvars(X).γ

        qθθ = dot(eθ, γ * eθ)
        qθφ = dot(eθ, γ * eφ)
        qφφ = dot(eφ, γ * eφ)
        f[i, j] = sqrt(qθθ * qφφ - qθφ^2)
    end

    # ∫ f dΩ = √(4π) f₀₀ in FSH's real-array normalisation.
    flm = spinsph_transform(f, 0)
    return sqrt(4π) * flm[spinsph_mode(0, 0, 0)]
end
horizon_area(admvars, result::NamedTuple) = horizon_area(admvars, result.origin, result.hlm)

function expansion(admvars, origin::SVector{3}, hlm::Matrix{Float64}; modification=nothing)
    N, M = size(hlm)
    lmax = N - 1

    # The public `hlm` storage stays in FSH's real-array spin-0 layout (kept
    # for backwards compatibility with `find_horizon`'s dipole arithmetic),
    # but the orthonormal-frame Hessian needs spin-2 quantities that the
    # real-array API does not support. We therefore round-trip h through the
    # complex spin-0 layout, build ð²h via the complex eth chain, and apply
    # an empirically-determined m = −1 column flip (see the FSH-convention
    # tests at the top of `test/runtests.jl`) to recover the textbook
    # spin-2 grid value (Hθθ − Hφφ) + 2i Hθφ.
    h = spinsph_evaluate(hlm, 0)                       # Float64 grid values
    hlm_c = spinsph_transform(ComplexF64.(h), 0)       # complex spin-0 coeffs

    # ∂_θ h and (1/sinθ) ∂_φ h via the real-array path — this version
    # already absorbs the m < 0 spin-1 sign convention into the SVector{2}
    # grid values, so they are interpretable as analytic gradient pieces:
    #   ðh|_{grid}[1] = −∂_θ h,   ðh|_{grid}[2] = −(1/sinθ) ∂_φ h.
    ðh_real = spinsph_evaluate(spinsph_eth(hlm, 0), 1)

    # Δh = ð̄ð h directly in spin-0 coefficient space (eigenvalue −l(l+1)).
    Δhlm_c = copy(hlm_c)
    for l in 0:lmax, m in (-l):l
        Δhlm_c[spinsph_mode(0, l, m)] *= -l * (l + 1)
    end
    Δh = real.(spinsph_evaluate(Δhlm_c, 0))

    # ð²h via the complex eth chain. FSH's spin-2 grid evaluation agrees with
    # the textbook (Hθθ − Hφφ) + 2i Hθφ at every m EXCEPT m = −1, where it
    # has the opposite sign; flip column 2 of the spin-2 coefficients to
    # absorb that.
    ð2hlm_c = spinsph_eth(spinsph_eth(hlm_c, 0), 1)
    ð2hlm_c[:, 2] .*= -1
    ð2h = spinsph_evaluate(ð2hlm_c, 2)

    θgrid, φgrid = sph_points(N)
    H = Matrix{Float64}(undef, N, M)
    for j in 1:M, i in 1:N
        vθ = -ðh_real[i, j][1]              # ∂_θ h
        vφ = -ðh_real[i, j][2]              # (1/sinθ) ∂_φ h
        Δhij = Δh[i, j]
        re_ð2 = real(ð2h[i, j])
        im_ð2 = imag(ð2h[i, j])
        Hθθ = (Δhij + re_ð2) / 2            # ∂²_θ h
        Hφφ = (Δhij - re_ð2) / 2            # (1/sin²θ) ∂²_φ h + cotθ ∂_θ h
        Hθφ = im_ð2 / 2                     # (1/sinθ) ∂_θ ∂_φ h − cotθ v_φ̂
        H[i, j] = expansion_at_point(
            admvars, origin, θgrid[i], φgrid[j], h[i, j], vθ, vφ, Hθθ, Hφφ, Hθφ; modification
        )
    end

    Hlm = spinsph_transform(H, 0)
    return Hlm::Matrix{Float64}
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
