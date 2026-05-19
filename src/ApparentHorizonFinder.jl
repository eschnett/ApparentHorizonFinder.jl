module ApparentHorizonFinder

using FastSphericalHarmonics
using LinearAlgebra
using Printf
using StaticArrays

export ADMVars
struct ADMVars{T}
    γ::SMatrix{3,3,T,3^2}
    ∂γ::SArray{Tuple{3,3,3},T,3,3^3}    # ∂γ[i,j,k] = ∂_k γ_{ij}
    K::SMatrix{3,3,T,3^2}
end

function spinsph_mapreduce(op, f, alm::Matrix{Complex{Float64}}, s::Int; init)
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
    function find_horizon(admvars, origin::SVector{3,Float64}, hlm::Matrix{Complex{Float64}}, atol::Float64, maxiters::Int)
    function find_horizon(admvars, origin::SVector{3,Float64}, N::Int, r::Float64, atol::Float64, maxiters::Int)
"""
function find_horizon(admvars, origin::SVector{3,Float64}, hlm::Matrix{Complex{Float64}}, atol::Float64, maxiters::Int)
    N, M = size(hlm)
    @assert M == 2N - 1

    @assert atol >= 0
    @assert maxiters >= 0

    lmax = sph_lmax(N)          # N-1

    # Keep hlm in the spin-0 complex layout (although it is real-valued) so the eth/ethbar chain is uniform

    # Fast flow
    α = 1.0
    β = 0.5
    for iter in 0:maxiters
        Hlm = expansion(admvars, origin, hlm)

        h00 = hlm[spinsph_mode(0, 0, 0)]
        h_avg = real(h00) / sqrt(4π)               # mean = ∫h dΩ / 4π

        h1lm = copy(hlm)
        h1lm[spinsph_mode(0, 0, 0)] = 0
        h1_norm = sqrt(spinsph_mapreduce(+, abs2, h1lm, 0; init=0.0))

        H_norm = sqrt(spinsph_mapreduce(+, abs2, Hlm, 0; init=0.0))

        @printf("iter: %4d   ⟨h⟩: %6.3f   |h-⟨h⟩|: %9.3e   |H|: %9.3e\n", iter, h_avg, h1_norm, H_norm)
        (iter == maxiters || H_norm < atol) && break

        A = α / (lmax * (lmax+1)) + β
        B = β / α
        ρ = 1                   # (28)
        for l in 0:lmax, m in (-l):(+l)
            i = spinsph_mode(0, l, m)
            hlm[i] = hlm[i] - A / (1 + B * l * (l+1)) * (ρ * Hlm[i])
        end
    end

    return hlm
end

function find_horizon(admvars, origin::SVector{3,Float64}, N::Int, r::Float64, atol::Float64, maxiters::Int)
    @assert N > 0
    @assert r > 0
    M = 2N - 1
    h = fill(complex(float(r)), N, M)
    hlm = spinsph_transform(h, 0)
    return find_horizon(admvars, origin, hlm, atol, maxiters)
end

function expansion(admvars, origin::SVector{3}, hlm::Matrix{Complex{Float64}})
    N, M = size(hlm)

    # Spectral derivatives
    ðhlm = spinsph_eth(hlm, 0)     # spin-1 coefficients of ðh
    ð²hlm = spinsph_eth(ðhlm, 1)   # spin-2 coefficients of ð²h
    Δhlm = spinsph_ethbar(ðhlm, 1) # spin-0 coefficients of Δ_S h

    # Grid synthesis
    h = real.(spinsph_evaluate(hlm, 0))
    ðh = spinsph_evaluate(ðhlm, 1)
    ð²h = spinsph_evaluate(ð²hlm, 2)
    Δh = real.(spinsph_evaluate(Δhlm, 0))

    # Orthonormal-frame surface gradient and Hessian of h
    v_θ̂ = real.(ðh)             # ∂_θ h
    v_φ̂ = imag.(ðh)             # (1/sinθ) ∂_φ h
    Hθθ = (Δh + real.(ð²h)) / 2 # ∇_θ̂ ∇_θ̂ h
    Hφφ = (Δh - real.(ð²h)) / 2 # ∇_φ̂ ∇_φ̂ h
    Hθφ = imag.(ð²h) / 2        # ∇_θ̂ ∇_φ̂ h

    # Expansion H at each collocation point
    θgrid, φgrid = sph_points(N)
    H = Matrix{Complex{Float64}}(undef, N, M)
    for j in 1:M, i in 1:N
        H[i, j] = expansion_at_point(
            admvars, origin, θgrid[i], φgrid[j], h[i, j], v_θ̂[i, j], v_φ̂[i, j], Hθθ[i, j], Hφφ[i, j], Hθφ[i, j]
        )
    end

    Hlm = spinsph_transform(H, 0)

    return Hlm::Matrix{Complex{Float64}}
end

# Build ∂_i F and ∂_i ∂_j F in Cartesian components at one collocation point, then
# evaluate the expansion using the ADM data at that point.
function expansion_at_point(admvars, origin::SVector{3}, θ, φ, h, vθ, vφ, Hθθ, Hφφ, Hθφ)
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

    return D_s + Kss - Ktr
end

end
