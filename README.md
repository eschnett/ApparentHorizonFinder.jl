# ApparentHorizonFinder.jl

Find apparent horizons in a spacelike hypersurface.

[![CI](https://github.com/eschnett/ApparentHorizonFinder.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/ApparentHorizonFinder.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/eschnett/ApparentHorizonFinder.jl/actions/workflows/docs.yml/badge.svg)](https://eschnett.github.io/ApparentHorizonFinder.jl/dev/)

## Overview

In [General
Relativity](https://en.wikipedia.org/wiki/General_relativity),
[singularities](https://en.wikipedia.org/wiki/Gravitational_singularity)
are surrounded by [event
horizons](https://en.wikipedia.org/wiki/Event_horizon). ([Most
likely](https://en.wikipedia.org/wiki/Cosmic_censorship_hypothesis).)
Finding event horizons is difficult in practice since one needs to
start in the infinite future (or at a time in the future when the
spacetime has becone stationary) and then follow the event horizon
backwards in time. This is not possible when the future of the
spacetime is not yet known, for example during a numerical simulation.

[Apparent horizons](https://en.wikipedia.org/wiki/Apparent_horizon)
are a different kind of horizon which share many properties of event
horizons. They are defined quasi-locally in time, i.e. they can be
found if one knows only one instant of time of the spacetime.
Determining where the horizon is located is non-trivial; it requires
solving a non-linear elliptic equation.

This package implements the fast flow method by Gundlach to find
apparent horizons:
- C. Gundlach, "Pseudo-spectral apparent horizon finders: an efficient
  new algorithm", Phys. Rev. D 57, 863 (1998),
  [DOI:10.1103/PhysRevD.57.863](https://doi.org/10.1103/PhysRevD.57.863),
  [arXiv:gr-qc/9707050](https://arxiv.org/abs/gr-qc/9707050).

See also
- J. Thornburg, "Event and Apparent Horizon Finders for 3 + 1
  Numerical Relativity", Living Rev. Relativ. 10, 3 (2007).
  [DOI:10.12942/lrr-2007-3](https://doi.org/10.12942/lrr-2007-3),
  [arXiv:gr-qc/0512169](https://arxiv.org/abs/gr-qc/0512169).

## Example

### Define spacetime

Let us look for the horizon of a [rotating black
hole](https://en.wikipedia.org/wiki/Kerr_metric) in Kerr-Schild
coordinates. We first define the metric. This function is called from
the horizon finder; it needs to take a 3d point as input and return an
`ADMVars` struct.

```julia
using ApparentHorizonFinder
using SpacetimeMetrics
using StaticArrays
function kerr_schild_metric(p::SVector{3})
    x, y, z = p
    M = 1.0
    a = 0.9
    ks = KerrSchild(M, a)
    t = 0
    g, ∂g = dmetric(ks, SVector{4}(t, x, y, z))
    K = ExtrinsicCurvature(ks, SVector{4}(t, x, y, z))
    γ = SMatrix{3,3}(g[i,j] for i in 2:4, j in 2:4)
    ∂γ = SArray{Tuple{3,3,3}}(∂g[i,j,k] for i in 2:4, j in 2:4, k in 2:4)
    admvars = ADMVars(γ, ∂γ, K)
    return admvars
end
```

### Find horizon

Next we call the horizon finder:

```julia
x₀ = SVector{3}(0.0, 0.0, 0.1)
N = 16
r = 2.0
atol = 1.0e-8
maxiters = 100
AH = find_horizon(kerr_schild_metric, x₀, N, r, atol, maxiters)
pts = horizon_points(AH.origin, AH.hlm)
```

The number of points (and the number of multipoles) depends on the chosen `N`.

### Plot result

```julia
using CairoMakie
using SixelTerm   # optional, to show output directly in the terminal
pts = hcat(pts, pts[:, 1:1]);   # close the φ seam at the back
X, Y, Z = getindex.(pts, 1), getindex.(pts, 2), getindex.(pts, 3);
fig, ax, _ = wireframe(X, Y, Z; color=:black, linewidth=0.5);
scatter!(ax, vec(X), vec(Y), vec(Z); color=:red, markersize = 6);
fig
save("horizon.png", fig)
```

![Horizon surface](horizon.png "Apparent horizon for a Kerr-Schild metric with M=1, a=0.6")
