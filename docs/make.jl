# Generate documentation with this command:
# (cd docs && julia make.jl)

push!(LOAD_PATH, "..")

using Documenter
using ApparentHorizonFinder

makedocs(; sitename="ApparentHorizonFinder", format=Documenter.HTML(), modules=[ApparentHorizonFinder])

deploydocs(; repo="github.com/eschnett/ApparentHorizonFinder.jl.git", devbranch="main", push_preview=true)
