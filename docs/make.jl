using Documenter

# Note: the docs are hand-authored markdown (no `@docs` autodocs), so this build
# does NOT load MotifInference — it therefore builds anywhere, with no GPU / CUDA
# stack required on CI. If you later add `@docs`/doctest blocks, add MotifInference
# to docs/Project.toml (via `Pkg.develop`) and `using MotifInference` here.

makedocs(
    sitename = "MotifInference.jl",
    authors = "Shane Kuei-Hsien Chu",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://kchu25.github.io/MotifInference.jl",
    ),
    pages = [
        "Home" => "index.md",
        "run_method reference" => "run_method.md",
    ],
)

deploydocs(
    repo = "github.com/kchu25/MotifInference.jl.git",
    devbranch = "main",
    push_preview = false,
)
