# MotifInference.jl

Motif inference from biological sequences via a CNN-based pipeline.

## Installation

This package depends on several unregistered packages that are referenced by
local path in `Project.toml` (`[sources]`): `BanzhafInference`,
`GlyphEctoplasm`, `AutoComputationalGraphTuning`, and `SEQ2EXPdata`. Make sure
those repositories are available at the expected paths before instantiating.

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

```julia
using MotifInference

# Run the pipeline directly from a .jld2 dataset:
run_method("/path/to/mydata.jld2"; seq_type=:rna, seed=42)

# Or build a run configuration first:
trc = make_trc("/path/to/mydata.jld2"; seed=42)
```

## Project layout

```
src/
  MotifInference.jl       # module entry point
  struct_def.jl           # run-config / data types
  tuning_and_train_final.jl
  plotting.jl
  run_thru.jl
  dataset_utils.jl
  datasets.jl             # dataset definitions
  pipeline.jl             # make_trc / run_method
  VeryBasicCNN2/          # vendored CNN sub-module
test/
  runtests.jl             # smoke tests
  integration_tests.jl    # data-dependent (requires MOTIFINFERENCE_TEST_DATA)
```

## Testing

```julia
using Pkg; Pkg.test()
```

The integration tests need a real dataset and a GPU. Enable them by pointing
`MOTIFINFERENCE_TEST_DATA` at a `.jld2` file:

```bash
MOTIFINFERENCE_TEST_DATA=/path/to/data.jld2 julia --project -e 'using Pkg; Pkg.test()'
```
