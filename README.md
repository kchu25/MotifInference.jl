# MotifInference.jl

Discover sequence motifs — and the interactions between them — from
sequence-to-measurement data (DNA / RNA / protein). A neural network is trained
to predict the measured signal from sequence, then **motifs are read back out of
the trained model** via game-theoretic attribution rather than read off the raw
weights.

## Method

The pipeline has three stages:

1. **Model** — `SeqCNN`, a sequence-to-expression network with:
   - a base layer of **learnable PWMs** (position weight matrices) for primary
     motif detection,
   - a stack of hierarchical convolutional layers (optional LayerNorm,
     channel masking),
   - an **EfficientNet-style backbone** of MBConv blocks with
     squeeze-and-excitation for feature refinement, and
   - a linear output head (single- or multi-output).

   Architecture hyperparameters can be sampled and tuned automatically when no
   `seed` is supplied.

2. **Motif attribution** — instead of inspecting filter weights directly, the
   contribution of each detected motif (and of multi-motif *combinations*) to
   the model's prediction is quantified with **Banzhaf indices** from
   cooperative game theory (via `BanzhafInference`). Significance is assessed
   against random-coalition backgrounds, producing filtered singleton motifs and
   higher-order interaction terms.

3. **Rendering** — results are written out as **HTML motif reports** (logos,
   contribution tables, and interaction summaries) via `GlyphEctoplasm`.

## Installation

This package depends on several unregistered packages referenced by **local
path** in `Project.toml` (`[sources]`): `BanzhafInference`, `GlyphEctoplasm`,
`AutoComputationalGraphTuning`, and `SEQ2EXPdata`. Those repositories must exist
at the expected paths before instantiating. A **CUDA-capable GPU** is required
for training.

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

The entry point is `run_method`, which runs the full pipeline (load → optional
tune → train → attribute → render).

```julia
using MotifInference

# Simplest — point at a .jld2 dataset. With no `seed`, hyperparameters are tuned.
run_method("/path/to/mydata.jld2")

# With overrides
run_method("/path/to/mydata.jld2"; seq_type=:rna, seed=42)

# Multi-output data — process only a subset of output targets
run_method("/path/to/mydata.jld2"; seq_type=:rna, seed=42, output_indices=1:5)
```

Build a run configuration explicitly if you want to inspect or reuse it:

```julia
trc = make_trc("/path/to/mydata.jld2"; seq_type=:dna, seed=42, motif_sizes=[2,3,4])
run_method(trc)
```

Batch over several registered datasets:

```julia
run_all(load_datasets("ecoli", "yeast"))
run_all(load_datasets())                      # everything
```

### Key options

Passed through `make_trc` / `run_method`:

| Option                 | Default            | Meaning                                                        |
| ---------------------- | ------------------ | -------------------------------------------------------------- |
| `seq_type`             | `:dna`             | `:dna`, `:rna`, or `:protein`                                  |
| `seed`                 | `nothing`          | RNG seed; `nothing` triggers hyperparameter tuning             |
| `motif_sizes`          | `[2, 3, 4, 5]`     | combination sizes to score for multi-motif interactions        |
| `normalization_method` | `:zscore`          | label normalization (`:identity`, `:zscore`, `:log`, …)        |
| `activation_thresh`    | `0.9`              | percentile threshold for calling a motif "active" in a seq     |
| `multioutput`          | `false`            | multi-output model variant                                     |
| `output_indices`       | all                | which output targets to process                                |
| `save_root`            | `"."`              | where the per-dataset results folder is created                |

Outputs (trained model, motif caches, and HTML reports) are written under
`save_root/<dataset name>/`.

## Project layout

```
src/
  MotifInference.jl          # module entry point
  pipeline.jl                # make_trc / run_method / run_all
  run_thru.jl                # train model, train processor, attribute motifs
  struct_def.jl              # training_and_rendering_config
  tuning_and_train_final.jl  # training + hyperparameter tuning
  datasets.jl                # registered dataset definitions
  dataset_utils.jl
  plotting.jl
  VeryBasicCNN2/             # the SeqCNN model (PWM + conv + MBConv backbone)
test/
  runtests.jl                # smoke tests (loads package, checks exports)
  integration_tests.jl       # data-dependent; requires MOTIFINFERENCE_TEST_DATA
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
