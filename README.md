# MotifInference.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kchu25.github.io/MotifInference.jl/dev/)

Discover functional motif combinations from sequence-to-measurement data (DNA / RNA).

📖 **Documentation:** [kchu25.github.io/MotifInference.jl](https://kchu25.github.io/MotifInference.jl/dev/) — including the full [`run_method` argument reference](https://kchu25.github.io/MotifInference.jl/dev/run_method/).

## Installation

```julia
using Pkg
Pkg.add("MotifInference")
```

A CUDA-capable GPU is required for training.

## Usage

Put your data in a CSV with two columns (`sequence, expression`), one row per
sequence:

```csv
sequence,expression
GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATT,1.5
TTGACAGCTAGCTCAGTCCTAGGTATTATGCTAGCTACTAGAGAAAGAGGA,2.0
CGTACGATCGATCGTAGCTAGCTAGCATCGATCGATCGTACGTAGCATCGA,-0.5
AATTGGCCAATTCCGGAATTCCGGTTAACCGGTTAACCGGAATTCCGGAAT,0.8
TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC,1.2
```

Then point `run_method` at it:

```julia
using MotifInference

run_method("/path/to/mydata.csv")
```

That's the whole pipeline: load, tune, train, attribute, render.

### Where results are saved

Results (trained model, motif caches, and HTML reports) are written to a folder
named after the CSV, created under `save_root` (the current directory by
default):

```
<save_root>/<csv basename>/
```

So `run_method("data/mydata.csv")` writes to `./mydata/`. Control the location
with two keyword arguments:

```julia
# choose the parent directory
run_method("data/mydata.csv"; save_root="/path/to/results")   # -> /path/to/results/mydata/

# choose the folder name explicitly
run_method("data/mydata.csv"; save_folder_name="run01")        # -> ./run01/
```

### Options

```julia
# RNA sequences, fixed seed (skips hyperparameter tuning)
run_method("/path/to/mydata.csv"; seq_type=:rna, seed=42)
```

| Option      | Default     | Meaning                                            |
| ----------- | ----------- | -------------------------------------------------- |
| `seq_type`  | `:dna`      | `:dna` or `:rna`                                   |
| `seed`      | `nothing`   | RNG seed; `nothing` triggers hyperparameter tuning |
| `save_root` | `"."`       | where the results folder is created                |

A header row is auto-detected and skipped; blank lines are ignored. Sequences of
differing length are padded to the max length before encoding.
