# MotifInference.jl

Discover functional motif combinations from sequence-to-measurement data
(DNA / RNA / protein).

A CUDA-capable GPU is required for training.

## Installation

```julia
using Pkg
Pkg.add("MotifInference")
```

## Quick start

Put your data in a two-column CSV (`sequence, measurement`), one row per
sequence:

```csv
sequence,expression
GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATT,1.5
TTGACAGCTAGCTCAGTCCTAGGTATTATGCTAGCTACTAGAGAAAGAGGA,2.0
CGTACGATCGATCGTAGCTAGCTAGCATCGATCGATCGTACGTAGCATCGA,-0.5
```

Then point [`run_method`](@ref run_method-reference) at it:

```julia
using MotifInference

run_method("/path/to/mydata.csv")
```

That is the whole pipeline: **load → tune → train → attribute → render**.

## Where results are saved

Results (trained model, motif caches, and HTML reports) are written to a folder
named after the CSV, created under `save_root` (the current directory by
default):

```
<save_root>/<csv basename>/
```

So `run_method("data/mydata.csv")` writes to `./mydata/`. Change the location
with `save_root` (parent directory) and/or `save_folder_name` (exact folder
name) — see the reference for details.

## Reference

For the complete, crystal-clear list of every argument `run_method` accepts —
grouped, with defaults, meanings, and copy-paste recipes — see:

- [`run_method` reference](@ref run_method-reference)
