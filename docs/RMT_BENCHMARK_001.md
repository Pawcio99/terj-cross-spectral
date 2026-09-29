# RMT benchmark 001

Date: 2026-09-29

## Hardware

CPU: Intel Core i7-1165G7
NumPy: 2.3.5

## Purpose

This benchmark was performed before RMT production in order to determine
the computational cost of the initial dense control generators.

No scientific cross-domain inference was performed.

## Approximate wall times for n=1024

- GUE: 0.53 s
- GOE: 0.44 s
- real Wishart: 0.30 s
- CUE: 5.45 s
- Poisson: 0.16 s

CUE is substantially more expensive than the other controls.

## Methodological issue discovered before production

The benchmark parameter n currently denotes matrix dimension in the
initial RMT generator.

This must not be confused with the number of spectral levels entering
the common feature extractor.

The production implementation must therefore distinguish:

- matrix_size
- analysis_size
- bulk_slice
- network_width
- ensemble_role

## Neural / Wishart matching

For a neural network of width N, Stage A analyzes the central 50 percent
of the Gram spectrum.

Therefore:

    matrix_size = N
    analysis_size = N / 2
    bulk_slice = eig[N//4 : 3*N//4]

Required mapping:

    Neural width 96   -> 48 analyzed levels
    Neural width 256  -> 128 analyzed levels
    Neural width 512  -> 256 analyzed levels
    Neural width 1024 -> 512 analyzed levels
    Neural width 2048 -> 1024 analyzed levels

The matched real Wishart null must be generated at the full matrix
dimension N and reduced to the same central spectral window.

A 48 x 48 Wishart matrix is therefore NOT the matched null for the
48-level central window of a width-96 neural network.

## Riemann / GUE matching

Riemann windows represent local bulk statistics.

The GUE reference must therefore not mix spectral edges with the bulk.

The production generator must generate a sufficiently larger GUE
spectrum and extract a central bulk window containing exactly the same
number of analyzed levels as the corresponding Riemann window.

Required analysis sizes:

    48
    128
    256
    512
    1024

## Ensemble roles

Primary Riemann null:
    GUE beta=2

Riemann universality / sensitivity control:
    CUE beta=2

Primary Neural null:
    real Wishart/Laguerre beta=1

Beta=1 universality control:
    GOE beta=1

Non-repulsive negative control:
    Poisson

## Production rule

The benchmark generator must not be used for final RMT production until
the matrix-size / analysis-size distinction and bulk extraction are
implemented and tested.

No RMT production results were inspected before this correction.
