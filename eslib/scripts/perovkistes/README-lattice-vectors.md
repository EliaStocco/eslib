# Lattice-vector analysis for perovskite NpT campaigns

This three-step workflow keeps raw trajectory labels intact during the temporal
average and aligns only the resulting run-mean cells.  It is therefore suitable
for comparing symmetry-equivalent perovskite domains without introducing
frame-by-frame label-switching noise.

```bash
# 1. Extract native cell matrices from T=<temperature>K-run=<run> folders.
python analyze-lattice-vectors.py all_runs/supercell-10x10x10/std \
    -o all_lattice_vectors.csv

# 2. Discard equilibration, align run means, and create a temperature summary.
python align-lattice-vectors.py all_lattice_vectors.csv --discard 200 \
    --run-output lattice_run_means.csv \
    --summary-output lattice_temperature_summary.csv \
    --transform-output lattice_alignment_transforms.npz

# 3. Plot the summary; divide lengths by ten for a 10x10x10 supercell.
python plot-lattice-vectors.py lattice_temperature_summary.csv \
    --length-scale 10 -o lattice_parameters.png
```

In addition to the two CSV tables, the alignment step writes
`lattice_alignment_transforms.npz` and a matching CSV inspection table. There
is one fixed transform for each `(temperature_K, run)`: a signed lattice-basis
permutation `A` and a proper Cartesian rotation `R`. For a row-vector cell
matrix `H`, they satisfy `H_reference = A @ H @ R`. Cartesian row vectors
(positions, dipoles, forces, etc.) transform as `v_reference = v @ R`.

Apply a selected transform to a trajectory and a per-atom local-dipole array
named `local_dipole` with:

```bash
python apply-lattice-alignment.py \
    -i all_runs/supercell-10x10x10/std/T=100K-run=0/i-pi.positions_0.extxyz \
    -t lattice_alignment_transforms.npz --temperature 100 --run 0 \
    --vector-array local_dipole -o T=100K-run=0-aligned.extxyz
```

Use `--info-vector NAME` instead when a single frame-level vector, such as a
total dipole, is stored in `atoms.info[NAME]`. The transformation is selected
once from the run mean and reused for every frame, so it cannot introduce
frame-by-frame axis switching.

`align-lattice-vectors.py` permutes cell-vector labels together with their
associated angles. It also selects an equivalent signed basis with all acute
angles for a rhombohedral cell whenever possible. It preserves the conventional
orthorhombic representation when the angle between the two long axes is
observably obtuse while the other two angles are right angles (default
tolerance: 0.01 degrees). Vector signs are chosen only after each run's mean
cell has been formed; instantaneous frames are never relabelled.

The plotter's error bands quantify variation **between independent runs**. The
per-run temporal standard deviations are retained in both generated CSV files
as `*_fluctuation` columns.

## Aligned piezoelectric coefficients

For raw synchronized dipole--strain data in `piezo_raw_frames.csv`, first
align every run into the same Cartesian cell frame:

```bash
python align-piezo-raw-data.py piezo_raw_frames.csv \
    -o piezo_raw_frames_aligned.csv \
    --transform-output piezo_alignment_transforms.npz
```

This derives the fixed transform from each run's post-equilibration mean cell,
then applies it to all frames: cell matrices transform as `A @ H @ R`, total
dipoles as `M @ R`, and Green--Lagrange strain tensors as `R.T @ E @ R`.

Compute and plot the consistently aligned tensor with:

```bash
python compute-piezoelectric-coefficients.py piezo_raw_frames_aligned.csv \
    --supercell 10x10x10 \
    -o piezoelectric_coefficients_aligned.csv \
    --plot-output piezoelectric_coefficients_aligned.png
```

The coefficient CSV is in m/V (equivalently C/N); the plot displays pC/N with
between-run standard-error bands. Both steps use a 10% equilibration removal by
default, which must be kept consistent unless deliberately changed.

### Fixed-field and covariant estimators

For an explicit comparison between the fixed-Cartesian-field response and the
reference-frame covariant convention, run:

```bash
python compute-piezoelectric-covariant.py piezo_raw_frames_aligned.csv \
    --supercell 10x10x10 \
    -o piezoelectric_coefficients_covariant.csv
```

The script defines a post-equilibration reference cell for each run and
recomputes the Green--Lagrange strain from the cell trajectory. Its output has
both `fixed_cartesian` coefficients, `cov(M, eta)/(k_B T)`, and
`reference_covariant` coefficients, `cov(F^-1 M, eta)/(k_B T)`. The latter is
the finite-frame counterpart of the covariant proper-dipole derivative and
corresponds to a material/reference-frame electric field; it must not be
silently substituted for the fixed-Cartesian-field result.
