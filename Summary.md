# eslib usage summary and replacement-package scope

## Basis of this analysis

This document summarizes the command history in the repository-local `.log/`
directory as inspected on 2026-09-29.  A record is written by the `esfmt`
decorator only when the `ESLIB_LOG` environment variable is enabled.  It
contains the script path, working directory, and command line, but no
per-invocation timestamp.

The logs contain 30,445 invocations in 30 files (about 15 MB).  This inventory
includes every script invoked at least 10 times: 74 scripts and 30,233
invocations (99.3% of the logged activity).  Counts are therefore a useful
measure of repeated workflows, not a complete measure of all package usage or
of recency.

## Invocation-day counts are not recoverable

It is not possible to count the number of distinct calendar days on which an
individual script was invoked from the current logs.  Each record has the same
five fields—script file, absolute script path, working directory, VS Code
arguments, and command line—and none includes a date or time.  A log file's
filesystem modification time only identifies the most recent append to that
*category-wide* file; it cannot date any of the preceding records or attribute
the final write to a particular script.

Do not infer dates from input/output filenames or working-directory names:
those values can contain arbitrary identifiers and would produce unreliable
results.  The invocation counts and number of distinct working directories in
this report are the useful available proxies for distinguishing a repeated
batch workflow from broad recurring use.

For future data, add an ISO-8601 timestamp (including timezone) to every log
record, for example `timestamp : 2026-09-29T19:31:30+02:00`.  With that one
field, the same analysis can report, per script: total invocations, distinct
days, first/last use, and invocations per active day.  This will distinguish a
one-day batch run from a tool used repeatedly across projects.

Some high counts are batch workflows rather than independent interactive
uses.  For example, `rename-field.py` was run 9,396 times but from only eight
working directories.  Conversely, the current `convert/convert-file.py` was
run in 128 directories and `information-and-primitive.py` in 58, which makes
them especially strong candidates for a general-purpose public API.

## Main conclusions

The replacement should be centered on atomic-structure and trajectory
manipulation, with ASE/extxyz as the first-class interchange format.  The
largest operational clusters are:

| Workflow cluster | Logged invocations | Share of all logs | Package implication |
| --- | ---: | ---: | --- |
| Trajectory operations | 18,058 | 59.3% | This is the core: metadata, conversion, merging, inspection and cleanup. |
| General conversion | 2,953 | 9.7% | Provide one well-tested conversion API and a single CLI with subcommands/options. |
| BEC parsing | 2,759 | 9.1% | Support VASP and Quantum ESPRESSO output via a shared parser/result model. |
| Structure inspection | 2,396 | 7.9% | Keep summaries, primitive-cell/symmetry information and property comparison close to the core. |
| FHI-aims tools | 1,596 | 5.2% | Make FHI-aims parsing/input helpers a focused integration module. |
| Builders | 883 | 2.9% | Supercells, cells and FHI-aims input generation are worthwhile early features. |

The first five clusters make up 91.2% of the logged commands.  The ten most
used individual commands account for 81.2%.

### Recommended initial package shape

```text
newpkg/
  structures/      # ASE I/O, extxyz fields, units, selection and validation
  trajectories/    # merge, align, clean, summarize and transform frames
  parsers/         # VASP, Quantum ESPRESSO and FHI-aims parsers
  inspection/      # structure summaries, primitive cells, comparisons
  builders/        # supercells and program-input generation
  cli/             # thin stable commands delegating to the library
```

`structures` should own a typed/validated representation of structure-level
metadata (`Atoms.info`) and per-atom fields (`Atoms.arrays`).  Functions such
as rename, remove, add, convert units and extract should operate on that shared
representation, rather than each CLI reimplementing file handling.  The
current most-used scripts already rely on `AtomicStructures`, which supports
this boundary.

### Delivery priorities

1. **Foundation and transformation CLI.**  ASE/extxyz I/O; field
   rename/remove/add/extract; format/unit conversion; structure merge;
   selection/subsampling; reliable in-place/output-file semantics.
2. **Inspection and construction.**  Trajectory/structure summaries,
   primitive-cell information, supercell generation, cell manipulation and
   FHI-aims input helpers.
3. **Electronic-structure parsers.**  VASP BEC, QE BEC/dielectric and
   FHI-aims Hirshfeld parsing behind a common result interface.
4. **Domain workflows.**  Water, descriptors, committee, polarization,
   phonopy and materials-specific analyses as optional modules/plugins.

The most repeated broad operations are metadata editing (12,158 invocations),
file conversion/output preparation (5,390), trajectory combination/alignment
(2,955), and electronic-structure parsing (4,332).  These groups overlap only
minimally and should drive the first API design and test matrix.

## Compatibility and migration observations

- `trajectories/convert-file.py` appears 2,481 times in the logs.  It was
  moved to `convert/convert-file.py` in February 2026, which has a further
  2,287 logged uses.  The new package should expose one canonical conversion
  command and, during migration, retain an alias for the old name.
- `inspect/trajectory-summary.py` has 976 logged uses but is no longer in the
  working tree; `trajectories/trajectory-summary.py` is the apparent successor
  and has 101 uses.  Preserve the behavior and consider an alias.
- Other removed historical scripts still cross the threshold:
  `inspect/compute-RMSE.py` (271), `inspect/compare-properties.py` (161), and
  `trajectories/properties2df.py` (12).  Their replacement needs should be
  resolved explicitly before dropping compatibility.
- Counts alone do not mean that every old command deserves a separate CLI.
  Several can become subcommands of one coherent interface, for example
  `trajectory fields rename`, `trajectory fields remove`, and `trajectory
  fields add`.

## Complete inventory: scripts invoked at least 10 times

The descriptions are concise functional summaries, based on each script's
name, parser arguments, and source description where available.  “Historical”
means the logged path is absent from the current working tree.

### Trajectories (18,030 qualifying invocations; 20 scripts)

| Runs | Script | Functional role | Status / planning note |
| ---: | --- | --- | --- |
| 9,396 | `trajectories/rename-field.py` | Rename an `info` or per-atom array field in an extxyz file. | Core transformation API; heavily batch-used. |
| 2,481 | `trajectories/convert-file.py` | Convert ASE-readable files, units and selected fields. | Historical path; moved to `convert/convert-file.py`. |
| 2,451 | `trajectories/merge-structures.py` | Combine two structures into one extxyz structure. | Core merge primitive. |
| 2,199 | `trajectories/remove-field.py` | Remove one or more `info`/array fields. | Core transformation API. |
| 453 | `trajectories/add-tag2extxyz.py` | Add structure-level tags from JSON data. | Core metadata API. |
| 301 | `trajectories/match-snapshots.py` | Match equivalent snapshots between trajectories. | Important alignment utility. |
| 140 | `trajectories/subsample.py` | Select trajectory frames by indices. | Core selection API. |
| 110 | `trajectories/add2extxyz.py` | Add data to an extxyz trajectory. | Core metadata API. |
| 101 | `trajectories/trajectory-summary.py` | Summarize an MD trajectory. | Current apparent successor to the historical inspect command. |
| 81 | `trajectories/extxyz2array.py` | Extract an `info`/array field into a text array. | Core export API. |
| 63 | `trajectories/merge-unequal-trajectories.py` | Merge trajectories after snapshot matching. | Build on matching/merge primitives. |
| 38 | `trajectories/reverse-indices.py` | Reverse a set of indices. | Small selection helper. |
| 37 | `trajectories/convert-quantity.py` | Convert a trajectory physical quantity between units. | Fold into shared unit support. |
| 36 | `trajectories/count-NaN.py` | Count frames containing NaN-valued properties. | Validation/quality-control API. |
| 33 | `trajectories/merge-datasets.py` | Add/merge a second dataset into a first. | Dataset-level merge API. |
| 31 | `trajectories/set-properties-weight.py` | Set weights for missing properties using JSON metadata. | Dataset preparation utility. |
| 26 | `trajectories/properties2json.py` | Create JSON metadata/summary for trajectory properties. | Pair with validation utilities. |
| 23 | `trajectories/split-nan.py` | Split frames with/without NaN values. | Validation/cleanup workflow. |
| 18 | `trajectories/clean-trajectory.py` | Remove all info and arrays from a trajectory. | A convenient preset over field removal. |
| 12 | `trajectories/properties2df.py` | Export trajectory properties to a data frame. | Historical path; decide whether tabular export is core or an optional integration. |

### Conversion and external-format preparation (2,939 qualifying invocations; 6 scripts)

| Runs | Script | Functional role | Status / planning note |
| ---: | --- | --- | --- |
| 2,287 | `convert/convert-file.py` | ASE-based format conversion, unit conversion and field cleanup. | Broadest conversion tool; canonical replacement CLI/API. |
| 215 | `convert/fix-bonds.py` | Repair bonding/connectivity in converted structures. | Optional structure-normalization feature. |
| 182 | `convert/extxyz2espresso-in.py` | Create Quantum ESPRESSO input from extxyz data. | External-code writer. |
| 154 | `convert/extxyz2aims-output.py` | Convert extxyz data to an FHI-aims-style output representation. | Keep with FHI-aims integration. |
| 71 | `convert/fold.py` | Fold periodic coordinates into the unit cell. | Core geometry operation. |
| 30 | `convert/deepmd2extxyz.py` | Convert DeepMD data into extxyz. | Optional external-format adapter. |

### Electronic-structure parsers (4,343 qualifying invocations; 4 scripts)

| Runs | Script | Functional role | Status / planning note |
| ---: | --- | --- | --- |
| 2,148 | `bec/vasp2bec.py` | Read BECs and related quantities from VASP `OUTCAR`. | First parser priority. |
| 1,585 | `aims/aims-read-hirshfeld.py` | Extract FHI-aims Hirshfeld results to extxyz. | First parser priority; batch-oriented. |
| 599 | `bec/qe2bec.py` | Read BECs and dielectric data from QE `ph.x` output. | Share property schema with VASP parser. |
| 11 | `aims/aims-read-forces-wo-D3.py` | Read FHI-aims forces excluding D3 contribution. | Specialized FHI-aims extension. |

### Inspection (2,365 qualifying invocations; 6 scripts)

| Runs | Script | Functional role | Status / planning note |
| ---: | --- | --- | --- |
| 976 | `inspect/trajectory-summary.py` | Display trajectory size, composition and structural information. | Historical; preserve through current trajectory-summary functionality. |
| 605 | `inspect/information-and-primitive.py` | Show structure information and find a primitive cell. | General inspection API; broad project use. |
| 314 | `inspect/homo-lumo.py` | Inspect HOMO/LUMO information. | Electronic-structure inspection extension. |
| 271 | `inspect/compute-RMSE.py` | Compute an RMSE between properties/structures. | Historical; likely merge with metrics/comparison module. |
| 161 | `inspect/compare-properties.py` | Compare properties across structures/trajectories. | Historical; merge with metrics/comparison module. |
| 38 | `inspect/set-com.py` | Set or inspect center-of-mass-related geometry. | Small geometry utility. |

### Builders (869 qualifying invocations; 4 scripts)

| Runs | Script | Functional role | Status / planning note |
| ---: | --- | --- | --- |
| 391 | `build/make-supercell.py` | Generate supercells using a transformation matrix. | Core builder. |
| 391 | `build/create-fhi-aims.py` | Create FHI-aims `species.in` input. | Keep with FHI-aims support. |
| 77 | `build/set-cell.py` | Set/change a structure cell. | Core geometry operation. |
| 10 | `build/create-path-with-cell.py` | Build an interpolation/path with cell information. | Optional path-building feature. |

### Water workflows (661 qualifying invocations; 8 scripts)

| Runs | Script | Functional role | Status / planning note |
| ---: | --- | --- | --- |
| 166 | `water/analyse-hirshfeld.py` | Analyze Hirshfeld data for water systems. | Domain module built over FHI-aims parser. |
| 156 | `water/divide-into-water-molecules.py` | Identify/group atoms into water molecules. | Reusable molecule-grouping primitive. |
| 145 | `water/analyse-hessian.py` | Analyze water-system Hessians. | Domain analysis. |
| 81 | `water/sort-water-molecules.py` | Order water molecules consistently. | Build on molecule grouping. |
| 45 | `water/analyse-hessian-per-molecule.py` | Analyze Hessians molecule by molecule. | Domain analysis. |
| 37 | `water/displace-2D-ice.py` | Displace a two-dimensional ice structure. | Specialized workflow. |
| 16 | `water/prettify-water-trajectory.py` | Normalize/reorder a water trajectory. | Build on molecule grouping/sorting. |
| 15 | `water/aggregate-into-molecules.py` | Aggregate atom-level data into molecular data. | Reusable grouped-property reduction. |

### Descriptors, plots and optimization (412 qualifying invocations; 7 scripts)

| Runs | Script | Functional role | Status / planning note |
| ---: | --- | --- | --- |
| 115 | `descriptors/soap.py` | Compute SOAP descriptors. | Optional descriptor module. |
| 71 | `committee/compute-committee-disagreement.py` | Quantify committee/model disagreement. | ML workflow extension. |
| 57 | `plot/plot-time-series.py` | Plot time-series data. | Keep plotting separate from core transforms. |
| 56 | `descriptors/test-featomic.py` | Test/calculate featomic descriptors. | Optional descriptor backend integration. |
| 51 | `ase_scripts/optimize.py` | Run ASE structure optimization. | Optional execution workflow. |
| 36 | `plot/plot-cellpar.py` | Plot cell parameters. | Plotting extension. |
| 26 | `descriptors/fps.py` | Farthest-point sampling for descriptors. | Descriptor/dataset-selection extension. |

### Specialized scientific workflows (614 qualifying invocations; 19 scripts)

| Runs | Script | Functional role | Status / planning note |
| ---: | --- | --- | --- |
| 84 | `cube/hartree2polarization.py` | Convert Hartree/cube data to polarization-related data. | Specialized electrostatics module. |
| 154 | `charges/add-charges2extxyz.py` | Add charge data to an extxyz trajectory. | Reusable property-enrichment operation; place near the core metadata API. |
| 45 | `phonopy/get-path.py` | Generate/get a phonon band path. | Optional phonopy integration. |
| 45 | `2D-ice/cut-ice-structure-along-xy.py` | Cut an ice structure in the xy plane. | Water/ice specialty workflow. |
| 35 | `piezo/displace-lattice-vectors.py` | Apply lattice-vector displacements. | Response-property utility. |
| 18 | `unit_cell_dipoles/cellpars2txt.py` | Export unit-cell dipole/cell parameters. | Specialized analysis. |
| 18 | `properties/prop2txt.py` | Export a named property to text. | Could join generic property-export API. |
| 17 | `ferroelectrics/hysteresis_loop.py` | Analyze or construct a ferroelectric hysteresis loop. | Domain module. |
| 16 | `nn/create-dataset.bis.py` | Create a neural-network dataset. | ML/dataset workflow. |
| 16 | `dipole/fix-dipole-with-reference-data.py` | Correct dipoles using reference data. | Specialized polarization workflow. |
| 15 | `rmse/compute-RMSE.py` | Compute RMSE metrics. | Consolidate with the historical inspect RMSE command. |
| 15 | `perovkistes/plot-perovskite-AB-vector-histogram.py` | Plot perovskite A–B vector distributions. | Materials-specific plotting. |
| 15 | `matrix/compare-matrices.py` | Compare numerical matrices. | General but low-volume utility. |
| 15 | `analysis/gtrdf.py` | Compute/plot a generalized time-resolved RDF. | Domain analysis. |
| 14 | `symmetry/find-symmetry.py` | Find structure symmetry. | Could share a symmetry service with primitive-cell inspection. |
| 11 | `perovkistes/divide-perovskite-into-unitcells.py` | Divide a perovskite structure into unit cells. | Materials-specific workflow. |
| 11 | `dipole/compute-quanta.py` | Compute dipole polarization quanta. | Specialized polarization workflow. |
| 10 | `water+graphene/prettify-confined-water.py` | Normalize a confined-water trajectory. | Water/graphene specialty workflow. |
| 60 | `eslib-help.py` | Search/display package help. | The replacement should provide discoverable CLI help and examples. |

## What to test first

The highest-value regression fixtures should exercise the workflows that occur
repeatedly in the logs:

1. An extxyz trajectory with both `info` and per-atom arrays, transformed by
   rename, remove, add, extract, clean and unit conversion operations.
2. ASE format round trips and output writers for extxyz, FHI-aims and Quantum
   ESPRESSO, including cell/PBC and unit preservation.
3. Two-structure and two-trajectory merge/match cases, including unequal
   lengths and missing/NaN properties.
4. Representative VASP `OUTCAR`, QE `ph.x`, and FHI-aims output fixtures with
   standardized parsed property names and shapes.
5. Primitive-cell/symmetry and supercell operations that preserve atom order,
   metadata and periodicity.

This order follows observed use rather than the current directory layout, and
should keep the replacement small, composable and compatible with the workflows
that matter most.
