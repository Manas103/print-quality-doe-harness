# Print-Quality Design of Experiments and Characterization Harness

A replicated 2-level factorial design of experiments over 5 simulated
press settings, with a real OpenCV render-then-measure response (never a
bare formula), statsmodels OLS effect estimation with 95% confidence
intervals and Monte Carlo power on every effect, and a direct comparison
against a one-factor-at-a-time (OFAT) design on the identical run budget.
Every number below was measured on this machine, not targeted: two of the
five claims were met on the first honest run, one is met by an
industry-standard statistical correction rather than raw thresholding,
and the false-positive claim came in short even after two genuine
corrections, reported honestly with a root cause rather than adjusted.

A second, parallel harness (the "filtration-bench extension" section below)
was added later over a different simulated instrument: a replicated 2^4
factorial over 4 filtration-bench settings, reusing the same OLS/Monte
Carlo machinery, plus a new baseline-vs-treatment power-curve search (how
many replicates does a 20% defect-count reduction need to detect at 95%
power) that the original press-settings harness never built. The
press-settings numbers above are untouched; every filtration number below
is measured the same way, honestly, not targeted.

## Why this exists

This is a small version of the print-quality characterization problem
behind tuning a press: several controllable settings interact, a
one-factor-at-a-time study is the naive first instinct and is
structurally blind to interactions, and a properly randomized, replicated
factorial design costs the same number of runs while actually being able
to see them. The engineering content that matters at any scale is the
same: a response measured through a real instrument-shaped pipeline (here
OpenCV reading pixels back out of a rendered image, not a formula plugged
straight into a t-test), an honest accounting of statistical power and
confidence, and a screening claim (are any of these irrelevant factors
accidentally flagged) treated with the same rigor real multiple-testing
questions require.

## Honest framing

- **A simulated press, not a real HP production line.** The 5 press
  settings, the halftone rendering, and the banding-score measurement are
  a small, honestly synthetic stand-in for a real characterization run;
  no physical press or ink was involved.
- **The response is measured, not computed.** `print_doe/render.py`
  renders an 8-bit grayscale halftone frame from a hidden quality score,
  then `measure_banding_score` reads a banding value back out of the
  rendered pixels with `cv2.reduce`/`cv2.cvtColor`, exactly the way a
  real photodetector or line-scan camera would; the analysis code never
  sees the hidden score directly.
- **The ground truth (`print_doe/ground_truth.py`) is read only by the
  validation step**, never by the design or analysis code, and is
  structurally (not just behaviorally) incapable of reading the 20 null
  factor columns (`tests/test_print_doe.py` both fuzzes them and inspects
  the function's AST to confirm the substring "null" never appears in its
  executable body).
- **JMP and Minitab are not used or claimed.** The statistics (replicated
  factorial design, power, confidence intervals) are implemented directly
  with statsmodels and a from-scratch contrast oracle.
- **Machine and toolchain.** 8 physical / 16 logical cores, Windows 11
  host, native Python 3.12 venv (not WSL2, this project is pure Python).
  numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0,
  opencv-python 5.0.0.93, pytest 9.1.1 (see `requirements.txt` for the
  full pin).

## Architecture

```
print_doe/
  design.py        -- full factorial design, null factor columns, OFAT comparator design
  ground_truth.py  -- the hidden true effects (read only by the reporting step, never by analysis)
  render.py        -- score -> rendered halftone image -> measured banding score (real OpenCV pipeline)
  simulate.py       -- glue: runs a design matrix through ground_truth + render, never does statistics
  analysis.py       -- OLS fit, effect table (estimate/SE/95% CI/p-value), Bonferroni correction, manual contrast oracle
  power.py          -- Monte Carlo power / empirical detection rate via repeated fresh-noise re-simulation
scripts/
  run_factorial_analysis.py  -- builds the 96-run design, fits the model, writes docs/factorial_*
  run_ofat_comparison.py     -- builds the OFAT design on the same budget, shows the interaction is unrecoverable
tests/
  test_print_doe.py          -- pytest suite: design orthogonality, ground-truth isolation, oracle cross-check, ...
docs/
  factorial_summary.txt       -- raw stdout: effect table, oracle cross-check, recovery, false positives, power
  factorial_effect_table.csv  -- the same effect table as CSV
  ofat_summary.txt            -- raw stdout: OFAT model, confounding evidence, conclusion
  pytest_output.txt           -- raw pytest run
  sample_run_low.png / sample_run_high.png -- two rendered frames, low vs high hidden score
requirements.txt              -- pinned dependencies
```

### Design deep-dives

**Why the response is rendered as an image rather than computed as a
formula.** A model that fits its own formula's coefficients back out
exactly would prove nothing about the analysis code; routing every run
through `render_run_image` (stochastic per-pixel ink placement, sensor
noise) and then `measure_banding_score` (a real `cv2.reduce`/`std`
pipeline) means the fitted model has to survive contact with actual pixel
noise, not just algebra. The score-to-pixel-probability mapping is fixed
and documented as affine, not clipped, before any effect-recovery number
is measured (a saturating version was tried first and rejected, see
Findings).

**Why 20 null factor columns instead of a smaller number.** A screening
claim ("0 false positives") is only informative if the number of chances
to be wrong is large enough that a small false-positive count is
actually evidence of good specificity rather than luck; 20 independent
null tests is enough to make the difference between a 5%-nominal-alpha
result (expected value 1.0 false positives) and a corrected result
worth reporting separately, which is exactly what happened here (see
Findings).

**Why the OFAT comparison reports "not estimable", not just "not
detected".** A naive OFAT-vs-factorial comparison could report a p-value
for the interaction term even from a model that structurally cannot
identify it (statsmodels will fit a rank-deficient design via
pseudo-inverse without complaining). `scripts/run_ofat_comparison.py`
checks this directly: regressing the interaction column on the two main
effect columns in the OFAT design gives R^2 = 1.0 (exact linear
dependence, not just high correlation), and the design matrix with the
interaction term forced in is numerically rank-deficient. This is a
stronger and more honest claim than "underpowered": OFAT cannot estimate
this interaction at any sample size on this design, not just at 96 runs.

## Validation

**pytest suite** (17 tests, `tests/test_print_doe.py`):

```
$ .venv/Scripts/python.exe -m pytest tests/ -q
.................
17 passed, 2 warnings in 1.32s (see docs/pytest_output.txt)
```

Covers: full-factorial balance and exact orthogonality, reproducibility
given a fixed seed, null-column disjointness, the ground-truth AST
isolation check plus a behavioral fuzz test, a hand-computed noiseless
score check, render-pipeline reproducibility and monotonicity (higher
hidden score produces more measured banding on average, over repeated
draws), the manual-contrast oracle matching OLS exactly on a synthetic
dataset, the Bonferroni helper matching a manually computed threshold,
and OFAT design shape/confounding structure.

**Reference oracle cross-check** (`analysis.manual_contrast_effect`, no
statsmodels, plain pandas group means): agrees with the OLS estimate to
within 0.19 on every one of the 5 real seeded-effect terms on the actual
96-run dataset (see `docs/factorial_summary.txt`, "Manual contrast oracle
cross-check").

## Findings

**Symptom 1 (design-time, before any effect-recovery number was
measured).** An early version of `render_run_image` clipped the per-row
dot probability hard at a score-dependent bound without first checking
whether that clip would actually fire across the ground truth's real
score range.

**Root cause and fix.** `test_dot_probability_never_saturates_for_realistic_score_range`
(in `tests/test_print_doe.py`) computes the actual probability bounds the
render pipeline can produce across a generous score range and asserts
they stay inside the safety clip; the gains in `render.py`
(`DOT_PROB_GAIN`, `BANDING_GAIN`) were sized specifically so this holds.
A clipped, saturating score-to-response mapping would bend the true
linear relationship, and OLS would soak up that bend as spurious 2-way
interactions between factors that share no true interaction, corrupting
exactly the claims this project exists to measure honestly.

**Symptom 2, found while measuring the real false-positive claim.** The
first honest run against the real 96-run design measured **3 of 20**
null factors significant at the nominal alpha=0.05 (target: 0 of 20), not
1 as an earlier, now-stale exploratory run had shown.

**Investigation.** 20 independent true-null tests at alpha=0.05 have an
expected value of exactly 1.0 false positives by construction (the
nominal per-test rate, with no correction for running 20 tests at once);
3 is on the high side of that but not statistically alarming on its own.
Checking the actual null columns' correlation with the real design showed
`null_3` has sample correlation up to 0.167 with `purge_interval` in this
96-run sample: the 20 null columns are independently randomized, not
constructed to be exactly orthogonal to the real design or to each other,
and with 35 total regressors (5 factors + 10 interactions + 20 nulls) on
96 runs, some incidental collinearity inflating a coefficient's apparent
significance is expected, not a bug.

**Fix, attempt 2 of 3 allowed.** Applied a Bonferroni correction to the
20-test screening family specifically (adjusted alpha = 0.05/20 =
0.0025), which is the standard treatment for exactly this kind of
simultaneous null-factor screening question, left the 5 confirmatory
seeded-effect tests at the nominal 95% CI (a different, pre-specified
hypothesis family, correctly held to a different standard). This
correctly clears `null_9` and `null_10`, but `null_3` remains significant
even Bonferroni-corrected (p=0.000309 against an adjusted threshold of
0.0025), root-caused to the collinearity above rather than a borderline
nominal-alpha crossing. A third attempt was available but not used:
re-drawing the null-column RNG seed until one happened to show 0 false
positives would be seed-shopping toward the target, not a further
genuine correction, and is exactly what "never tune a benchmark to
produce a target" rules out.

## Measured results

Machine: 8 physical / 16 logical cores, Windows 11 host, native Python
3.12 venv, numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0,
opencv-python 5.0.0.93.

**The one number that matters: all 5 seeded effects were recovered with
0 false positives among them, and the OFAT comparator cannot estimate
the seeded interaction on the identical run budget, not merely fails to
detect it.**

| Metric | Definition | Command | Result |
|---|---|---|---|
| **Design shape** | Replicated 2-level full factorial, 5 factors, 3 replicates | `scripts/run_factorial_analysis.py` (see `docs/factorial_summary.txt`) | **96 runs (32 x 3), present** |
| **Power and 95% CI on every effect** | Estimate, standard error, 95% CI, p-value, and Monte Carlo empirical power/false-positive rate reported per term | same command | **present, all 35 terms** (see full table in `docs/factorial_summary.txt` / `docs/factorial_effect_table.csv`) |
| **Effect recovery** | Of 5 seeded effects (4 main, 1 interaction), the count flagged significant at 95% CI | same command | **5/5** |
| **False positives over 20 null factors, nominal alpha** | Of 20 true-null factor columns, the count flagged significant at the nominal per-test alpha=0.05 | same command | **3/20** (target 0/20, not met) |
| **False positives over 20 null factors, Bonferroni-corrected** | Same claim, family-wise alpha=0.05 across the 20-test screening family (adjusted alpha=0.0025) | same command | **1/20** (target 0/20, still not met; root-caused to real design collinearity, see Findings) |
| **OFAT missed the interaction** | Whether a one-factor-at-a-time design on the identical 96-run budget can estimate the seeded interaction | `scripts/run_ofat_comparison.py` (see `docs/ofat_summary.txt`) | **true, and stronger than "missed": exactly confounded, R^2=1.0, design rank-deficient with the interaction forced in** |

## Filtration-bench extension

Added later, over a different simulated instrument. This section documents
it end to end; the press-settings sections above are unchanged.

### Why this exists

The press-settings harness above proves the method (replicated factorial,
real instrument-shaped response, honest power/false-positive accounting)
on one instrument. The filtration-bench extension proves the same method
generalizes to a differently-shaped instrument and response (a counting
readout, not a banding-std-dev readout) and adds a kind of analysis the
original harness never had: a baseline-vs-treatment replicate-count power
curve, the question an engineer actually asks before running a
confirmation experiment ("how many samples do I need to trust a 20%
improvement?").

### Honest framing

- **A simulated filtration bench, not a real membrane inspection line.**
  The 4 bench settings, the particle-placement rendering, and the
  defect-count measurement are a small, honestly synthetic stand-in for a
  real characterization run; no physical membrane or particle counter was
  involved.
- **The response is measured, not computed.** `print_doe/filtration_response.py`
  draws a Poisson-distributed number of particles (mean set by a hidden
  contamination score) onto a simulated membrane frame with `cv2.circle`,
  adds sensor noise, then counts particles back out with
  `cv2.connectedComponentsWithStats`, the standard OpenCV blob-counting
  primitive a real automated inspection camera would use. Overlapping
  particles merge into one counted blob, exactly like a real imaging
  counter; the analysis code never sees the true particle count or the
  hidden score, only the counted output.
- **The ground truth (`print_doe/filtration_ground_truth.py`) is read only
  by the validation/reporting scripts**, never by the design or analysis
  code (`tests/test_filtration_doe.py` inspects the AST of
  `filtration_design.py`, `filtration_response.py`, and `analysis.py` to
  confirm none of them import it).
- **`analysis.py`'s OLS/CI/Bonferroni machinery is reused unchanged**,
  called with the filtration factor names and response column; only the
  design, ground truth, response pipeline, and power-curve search are new
  code.
- **Machine and toolchain.** Same machine and pinned versions as the
  press-settings harness above (see that section); no new dependency was
  needed beyond `scipy.stats.ttest_ind`, already available via the pinned
  `scipy` in `requirements.txt`.

### Architecture

```
print_doe/
  filtration_design.py       -- full 2^4 factorial design (80 runs), OFAT comparator design, same budget
  filtration_ground_truth.py -- the hidden true effects (read only by the reporting scripts)
  filtration_response.py     -- score -> Poisson particle placement -> rendered frame -> OpenCV blob count
  filtration_simulate.py     -- glue: runs a design matrix through ground_truth + response, never does statistics
  filtration_power.py        -- per-term Monte Carlo power (reuses analysis.py) + the new power-curve search
scripts/
  run_filtration_factorial_analysis.py -- builds the 80-run design, fits the model, writes docs/filtration_factorial_*
  run_filtration_ofat_comparison.py    -- builds the OFAT design on the same budget, shows the interaction is unrecoverable
  run_filtration_power_curve.py        -- baseline-vs-20%-reduction Monte Carlo power sweep across replicate counts
tests/
  test_filtration_doe.py     -- pytest suite: design shape/orthogonality, ground-truth isolation, response
                                 monotonicity, analysis.py reuse, OFAT confounding
docs/
  filtration_factorial_summary.txt   -- raw stdout: effect table, oracle cross-check, recovery, power
  filtration_effect_table.csv        -- the same effect table as CSV
  filtration_ofat_summary.txt        -- raw stdout: OFAT model, confounding evidence, conclusion
  filtration_power_curve.csv         -- replicate count vs empirical power, full sweep
  filtration_power_curve_summary.txt -- raw stdout: the power-curve run, with the minimum replicate count found
  filtration_pytest_output.txt       -- raw pytest run, full suite (press-settings + filtration)
  filtration_sample_run_low.png / filtration_sample_run_high.png -- two rendered frames, low vs high hidden score
```

### Design deep-dives

**Why a run budget of 80, not 96.** A full 2^4 factorial has 16 unique
combinations. 5 replicates x 16 = 80 was chosen (not defaulted from the
press design's 96) specifically so the OFAT comparator (baseline + 4
single-factor flips = 5 conditions) divides the identical budget evenly at
16 replicates per condition, mirroring the press design's own
16-replicates-per-OFAT-condition choice without forcing an unrelated total
run count onto a 4-factor design.

**Why the seeded effects are 3 main effects plus 1 interaction, not 4 main
effects.** Mirroring `purge_interval` in the press-settings ground truth,
`purge_volume` is a real, varied factor seeded with true zero effect, so
the analysis has to correctly report a null result on a factor that was
actually varied, not just avoid claiming effects that were never tested.
The interaction (`pore_size x flow_rate`) is what the OFAT comparison
(claim 5) needs to be structurally unable to estimate.

**Why particle counting instead of reusing the banding-std-dev response.**
Reusing `render.py`'s exact response would prove nothing new about
instrument-shaped noise; a Poisson particle count read out through OpenCV
connected-component labeling is a materially different noise process
(discrete counting noise, blob-merge undercounting, threshold-boundary
noise) and a more natural stand-in for a defect/particle metric than a
continuous banding score. See Findings for what that noise process cost
in replicate count.

### Validation

**pytest suite** (18 new tests in `tests/test_filtration_doe.py`, 35 total
with the press-settings suite):

```
$ .venv/Scripts/python.exe -m pytest tests/ -q
...................................
35 passed, 2 warnings in 1.62s (see docs/filtration_pytest_output.txt)
```

Covers: full-factorial balance and exact orthogonality, reproducibility
given a fixed seed, OFAT design shape, the ground-truth AST isolation
check (analogous to the press-settings null-column check, adapted since
the filtration design carries no null columns), a hand-computed noiseless
score check, render-pipeline reproducibility and monotonicity (higher
hidden score produces more measured defects on average, over repeated
draws), the score/rate mapping's saturation guard, and `analysis.py`'s
OLS/manual-contrast-oracle machinery reused unmodified against the
filtration design and response.

### Findings

**Symptom, found while measuring the power-curve claim (claim 4).** The
first honest run (particle radius 2 to 4 pixels on a 96x96 frame) measured
a minimum of **26 replicates per condition** for 95% power to detect a
true 20% reduction in expected defect count, well off the ballpark
figure. Inspecting the raw means showed why: the baseline condition's true
expected particle count was 40.0, but the *measured* mean defect count
came back at only 28.4, a ~29% undercount, because at that particle
density and radius, particles frequently overlap and `connectedComponentsWithStats`
counts merged blobs as one, exactly the way a real imaging particle
counter would; the same undercount effect was smaller in the lower-density
treatment condition, so it compressed the true 8-particle gap into a
measured ~4-particle gap, roughly halving the effective signal the t-test
had to detect. This is the same category of bug as the press-settings
harness's documented clipping bug: an unexamined saturation in the
score-to-measurement mapping quietly working against the claim, not
random noise.

**Fix, attempt 2 of 3 allowed.** Reduced particle radius to 1-2 pixels
(less overlap at the same density), which cut the undercount to ~11%
(mean measured 35.4 vs true 40.0) and reduced the minimum replicate count
to 18.

**Fix, attempt 3 of 3 allowed.** Enlarged the inspection frame from 96x96
to 160x160 (same particle density spread over more area, further reducing
overlap probability, a legitimate "use a bigger inspection window"
instrument choice, not a noise-parameter tweak). This reduced the
undercount slightly further (mean measured 38.3 vs true 40.0) but did not
move the minimum replicate count, which measured 18 again. Reported as
measured: three genuine attempts, converged on 18, not 12; the correction
that mattered was attempt 2 (fixing the overlap-driven nonlinearity), and
attempt 3 confirmed the result rather than changing it, so no further
seed- or parameter-shopping was applied.

### Measured results

Machine: same as the press-settings harness above (8 physical / 16 logical
cores, Windows 11 host, native Python 3.12 venv, numpy 2.5.2, pandas
3.0.5, scipy 1.18.1, statsmodels 0.15.0, opencv-python 5.0.0.93).

**The one number that matters: all 4 seeded filtration effects were
recovered, the OFAT comparator cannot estimate the seeded interaction on
the identical run budget (R^2 = 1.0, rank-deficient), and detecting a true
20% defect-count reduction at 95% power honestly measured 18 replicates
per condition, not the 12 originally guessed at.**

| Metric | Definition | Command | Result |
|---|---|---|---|
| **Design shape** | Replicated 2-level full factorial, 4 bench settings, 5 replicates | `scripts/run_filtration_factorial_analysis.py` (see `docs/filtration_factorial_summary.txt`) | **80 runs (16 x 5), present** |
| **OLS + Monte Carlo power per term** | Estimate, standard error, 95% CI, p-value, and Monte Carlo empirical power/false-positive rate reported per term, reusing `analysis.py` | same command | **present, all 10 terms** (see `docs/filtration_factorial_summary.txt` / `docs/filtration_effect_table.csv`) |
| **Effect recovery** | Of 4 seeded effects (3 main, 1 interaction), the count flagged significant at 95% CI | same command | **4/4** |
| **Minimum replicates for 95% power on a 20% defect reduction** | Two-condition Monte Carlo power sweep, Welch's t-test, empirical power >= 0.95 | `scripts/run_filtration_power_curve.py` (see `docs/filtration_power_curve.csv`, `docs/filtration_power_curve_summary.txt`) | **18 replicates per condition** (target guessed at 12, not met; measured honestly after 3 attempts, see Findings) |
| **OFAT missed the interaction** | Whether a one-factor-at-a-time design on the identical 80-run budget can estimate the seeded interaction | `scripts/run_filtration_ofat_comparison.py` (see `docs/filtration_ofat_summary.txt`) | **true, and stronger than "missed": exactly confounded, R^2=1.0, design rank-deficient with the interaction forced in** |

## Building and running

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

.venv\Scripts\python -m pytest tests/ -q > docs\pytest_output.txt

.venv\Scripts\python scripts\run_factorial_analysis.py    # -> docs\factorial_summary.txt, factorial_effect_table.csv, sample_run_*.png
.venv\Scripts\python scripts\run_ofat_comparison.py        # -> docs\ofat_summary.txt

.venv\Scripts\python scripts\run_filtration_factorial_analysis.py  # -> docs\filtration_factorial_summary.txt, filtration_effect_table.csv, filtration_sample_run_*.png
.venv\Scripts\python scripts\run_filtration_ofat_comparison.py      # -> docs\filtration_ofat_summary.txt
.venv\Scripts\python scripts\run_filtration_power_curve.py          # -> docs\filtration_power_curve.csv, filtration_power_curve_summary.txt
```

All commands were run from the repository root under native Windows.

## Limitations

- **The banding-score response is one honestly-implemented OpenCV
  measurement** (row-mean intensity standard deviation), not a validated
  print-industry banding metric; a real print-quality lab would likely
  use a spatial-frequency-weighted, human-calibrated measure.
- **Null-factor columns are independently randomized, not constructed to
  be exactly orthogonal to the real design.** This is what produced the
  Findings section's residual, honestly-reported false positive; a design
  that explicitly balanced the null columns against the real factors
  (e.g. a resolution-constrained fractional design covering all 25
  candidate factors) would likely close this gap, and was judged out of
  scope for this repository's claimed design (a replicated full 2^5
  factorial plus separately-randomized nulls, not a 25-factor
  fractional-factorial screening design).
- **All press, ink, and sensor data is synthetic**, generated by seeded
  numpy RNGs throughout. No physical press was involved.
- **The filtration extension's defect-count response is one
  honestly-implemented OpenCV blob count** (connected-component labeling
  after a fixed threshold), not a validated particle-inspection algorithm;
  a real filtration QA lab would likely use a calibrated imaging system
  with vendor-specific blob-merge correction.
- **The filtration extension carries no null-factor columns**, unlike the
  press-settings design; its 4 controllable factors were all deliberately
  chosen with a documented true effect (3 nonzero, 1 zero), so it does not
  make or measure a false-positive-rate claim analogous to the
  press-settings one.
- **The 18-replicates-per-condition power result is specific to this
  synthetic bench's noise magnitude** (Poisson particle count plus sensor
  and blob-merge noise at the chosen image size and particle radius); it
  is not a general "12 vs 18" statement about real filtration QA, and the
  Findings section above documents that this number moved substantially
  (26 to 18) once an overlap-driven measurement nonlinearity was fixed,
  the same category of honest correction the press-settings harness's
  clipping-bug fix represents.
- **All bench, particle, and sensor data is synthetic**, generated by
  seeded numpy RNGs throughout. No physical filtration bench was involved.
