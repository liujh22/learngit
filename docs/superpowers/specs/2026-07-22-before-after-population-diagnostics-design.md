# Before/after population diagnostics design

## Goal

Restore the two population-level visual analyses from the original Group B
notebook and extend each one from the after-learning session to both TX105
sessions. These diagnostics supplement the existing area-average transfer
decoder; they do not replace or alter it.

## Scope

Add two session-specific analyses after the decoder results:

1. V1 `circle1`/`leaf1` condition embeddings using PCA and LDA.
2. Haufe-pattern summaries of each visual area's contribution to a
   session-level component-space decoder.

Run every calculation independently for:

- before learning: `TX105_2022_10_08_2`;
- after learning: `TX105_2022_10_19_2`.

No neuron registration, PCA-axis alignment, or model transfer is performed in
these figures.

## V1 condition embeddings

For one session, average the 400-component activity over textured positions
0–39 for every trial. Reconstruct only V1 neurons with the session's `U`
matrix, producing `trials × V1 neurons`, and retain `circle1` and `leaf1`
trials. Standardize every neuron across the retained trials.

Fit PCA with two components to the standardized V1 matrix. Fit binary LDA to
the same complete matrix; because binary LDA has only one discriminant axis,
plot `LD1` horizontally and the already computed `PC1` vertically. Use fixed
seed 0 for deterministic PCA and fixed colors for the two labels.

The figure has two rows and two columns:

- rows: before learning, after learning;
- columns: PCA, LDA.

PCA is an unsupervised descriptive view. LDA uses the displayed labels on the
same full dataset, so visible separation is descriptive and must not be read as
held-out decoding performance. The existing CV and transfer results remain the
appropriate performance estimates.

## Area contribution analysis

For one session, average component activity over positions 0–39 to obtain a
`trials × 400 components` matrix and retain `circle1`/`leaf1` trials. Fit the
same fixed `StandardScaler` plus balanced logistic-regression recipe used by
the main analysis, using all retained trials for this descriptive model.

Convert standardized classifier weights to original component units, compute
the Haufe activation pattern as feature covariance multiplied by those raw
weights, and project the component pattern to neurons with the same session's
`U` matrix. For V1, medial, lateral, and anterior, report:

- neuron count;
- sum of absolute neuron activations;
- percentage of the four-area absolute activation total;
- mean absolute activation per neuron.

The figure has two rows and two columns:

- rows: before learning, after learning;
- left column: share of decision drive (%);
- right column: mean absolute Haufe activation per neuron.

The percentage is normalized within each session. Absolute per-neuron
activation depends on session-specific SVD and activity scaling, so it is used
primarily to compare areas within a row, not absolute magnitude across rows.

## Notebook integration

Add one self-contained utility cell marked
`# POPULATION_DIAGNOSTICS_UTILS`, one Chinese Markdown explanation, and one
analysis/plot cell after the existing bidirectional decoder figures. Reuse the
already loaded `U`, component trial-position tensors, anatomical labels, and
encoded trial labels. Do not recreate interpolation or download data.

Python comments and docstrings remain in English. The reader-facing Markdown
and caveats remain in Chinese. New notebook cells have stable metadata IDs,
empty cached outputs, and null execution counts until the Colab analysis is
rerun.

## Verification

Tests must verify that:

- V1 reconstruction equals explicit component-to-neuron multiplication;
- PCA and LDA return finite two-dimensional coordinates for both labels;
- Haufe contribution tables contain the four declared areas, finite values,
  and within-session shares summing to 100%;
- the notebook calls both diagnostics for before and after sessions;
- the complete plotting cell executes with synthetic arrays using a non-GUI
  backend;
- all existing decoder tests continue to pass.

Full real-data execution remains a Colab-only step because the local machine
does not contain `/content/Zhong_et_al_2025`.
