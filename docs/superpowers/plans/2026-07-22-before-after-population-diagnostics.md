# Before/After Population Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add before/after V1 PCA-LDA embeddings and before/after Haufe area-contribution figures to the focused Group B decoder notebook.

**Architecture:** Add one self-contained population-diagnostics utility cell and one analysis/plot cell after the existing transfer plots. Reuse each session's component trial-position tensor, `U`, `iarea`, and encoded labels; calculate every diagnostic independently within a session.

**Tech Stack:** Jupyter/`nbformat`, NumPy, pandas, Matplotlib, scikit-learn, pytest, git, GitHub CLI.

## Global Constraints

- Analyze both `TX105_2022_10_08_2` and `TX105_2022_10_19_2` independently.
- Retain only `circle1 = 0` and `leaf1 = 1` trials.
- Use textured positions 0–39 and the existing visual-area code mapping.
- Do not align neurons, PCA axes, or decoder weights across sessions.
- Treat full-data LDA and Haufe contributions as descriptive session-level diagnostics.
- Keep the existing within-session CV and transfer decoder unchanged.
- Keep Python comments/docstrings in English and reader-facing Markdown in Chinese.
- Leave new cached outputs empty until Colab reruns the real data.

---

### Task 1: Define population-diagnostic behavior with failing tests

**Files:**
- Modify: `tests/test_cross_session_area_decoder.py`
- Test: `tests/test_cross_session_area_decoder.py`

**Interfaces:**
- Consumes: notebook cell marker `# POPULATION_DIAGNOSTICS_UTILS`.
- Produces: an executable utility namespace and regression coverage for reconstruction, embeddings, contributions, and both-session notebook integration.

- [ ] **Step 1: Add the utility-cell loader**

```python
POPULATION_DIAGNOSTICS_MARKER = "# POPULATION_DIAGNOSTICS_UTILS"


def load_population_diagnostics_namespace():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    matches = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code"
        and POPULATION_DIAGNOSTICS_MARKER in cell.source
    ]
    assert len(matches) == 1
    namespace = {}
    exec(compile(matches[0], str(NOTEBOOK), "exec"), namespace)
    return namespace
```

- [ ] **Step 2: Add a reconstruction-equivalence test**

```python
def test_v1_trial_features_equal_explicit_reconstruction():
    namespace = load_population_diagnostics_namespace()
    rng = np.random.default_rng(21)
    U = rng.normal(size=(3, 6))
    activity = rng.normal(size=(3, 8, 60))
    iarea = np.array([8, 0, 8, 5, 3, 8])
    observed = namespace["area_neuron_trial_features"](
        U, activity, iarea, area_codes=(8,)
    )
    component_trial = activity[:, :, :40].mean(axis=2)
    expected = (U[:, iarea == 8].T @ component_trial).T
    np.testing.assert_allclose(observed, expected)
```

- [ ] **Step 3: Add embedding and contribution tests**

```python
def test_population_embeddings_and_haufe_contributions_are_finite():
    namespace = load_population_diagnostics_namespace()
    rng = np.random.default_rng(22)
    labels = np.tile([0, 1], 12)
    trial_features = rng.normal(size=(24, 10)) + labels[:, None]
    embeddings = namespace["condition_embeddings"](trial_features, labels)
    assert embeddings["pca"].shape == (24, 2)
    assert embeddings["lda"].shape == (24, 2)
    assert np.isfinite(embeddings["pca"]).all()
    assert np.isfinite(embeddings["lda"]).all()

    U = rng.normal(size=(4, 8))
    activity = rng.normal(size=(4, 24, 60))
    activity[:, labels == 1, :40] += 0.8
    iarea = np.array([8, 8, 0, 1, 5, 6, 3, 4])
    table = namespace["haufe_area_contributions"](
        U, activity, iarea, labels
    )
    assert table["area"].tolist() == [
        "V1", "medial", "lateral", "anterior"
    ]
    assert np.isfinite(table.select_dtypes(include=[float, int])).all().all()
    assert table["share_percent"].sum() == pytest.approx(100.0)
```

- [ ] **Step 4: Add a notebook integration test**

```python
def test_notebook_runs_population_diagnostics_for_both_sessions():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    required = [
        'session_label="Before learning"',
        'session_label="After learning"',
        'title="V1 circle1/leaf1 population embeddings"',
        'title="Area contribution to the session decoder"',
    ]
    for value in required:
        assert value in source
```

- [ ] **Step 5: Run the new tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/nma-decoder-test-deps-clean.Q1Thli \
/opt/homebrew/bin/python3.11 -m pytest -q \
tests/test_cross_session_area_decoder.py -k population
```

Expected: failures because the marked utility cell and both-session calls do not yet exist.

- [ ] **Step 6: Commit the failing tests**

```bash
git add tests/test_cross_session_area_decoder.py
git commit -m "test: define population diagnostic behavior"
```

---

### Task 2: Implement session-specific embeddings and contributions

**Files:**
- Modify: `Group_B_working_code.ipynb`
- Test: `tests/test_cross_session_area_decoder.py`

**Interfaces:**
- Produces: `area_neuron_trial_features(...) -> trials × area neurons`.
- Produces: `condition_embeddings(...) -> {"pca": n×2, "lda": n×2}`.
- Produces: `haufe_area_contributions(...) -> pandas.DataFrame`.
- Produces: `plot_population_diagnostics(...)` and `plot_area_contributions(...)`.

- [ ] **Step 1: Add a self-contained utility cell**

Implement these signatures under `# POPULATION_DIAGNOSTICS_UTILS`:

```python
def area_neuron_trial_features(
    U, component_activity, iarea, area_codes=(8,), position_stop=40
):
    component_trial = component_activity[:, :, :position_stop].mean(axis=2)
    area_mask = np.isin(np.asarray(iarea).ravel(), area_codes)
    return (U[:, area_mask].T @ component_trial).T


def condition_embeddings(trial_features, labels, random_state=0):
    standardized = StandardScaler().fit_transform(trial_features)
    pca = PCA(
        n_components=2,
        svd_solver="randomized",
        random_state=random_state,
    ).fit_transform(standardized)
    ld1 = LinearDiscriminantAnalysis(n_components=1).fit_transform(
        standardized, labels
    )
    return {"pca": pca, "lda": np.column_stack([ld1[:, 0], pca[:, 0]])}
```

Implement `haufe_area_contributions(...)` with component trial means,
`StandardScaler + LogisticRegression`, raw-unit weights, covariance-times-weight
Haufe patterns, `U.T` back-projection, and the four existing area masks.

- [ ] **Step 2: Add plotting helpers**

`plot_population_diagnostics` must place before/after rows and PCA/LDA columns,
use fixed circle/leaf colors, and label the supervised axis `LD1` and the second
axis `PC1`.

`plot_area_contributions` must place before/after rows and total-share/per-neuron
columns, use `share_percent` on the left, and label per-neuron values as
session-specific arbitrary units.

- [ ] **Step 3: Add the Chinese methods/caveats Markdown**

Explain that PCA axes are independently fitted, LDA is descriptive rather than
cross-validated, Haufe shares are normalized within session, and absolute
per-neuron activations are not directly scaled across sessions.

- [ ] **Step 4: Add the both-session analysis/plot cell**

Build filtered V1 trial matrices for before and after with `keep_bef` and
`keep_aft`, compute both embedding dictionaries and both contribution tables,
display a combined table with a `session` column, and call both plot helpers
with the exact integration-test titles.

- [ ] **Step 5: Run all tests and verify GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/nma-decoder-test-deps-clean.Q1Thli \
/opt/homebrew/bin/python3.11 -m pytest -q
```

Expected: all decoder and population-diagnostic tests pass.

- [ ] **Step 6: Commit the notebook**

```bash
git add Group_B_working_code.ipynb
git commit -m "feat: compare population diagnostics before and after learning"
```

---

### Task 3: Validate, publish, and merge

**Files:**
- Verify: `Group_B_working_code.ipynb`
- Verify: `tests/test_cross_session_area_decoder.py`
- Commit: `docs/superpowers/plans/2026-07-22-before-after-population-diagnostics.md`

- [ ] **Step 1: Validate notebook structure and synthetic execution**

Run `nbformat.validate`, compile all code cells, execute the new utility cell,
exercise both sessions with synthetic arrays, and execute both plot helpers
under `MPLBACKEND=Agg`. Verify all new cells have empty outputs, null execution
counts, and unique metadata IDs.

- [ ] **Step 2: Run final tests and diff checks**

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/nma-decoder-test-deps-clean.Q1Thli \
/opt/homebrew/bin/python3.11 -m pytest -q
git diff --check master...HEAD
```

- [ ] **Step 3: Commit the implementation plan**

```bash
git add docs/superpowers/plans/2026-07-22-before-after-population-diagnostics.md
git commit -m "docs: plan before-after population diagnostics"
```

- [ ] **Step 4: Push and merge without another confirmation gate**

Push `agent/before-after-population-diagnostics`, create a pull request to
`master`, confirm it is mergeable with no failing checks, merge it, fast-forward
local `master`, and rerun the full test suite.

Full real-data execution remains documented as a Colab-only gap.
