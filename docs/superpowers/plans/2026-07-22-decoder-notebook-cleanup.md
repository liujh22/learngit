# Decoder Notebook Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `Group_B_working_code.ipynb` to the runnable TX105 decoder pipeline, add explanatory English code comments, preserve the approved statistics, and publish the verified changes.

**Architecture:** Keep one linear notebook path from six required Figshare files through SVD/behavior loading, one generic frame-to-position interpolation function, area-level feature construction, bidirectional decoding, and results visualization. Structural tests identify the utility cells by stable metadata markers and prevent removed exploratory identifiers from returning.

**Tech Stack:** Jupyter notebook JSON/`nbformat`, Python, NumPy, SciPy, pandas, Matplotlib, scikit-learn, pytest, git, GitHub CLI.

## Global Constraints

- Keep sessions `TX105_2022_10_08_2` and `TX105_2022_10_19_2`.
- Keep `circle1 = 0` and `leaf1 = 1`.
- Keep the current `iarea` mappings and `visual_all` union.
- Keep positions 40–59 for session-level label-free normalization and positions 0–39 as the 40 decoder features.
- Keep `StandardScaler` plus balanced logistic regression with the existing hyperparameters and seeds.
- Keep independent before-to-after and after-to-before transfer directions;
  fit each scaler/classifier only on its source session.
- Keep 5-fold stratified within-session CV, 2,000 bootstraps, and 1,000 label permutations.
- Keep Python comments/docstrings in English and reader-facing Markdown in Chinese.
- Do not present cached exploratory outputs as current decoder results.

---

### Task 1: Lock the focused notebook contract with failing tests

**Files:**
- Modify: `tests/test_cross_session_area_decoder.py`
- Test: `tests/test_cross_session_area_decoder.py`

**Interfaces:**
- Consumes: notebook cells in `Group_B_working_code.ipynb`.
- Produces: structural requirements for `# TRIAL_POSITION_UTILS`, the generic `activity_by_trial_and_position(...)` function, the removed-cell list, and required explanatory comments.

- [ ] **Step 1: Add a helper that executes the shared interpolation cell**

```python
TRIAL_POSITION_MARKER = "# TRIAL_POSITION_UTILS"


def load_trial_position_namespace():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    matches = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and TRIAL_POSITION_MARKER in cell.source
    ]
    assert len(matches) == 1
    namespace = {}
    exec(compile(matches[0], str(NOTEBOOK), "exec"), namespace)
    return namespace
```

- [ ] **Step 2: Add structural and functional cleanup tests**

```python
def test_notebook_uses_one_shared_trial_position_function():
    namespace = load_trial_position_namespace()
    activity = np.array([[10, 20, 30, 40, 50, 60, 70]], dtype=float)
    cumulative_position = np.array([0, 1, 1, 2, 3, 4, 5], dtype=float)

    result = namespace["activity_by_trial_and_position"](
        activity,
        cumulative_position,
        n_trials=1,
        corridor_length=6,
    )

    assert result.shape == (1, 1, 6)
    np.testing.assert_allclose(result[0, 0], [10, 20, 40, 50, 60, 70])


def test_notebook_excludes_decoder_irrelevant_exploration():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    removed_identifiers = [
        "sup_bef",
        "sup_aft",
        "example_activity_",
        "pattern_activity_",
        "activity_by_trial_and_position_aft",
        "activity_by_trial_and_position_bef",
        "Plot activity by trial and position",
    ]
    for identifier in removed_identifiers:
        assert identifier not in source


def test_notebook_documents_non_obvious_decoder_steps():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    required_explanations = [
        "Keep all 60 position bins until gray-region normalization",
        "Average the session-specific U loadings only within the requested area",
        "Within-session CV refits the scaler and classifier on every training fold",
        "The target session is used only for scoring",
    ]
    for explanation in required_explanations:
        assert explanation in source
```

- [ ] **Step 3: Run the new tests and verify the expected RED state**

Run:

```bash
PYTHONPATH=/tmp/nma-decoder-test-deps-clean.Q1Thli \
  /opt/homebrew/bin/python3.11 -m pytest -q \
  tests/test_cross_session_area_decoder.py \
  -k "shared_trial_position or excludes_decoder_irrelevant or documents_non_obvious"
```

Expected: failures because the notebook still contains duplicated session-specific interpolation functions and exploratory identifiers, and lacks the new marker/comments.

- [ ] **Step 4: Commit the failing contract tests**

```bash
git add tests/test_cross_session_area_decoder.py
git commit -m "test: define focused decoder notebook contract"
```

---

### Task 2: Replace the exploratory notebook with the focused pipeline

**Files:**
- Modify: `Group_B_working_code.ipynb`
- Test: `tests/test_cross_session_area_decoder.py`

**Interfaces:**
- Consumes: `activity_by_trial_and_position(activity_by_frame, cumulative_position, n_trials, corridor_length=60)` and the existing decoder utility functions.
- Produces: `component_activity_trial_pos_bef`, `component_activity_trial_pos_aft`, `cross_session_results`, and `cross_session_artifacts` for the final analysis and plots.

- [ ] **Step 1: Reduce setup and downloads to the six TX105 files**

Keep these filenames in one visible set and filter Figshare metadata by filename:

```python
REQUIRED_FILES = {
    "TX105_2022_10_08_2_SVD_dec.npy",
    "TX105_2022_10_08_trans.npz",
    "TX105_2022_10_19_2_SVD_dec.npy",
    "TX105_2022_10_19_trans.npz",
    "Beh_unsup_train1_before_learning.npy",
    "Beh_unsup_train1_after_learning.npy",
}
```

Remove `sup_bef`, `sup_aft`, the numeric download-ID list, and all supervised filenames. Add `response.raise_for_status()` for metadata and file requests, and retain the existing skip-if-present behavior.

- [ ] **Step 2: Keep only decoder inputs from the SVD and behavior files**

The SVD cells must produce only `U_bef`, `V_bef`, `U_aft`, and `V_aft`, with concise shape prints and comments documenting `components × neurons` and `components × frames`.

The behavior cells must produce only:

```python
beh_bef
beh_aft
ntrials_bef = int(beh_bef["ntrials"])
ntrials_aft = int(beh_aft["ntrials"])
cum_pos_fr_bef = np.asarray(beh_bef["ft_PosCum"])
cum_pos_fr_aft = np.asarray(beh_aft["ft_PosCum"])
```

Delete the full dictionary dumps and behavior variables used only by removed plots.

- [ ] **Step 3: Replace the duplicate interpolation cells with one tested utility**

```python
# TRIAL_POSITION_UTILS: shared frame-to-trial-position conversion
from scipy import interpolate


def activity_by_trial_and_position(
    activity_by_frame,
    cumulative_position,
    n_trials,
    corridor_length=60,
):
    """Interpolate signals from recording frames to trial-by-position bins."""
    activity_by_frame = np.asarray(activity_by_frame)
    cumulative_position = np.asarray(cumulative_position)
    n_frames = activity_by_frame.shape[1]
    if len(cumulative_position) < n_frames:
        raise ValueError(
            "The behavioral position array has fewer frames than the "
            "neural activity array."
        )

    source_positions = cumulative_position[:n_frames]
    unique_positions, unique_indices = np.unique(
        source_positions,
        return_index=True,
    )
    target_positions = np.arange(n_trials * corridor_length)
    interpolated = np.empty(
        (activity_by_frame.shape[0], len(target_positions)),
        dtype=np.float32,
    )
    for signal_index, signal in enumerate(activity_by_frame):
        model = interpolate.interp1d(
            unique_positions,
            signal[unique_indices],
            bounds_error=False,
            fill_value="extrapolate",
        )
        interpolated[signal_index] = model(target_positions)
    return interpolated.reshape(
        activity_by_frame.shape[0],
        n_trials,
        corridor_length,
    )
```

Add comments explaining why duplicate cumulative positions are removed and why all 60 bins are retained until decoder normalization.

- [ ] **Step 4: Remove decoder-independent cells and rewrite the section narrative**

Delete all cells for:

- full `U`/`V` printing;
- reconstructed example-neuron traces;
- example-activity shape printing;
- `pattern_activity_*` creation;
- component heatmaps and 5-by-5 inspections;
- trial-sorted component figures.

Renumber the remaining Chinese Markdown sections to describe setup, data loading, frame-to-position conversion, feature/decoder utilities, analysis, and visualization. Clear stale outputs and execution counts from rewritten cells.

- [ ] **Step 5: Expand comments in the decoder and plotting cells without changing calculations**

Add the exact test-visible explanations and nearby details for:

- area-average reconstruction via mean `U` loadings and `einsum`;
- session-level gray normalization;
- per-fold within-session fitting;
- source-only transfer fitting and target-only scoring;
- stratified bootstrap confidence intervals;
- source-label permutation null distributions;
- the meaning of the dedicated within-session plot and bidirectional transfer plot.

Keep the dedicated `Within-session decoding by learning stage` chart already present in the working tree.

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run:

```bash
PYTHONPATH=/tmp/nma-decoder-test-deps-clean.Q1Thli \
  /opt/homebrew/bin/python3.11 -m pytest -q \
  tests/test_cross_session_area_decoder.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit the focused notebook**

```bash
git add Group_B_working_code.ipynb
git commit -m "refactor: focus Group B notebook on decoder analysis"
```

---

### Task 3: Validate the notebook artifact and publish the branch

**Files:**
- Verify: `Group_B_working_code.ipynb`
- Verify: `tests/test_cross_session_area_decoder.py`
- Verify: `docs/superpowers/specs/2026-07-22-decoder-notebook-cleanup-design.md`
- Verify: `docs/superpowers/plans/2026-07-22-decoder-notebook-cleanup.md`

**Interfaces:**
- Consumes: the focused notebook and regression tests.
- Produces: a pushed `agent/decoder-notebook-cleanup` branch and a draft pull request targeting `master`.

- [ ] **Step 1: Validate notebook structure and compile standard Python cells**

Run a Python check that calls `nbformat.validate(notebook)`, compiles every code cell, asserts all `outputs` lists are empty, and verifies cell IDs are unique.

Expected: `nbformat.validate: PASS`, `code compile: PASS`, `outputs cleared: PASS`, and `cell IDs unique: PASS`.

- [ ] **Step 2: Execute utility, decoder, and plotting cells with synthetic arrays**

Load the marker cells into isolated namespaces, run the generic interpolation test, exercise both transfer directions with reduced bootstrap/permutation counts, and execute the complete plotting cell using an Agg backend and synthetic `cross_session_results`/`cross_session_artifacts`.

Expected: `synthetic interpolation: PASS`, `synthetic bidirectional decoder: PASS`, and `synthetic plots: PASS`.

- [ ] **Step 3: Run final repository checks**

```bash
PYTHONPATH=/tmp/nma-decoder-test-deps-clean.Q1Thli \
  /opt/homebrew/bin/python3.11 -m pytest -q
git diff --check origin/master...HEAD
git status --short --branch
```

Expected: all tests pass, no whitespace errors, and only the implementation-plan file remains uncommitted before its documentation commit.

- [ ] **Step 4: Commit the implementation plan**

```bash
git add docs/superpowers/plans/2026-07-22-decoder-notebook-cleanup.md
git commit -m "docs: plan decoder notebook cleanup"
```

- [ ] **Step 5: Confirm GitHub authentication and push**

```bash
gh --version
gh auth status
git push -u origin agent/decoder-notebook-cleanup
```

Expected: authenticated GitHub CLI and a remote tracking branch.

- [ ] **Step 6: Open a draft pull request**

Create a draft PR from `agent/decoder-notebook-cleanup` to `master` describing the removed exploratory code, expanded comments, preserved decoder behavior, and test/Colab execution status.

Expected: a GitHub draft PR URL.
