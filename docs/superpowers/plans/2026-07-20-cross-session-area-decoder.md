# Cross-Session Area Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current within-session decoder section with a reproducible area-level decoder trained on the before-learning session and evaluated without refitting on the after-learning session.

**Architecture:** Keep the existing data download, SVD loading, behavior loading, and frame-to-trial-position interpolation cells. Replace the duplicated decoder section with one self-contained utility cell that creates neuron-identity-invariant area position features, performs label-free gray-space normalization, trains one decoder per area, and computes within-session and cross-session controls. Test the utility cell directly from the notebook with synthetic arrays so the implementation is verifiable without the Colab dataset.

**Tech Stack:** Python 3.11, Jupyter/nbformat, NumPy, pandas, matplotlib, scikit-learn, pytest.

## Global Constraints

- Main areas are exactly `V1`, `medial`, `lateral`, and `anterior`; `visual_all` is an additional pooled reference.
- Use positions 0–39 as texture features and positions 40–59 for label-free session-level gray normalization.
- Encode `circle1 = 0` and `leaf1 = 1`; other stimuli are excluded only after normalization statistics are computed.
- The after-learning labels may not affect feature construction, normalization, hyperparameters, or model fitting.
- Fit `StandardScaler` and `LogisticRegression` only on before-learning data for the transfer result.
- Use `C=1.0`, `class_weight="balanced"`, `max_iter=5000`, and `random_state=0`.
- Use 2,000 stratified bootstrap resamples and 1,000 training-label permutations in the final notebook run.
- Use seeds 0 for the classifier/CV, 1 for permutation, and 2 for bootstrap.
- All new Markdown and code comments in the notebook are Chinese.
- Preserve the existing uncommitted Chinese-translation changes in `Group_B_working_code.ipynb`.

---

### Task 1: Add synthetic tests for the notebook decoder cell

**Files:**
- Create: `tests/test_cross_session_area_decoder.py`
- Test: `tests/test_cross_session_area_decoder.py`

**Interfaces:**
- Consumes: A notebook code cell containing the marker `# CROSS_SESSION_AREA_DECODER_UTILS`.
- Produces: Tests for `make_area_masks`, `area_position_features`, `encode_leaf_circle_labels`, `make_decoder`, and `evaluate_transfer`.

- [ ] **Step 1: Write the notebook-cell loader and failing structural test**

```python
from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).parents[1] / "Group_B_working_code.ipynb"
MARKER = "# CROSS_SESSION_AREA_DECODER_UTILS"


def load_decoder_namespace():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    matches = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and MARKER in cell.source
    ]
    assert len(matches) == 1, f"expected one decoder utility cell, found {len(matches)}"
    namespace = {}
    exec(compile(matches[0], str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_notebook_contains_one_decoder_utility_cell():
    namespace = load_decoder_namespace()
    assert callable(namespace["make_area_masks"])
    assert callable(namespace["area_position_features"])
    assert callable(namespace["encode_leaf_circle_labels"])
    assert callable(namespace["make_decoder"])
    assert callable(namespace["evaluate_transfer"])
```

- [ ] **Step 2: Run the structural test and verify it fails**

Run:

```bash
/opt/homebrew/bin/python3.11 -m pytest tests/test_cross_session_area_decoder.py::test_notebook_contains_one_decoder_utility_cell -v
```

Expected: FAIL because the marker cell does not yet exist.

- [ ] **Step 3: Add exact mapping, feature, validation, and transfer tests**

```python
import numpy as np
import pytest


def test_area_masks_use_exact_visual_area_mapping():
    ns = load_decoder_namespace()
    iarea = np.array([8, 0, 1, 2, 9, 5, 6, 3, 4, -1, 7])
    masks = ns["make_area_masks"](iarea)
    assert masks["V1"].tolist() == [True, False, False, False, False, False, False, False, False, False, False]
    assert masks["medial"].sum() == 4
    assert masks["lateral"].sum() == 2
    assert masks["anterior"].sum() == 2
    assert masks["visual_all"].sum() == 9


def test_area_position_features_equal_explicit_neuron_reconstruction():
    ns = load_decoder_namespace()
    rng = np.random.default_rng(4)
    U = rng.normal(size=(3, 5))
    component_activity = rng.normal(size=(3, 7, 60))
    component_activity[:, :, 40:60] += np.linspace(-1, 1, 20)
    iarea = np.array([8, 8, 0, 5, 3])

    X, metadata = ns["area_position_features"](
        U, component_activity, iarea, "V1"
    )

    reconstructed = np.einsum("cn,ctp->ntp", U[:, iarea == 8], component_activity)
    area_activity = reconstructed.mean(axis=0)
    gray = area_activity[:, 40:60]
    expected = (area_activity - gray.mean()) / gray.std()
    np.testing.assert_allclose(X, expected[:, :40])
    assert metadata["n_neurons"] == 2


def test_area_position_features_reject_invalid_shapes_and_constant_gray():
    ns = load_decoder_namespace()
    U = np.ones((2, 3))
    activity = np.ones((2, 4, 60))
    with pytest.raises(ValueError, match="gray_std"):
        ns["area_position_features"](U, activity, np.array([8, 8, 8]), "V1")
    with pytest.raises(ValueError, match="component"):
        ns["area_position_features"](U, np.ones((3, 4, 60)), np.array([8, 8, 8]), "V1")


def test_label_encoding_keeps_only_circle1_and_leaf1():
    ns = load_decoder_namespace()
    behavior = {"WallName": np.array(["circle1", "leaf1", "leaf2", "circle1"])}
    labels, keep = ns["encode_leaf_circle_labels"](behavior)
    assert keep.tolist() == [True, True, False, True]
    assert labels.tolist() == [0, 1, 0]


def test_transfer_fits_scaler_only_on_before_features():
    ns = load_decoder_namespace()
    rng = np.random.default_rng(5)
    y_before = np.tile([0, 1], 30)
    y_after = np.tile([0, 1], 35)
    X_before = rng.normal(scale=0.25, size=(60, 40)) + y_before[:, None] * 2.0
    X_after = rng.normal(scale=0.25, size=(70, 40)) + y_after[:, None] * 2.0 + 0.2

    metrics, artifacts = ns["evaluate_transfer"](
        X_before,
        y_before,
        X_after,
        y_after,
        n_bootstrap=20,
        n_permutations=20,
    )

    scaler = artifacts["pipeline"].named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, X_before.mean(axis=0))
    assert metrics["transfer_balanced_accuracy"] > 0.9
    assert 0.0 <= metrics["permutation_p"] <= 1.0
    assert len(artifacts["bootstrap_scores"]) == 20
    assert len(artifacts["permutation_scores"]) == 20
```

- [ ] **Step 4: Run the full test file and confirm failure is due only to the missing utility cell**

Run:

```bash
/opt/homebrew/bin/python3.11 -m pytest tests/test_cross_session_area_decoder.py -v
```

Expected: all tests FAIL at `expected one decoder utility cell, found 0`.

- [ ] **Step 5: Commit the red tests**

```bash
git add tests/test_cross_session_area_decoder.py
git commit -m "test: specify cross-session area decoder"
```

---

### Task 2: Replace duplicated decoder cells with tested utilities

**Files:**
- Modify: `Group_B_working_code.ipynb` decoder section after the existing learning-before/after visualization cells
- Test: `tests/test_cross_session_area_decoder.py`

**Interfaces:**
- Consumes: `U_bef`, `U_aft`, `component_activity_trial_pos_bef`, `component_activity_trial_pos_aft`, `beh_bef`, `beh_aft`, `root`, `unsup_bef`, and `unsup_aft` from earlier cells.
- Produces:
  - `make_area_masks(iarea: array) -> dict[str, bool array]`
  - `area_position_features(U, component_activity, iarea, area_name, eps=1e-12) -> (X, metadata)`
  - `encode_leaf_circle_labels(behavior: dict) -> (labels, keep_mask)`
  - `make_decoder() -> sklearn.pipeline.Pipeline`
  - `evaluate_transfer(X_before, y_before, X_after, y_after, n_bootstrap=2000, n_permutations=1000) -> (metrics, artifacts)`

- [ ] **Step 1: Add one self-contained utility cell with the marker and imports**

The cell must start with:

```python
# CROSS_SESSION_AREA_DECODER_UTILS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AREA_CODES = {
    "V1": (8,),
    "medial": (0, 1, 2, 9),
    "lateral": (5, 6),
    "anterior": (3, 4),
}
ANALYSIS_AREAS = ("V1", "medial", "lateral", "anterior", "visual_all")
```

- [ ] **Step 2: Implement exact area masks and area-position features**

```python
def make_area_masks(iarea):
    iarea = np.asarray(iarea).ravel()
    masks = {name: np.isin(iarea, codes) for name, codes in AREA_CODES.items()}
    masks["visual_all"] = np.logical_or.reduce(list(masks.values()))
    return masks


def area_position_features(U, component_activity, iarea, area_name, eps=1e-12):
    U = np.asarray(U)
    component_activity = np.asarray(component_activity)
    iarea = np.asarray(iarea).ravel()
    if U.ndim != 2 or component_activity.ndim != 3:
        raise ValueError("U and component_activity must be 2-D and 3-D")
    if U.shape[0] != component_activity.shape[0]:
        raise ValueError("component axis mismatch between U and component_activity")
    if U.shape[1] != len(iarea):
        raise ValueError("neuron axis mismatch between U and iarea")
    if component_activity.shape[2] < 60:
        raise ValueError("component_activity must contain 60 position bins")
    masks = make_area_masks(iarea)
    if area_name not in masks:
        raise ValueError(f"unknown area: {area_name}")
    area_mask = masks[area_name]
    if not area_mask.any():
        raise ValueError(f"area {area_name} contains no neurons")
    area_loading = U[:, area_mask].mean(axis=1)
    area_activity = np.einsum("c,ctp->tp", area_loading, component_activity)
    gray = area_activity[:, 40:60]
    gray_mean = float(gray.mean())
    gray_std = float(gray.std())
    if not np.isfinite(gray_std) or gray_std < eps:
        raise ValueError(f"gray_std is invalid for area {area_name}: {gray_std}")
    normalized = (area_activity - gray_mean) / gray_std
    X = normalized[:, :40]
    if not np.isfinite(X).all():
        raise ValueError(f"non-finite features for area {area_name}")
    return X, {
        "area": area_name,
        "n_neurons": int(area_mask.sum()),
        "gray_mean": gray_mean,
        "gray_std": gray_std,
    }
```

- [ ] **Step 3: Implement labels, model, CV, bootstrap, permutation, and transfer evaluation**

```python
def encode_leaf_circle_labels(behavior):
    wall_name = np.asarray(behavior["WallName"]).astype(str)
    keep = np.isin(wall_name, ["circle1", "leaf1"])
    labels = (wall_name[keep] == "leaf1").astype(int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("both circle1 and leaf1 must be present")
    return labels, keep


def make_decoder():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
        ),
    )


def _cv_balanced_accuracy(X, y):
    counts = np.bincount(y, minlength=2)
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        raise ValueError("each class needs at least two trials")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    return float(cross_val_score(
        make_decoder(), X, y, cv=cv, scoring="balanced_accuracy"
    ).mean())


def evaluate_transfer(
    X_before,
    y_before,
    X_after,
    y_after,
    n_bootstrap=2000,
    n_permutations=1000,
):
    X_before = np.asarray(X_before, dtype=float)
    X_after = np.asarray(X_after, dtype=float)
    y_before = np.asarray(y_before, dtype=int)
    y_after = np.asarray(y_after, dtype=int)
    if X_before.ndim != 2 or X_after.ndim != 2 or X_before.shape[1] != X_after.shape[1]:
        raise ValueError("before and after features must be 2-D with equal columns")
    if X_before.shape[1] != 40:
        raise ValueError("final feature count must be 40")
    if len(X_before) != len(y_before) or len(X_after) != len(y_after):
        raise ValueError("feature and label trial counts must match")
    if not np.isfinite(X_before).all() or not np.isfinite(X_after).all():
        raise ValueError("feature matrices must be finite")

    pipeline = make_decoder().fit(X_before, y_before)
    predicted = pipeline.predict(X_after)
    probability = pipeline.predict_proba(X_after)[:, 1]
    observed = balanced_accuracy_score(y_after, predicted)

    bootstrap_rng = np.random.default_rng(2)
    class_indices = [np.flatnonzero(y_after == value) for value in (0, 1)]
    bootstrap_scores = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sampled = np.concatenate([
            bootstrap_rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        bootstrap_scores[index] = balanced_accuracy_score(
            y_after[sampled], predicted[sampled]
        )

    permutation_rng = np.random.default_rng(1)
    permutation_scores = np.empty(n_permutations)
    for index in range(n_permutations):
        shuffled = permutation_rng.permutation(y_before)
        null_model = make_decoder().fit(X_before, shuffled)
        permutation_scores[index] = balanced_accuracy_score(
            y_after, null_model.predict(X_after)
        )

    metrics = {
        "before_cv_balanced_accuracy": _cv_balanced_accuracy(X_before, y_before),
        "after_cv_balanced_accuracy": _cv_balanced_accuracy(X_after, y_after),
        "transfer_balanced_accuracy": float(observed),
        "transfer_roc_auc": float(roc_auc_score(y_after, probability)),
        "bootstrap_ci_low": float(np.quantile(bootstrap_scores, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap_scores, 0.975)),
        "permutation_p": float(
            (1 + np.sum(permutation_scores >= observed)) / (1 + n_permutations)
        ),
    }
    artifacts = {
        "pipeline": pipeline,
        "predicted": predicted,
        "probability": probability,
        "confusion_matrix": confusion_matrix(y_after, predicted, labels=[0, 1]),
        "bootstrap_scores": bootstrap_scores,
        "permutation_scores": permutation_scores,
    }
    return metrics, artifacts
```

- [ ] **Step 4: Run the synthetic tests**

Run:

```bash
/opt/homebrew/bin/python3.11 -m pytest tests/test_cross_session_area_decoder.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit the tested utility cell**

```bash
git add Group_B_working_code.ipynb
git commit -m "feat: add cross-session area decoder utilities"
```

---

### Task 3: Add notebook orchestration, outputs, and visualizations

**Files:**
- Modify: `Group_B_working_code.ipynb` decoder section
- Test: `tests/test_cross_session_area_decoder.py`

**Interfaces:**
- Consumes: Utility functions from Task 2 and the before/after arrays created earlier in the notebook.
- Produces: `cross_session_results`, `cross_session_artifacts`, comparison plots, confusion matrices, and permutation plots.

- [ ] **Step 1: Load both session-specific area labels and encode trial labels**

```python
date_key_bef = "_".join(unsup_bef.split("_")[:-1])
date_key_aft = "_".join(unsup_aft.split("_")[:-1])
iarea_bef = np.load(
    os.path.join(root, date_key_bef + "_trans.npz"), allow_pickle=True
)["iarea"]
iarea_aft = np.load(
    os.path.join(root, date_key_aft + "_trans.npz"), allow_pickle=True
)["iarea"]

y_bef, keep_bef = encode_leaf_circle_labels(beh_bef)
y_aft, keep_aft = encode_leaf_circle_labels(beh_aft)
```

- [ ] **Step 2: Run one transfer analysis per area and build the results table**

```python
rows = []
cross_session_artifacts = {}

for area in ANALYSIS_AREAS:
    X_bef_all, meta_bef = area_position_features(
        U_bef, component_activity_trial_pos_bef, iarea_bef, area
    )
    X_aft_all, meta_aft = area_position_features(
        U_aft, component_activity_trial_pos_aft, iarea_aft, area
    )
    X_bef = X_bef_all[keep_bef]
    X_aft = X_aft_all[keep_aft]
    metrics, artifacts = evaluate_transfer(
        X_bef, y_bef, X_aft, y_aft,
        n_bootstrap=2000,
        n_permutations=1000,
    )
    rows.append({
        "area": area,
        "n_neurons_before": meta_bef["n_neurons"],
        "n_neurons_after": meta_aft["n_neurons"],
        "n_trials_before": len(y_bef),
        "n_trials_after": len(y_aft),
        **metrics,
    })
    cross_session_artifacts[area] = artifacts

cross_session_results = pd.DataFrame(rows)
display(cross_session_results)
```

- [ ] **Step 3: Add comparison, confusion, and permutation plots**

```python
fig, ax = plt.subplots(figsize=(8, 4.5))
transfer = cross_session_results["transfer_balanced_accuracy"].to_numpy()
ci_low = cross_session_results["bootstrap_ci_low"].to_numpy()
ci_high = cross_session_results["bootstrap_ci_high"].to_numpy()
ax.errorbar(
    cross_session_results["area"],
    transfer,
    yerr=np.vstack([transfer - ci_low, ci_high - transfer]),
    fmt="o",
    capsize=4,
)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
ax.set_ylim(0, 1)
ax.set_ylabel("学习前→学习后 balanced accuracy")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(cross_session_results))
width = 0.24
ax.bar(x - width, cross_session_results["before_cv_balanced_accuracy"], width, label="学习前内部 CV")
ax.bar(x, cross_session_results["after_cv_balanced_accuracy"], width, label="学习后内部 CV")
ax.bar(x + width, cross_session_results["transfer_balanced_accuracy"], width, label="学习前→学习后")
ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="机会水平")
ax.set_xticks(x, cross_session_results["area"])
ax.set_ylim(0, 1)
ax.set_ylabel("Balanced accuracy")
ax.legend()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, len(ANALYSIS_AREAS), figsize=(3 * len(ANALYSIS_AREAS), 3))
for ax, area in zip(axes, ANALYSIS_AREAS):
    matrix = cross_session_artifacts[area]["confusion_matrix"]
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, matrix[row, column], ha="center", va="center")
    ax.set_title(area)
    ax.set_xticks([0, 1], ["circle1", "leaf1"])
    ax.set_yticks([0, 1], ["circle1", "leaf1"])
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, len(ANALYSIS_AREAS), figsize=(3 * len(ANALYSIS_AREAS), 3))
for ax, area in zip(axes, ANALYSIS_AREAS):
    null_scores = cross_session_artifacts[area]["permutation_scores"]
    observed = cross_session_results.loc[
        cross_session_results["area"] == area, "transfer_balanced_accuracy"
    ].iloc[0]
    ax.hist(null_scores, bins=25, color="0.7")
    ax.axvline(observed, color="red", linewidth=2)
    ax.set_title(area)
    ax.set_xlabel("Permutation balanced accuracy")
plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Remove obsolete decoder/debug cells and add Chinese Markdown interpretation guidance**

Use `nbformat` to retain cells 0–36, remove the old cells 37–64, and append exactly these cells in order:

```python
new_cells = [
    nbformat.v4.new_markdown_cell(
        "## 跨学习阶段的脑区级 Decoder\n\n"
        "以下分析在学习前训练 decoder，并在不重新拟合的情况下直接测试学习后数据。"
    ),
    nbformat.v4.new_code_cell(decoder_utility_source),
    nbformat.v4.new_markdown_cell(
        "### 构建共同脑区特征\n\n"
        "每个脑区用 40 维平均空间活动曲线表示；灰色区域归一化不使用刺激标签。"
    ),
    nbformat.v4.new_code_cell(analysis_source),
    nbformat.v4.new_markdown_cell(
        "### 结果\n\n"
        "主要结果是学习前→学习后的 balanced accuracy；学习前和学习后内部 CV 仅用于解释。"
    ),
    nbformat.v4.new_code_cell(plot_source),
]
notebook.cells = notebook.cells[:37] + new_cells
```

`decoder_utility_source`、`analysis_source` 和 `plot_source` must contain the exact code from Steps 1–3. This removes both `%%writefile decoding.py` cells and all downstream cells that import the overwritten module, use hidden `keep`/`beh` state, or repeat the old within-after decoder.

- [ ] **Step 5: Validate notebook structure and rerun synthetic tests**

Run:

```bash
/opt/homebrew/bin/python3.11 - <<'PY'
import nbformat
nb = nbformat.read("Group_B_working_code.ipynb", as_version=4)
nbformat.validate(nb)
print(f"validated_cells={len(nb.cells)}")
PY
/opt/homebrew/bin/python3.11 -m pytest tests/test_cross_session_area_decoder.py -v
```

Expected: notebook validation succeeds and all tests PASS.

- [ ] **Step 6: Commit the integrated notebook analysis**

```bash
git add Group_B_working_code.ipynb tests/test_cross_session_area_decoder.py
git commit -m "feat: evaluate decoder transfer across learning"
```

---

### Task 4: Final reproducibility and scope verification

**Files:**
- Verify: `Group_B_working_code.ipynb`
- Verify: `tests/test_cross_session_area_decoder.py`
- Verify: `docs/superpowers/specs/2026-07-20-cross-session-area-decoder-design.md`

**Interfaces:**
- Consumes: Completed notebook and tests.
- Produces: Evidence that the notebook is structurally valid, the utility logic passes synthetic tests, and no stale decoder code remains.

- [ ] **Step 1: Verify no stale decoder or hidden-state references remain**

Run:

```bash
/opt/homebrew/bin/python3.11 - <<'PY'
import nbformat
nb = nbformat.read("Group_B_working_code.ipynb", as_version=4)
source = "\n".join(cell.source for cell in nb.cells)
assert source.count("# CROSS_SESSION_AREA_DECODER_UTILS") == 1
assert "%%writefile decoding.py" not in source
assert "from decoding import" not in source
assert "print(keep.shape)" not in source
assert 'beh["TrialStim"]' not in source
print("stale_decoder_scan=PASS")
PY
```

Expected: `stale_decoder_scan=PASS`.

- [ ] **Step 2: Run complete local validation**

Run:

```bash
/opt/homebrew/bin/python3.11 -m pytest tests/test_cross_session_area_decoder.py -v
/opt/homebrew/bin/python3.11 - <<'PY'
import nbformat
nb = nbformat.read("Group_B_working_code.ipynb", as_version=4)
nbformat.validate(nb)
print("nbformat_validation=PASS")
PY
git diff --check
```

Expected: all tests PASS, notebook validation passes, and `git diff --check` exits 0.

- [ ] **Step 3: Record the execution gap accurately**

If `/content/Zhong_et_al_2025` is not available locally, do not claim a complete data run. Record that the Colab notebook must be run from the first cell after downloading the Figshare inputs. The exact full-run command in a compatible environment is:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace Group_B_working_code.ipynb
```

- [ ] **Step 4: Review final Git state and commits**

Run:

```bash
git status --short --branch
git log --oneline -5
```

Expected: only intentional changes remain; no generated `decoding.py`, downloaded data, or temporary files are staged.
