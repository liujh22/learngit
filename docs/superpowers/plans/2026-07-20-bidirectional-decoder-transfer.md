# Bidirectional Decoder Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the area-average decoder notebook so it reports both before-learning→after-learning and after-learning→before-learning transfer with symmetric statistics and plots.

**Architecture:** Keep the tested single-direction `evaluate_transfer` function as the only fitting and scoring implementation. Add a small `evaluate_bidirectional_transfer` wrapper that calls it once per direction with swapped source and target sessions, then update the analysis and plotting cells to expose both results explicitly.

**Tech Stack:** Python 3.11, Jupyter/nbformat, NumPy, pandas, matplotlib, scikit-learn, pytest.

## Global Constraints

- Preserve the existing area-average 40-position feature definition and gray-space normalization.
- Before→after remains the primary direction and keeps its existing result-column names.
- After→before uses an independent scaler and classifier fitted only on after-learning trials.
- Each direction uses 2,000 target-stratified bootstrap resamples and 1,000 source-label permutations in the notebook.
- Keep model/CV seed 0, permutation seed 1, and bootstrap seed 2.
- All new notebook Markdown and code comments are Chinese.
- Do not execute or overwrite the notebook with incomplete local `/content` data.

---

### Task 1: Specify and implement the bidirectional evaluation wrapper

**Files:**
- Modify: `tests/test_cross_session_area_decoder.py`
- Modify: `Group_B_working_code.ipynb` utility cell marked `# CROSS_SESSION_AREA_DECODER_UTILS`
- Test: `tests/test_cross_session_area_decoder.py`

**Interfaces:**
- Consumes: `evaluate_transfer(X_train, y_train, X_test, y_test, n_bootstrap, n_permutations)`.
- Produces: `evaluate_bidirectional_transfer(X_before, y_before, X_after, y_after, n_bootstrap=2000, n_permutations=1000) -> (direction_metrics, direction_artifacts)` with keys `before_to_after` and `after_to_before`.

- [ ] **Step 1: Add a failing callable and source-isolation test**

Add `evaluate_bidirectional_transfer` to the callable assertions and append:

```python
def test_bidirectional_transfer_fits_each_scaler_only_on_source_session():
    namespace = load_decoder_namespace()
    rng = np.random.default_rng(8)
    labels_before = np.tile([0, 1], 20)
    labels_after = np.tile([0, 1], 25)
    features_before = (
        rng.normal(scale=0.3, size=(40, 40))
        + labels_before[:, None] * 1.5
    )
    features_after = (
        rng.normal(scale=0.3, size=(50, 40))
        + labels_after[:, None] * 1.5
        + 4.0
    )

    metrics, artifacts = namespace["evaluate_bidirectional_transfer"](
        features_before,
        labels_before,
        features_after,
        labels_after,
        n_bootstrap=10,
        n_permutations=10,
    )

    forward_scaler = artifacts["before_to_after"]["pipeline"].named_steps[
        "standardscaler"
    ]
    reverse_scaler = artifacts["after_to_before"]["pipeline"].named_steps[
        "standardscaler"
    ]
    np.testing.assert_allclose(forward_scaler.mean_, features_before.mean(axis=0))
    np.testing.assert_allclose(reverse_scaler.mean_, features_after.mean(axis=0))
    assert set(metrics) == {"before_to_after", "after_to_before"}
    assert len(artifacts["before_to_after"]["bootstrap_scores"]) == 10
    assert len(artifacts["after_to_before"]["permutation_scores"]) == 10
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/nma-decoder-venv/bin/python \
  -m pytest -p no:cacheprovider \
  tests/test_cross_session_area_decoder.py::test_bidirectional_transfer_fits_each_scaler_only_on_source_session -v
```

Expected: FAIL because `evaluate_bidirectional_transfer` is not defined.

- [ ] **Step 3: Add the minimal wrapper to the notebook utility cell**

Append immediately after `evaluate_transfer`:

```python
def evaluate_bidirectional_transfer(
    X_before,
    y_before,
    X_after,
    y_after,
    n_bootstrap=2000,
    n_permutations=1000,
):
    """分别拟合两个方向的独立模型，并返回方向明确的结果。"""
    forward_metrics, forward_artifacts = evaluate_transfer(
        X_before,
        y_before,
        X_after,
        y_after,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
    )
    reverse_metrics, reverse_artifacts = evaluate_transfer(
        X_after,
        y_after,
        X_before,
        y_before,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
    )
    return (
        {
            "before_to_after": forward_metrics,
            "after_to_before": reverse_metrics,
        },
        {
            "before_to_after": forward_artifacts,
            "after_to_before": reverse_artifacts,
        },
    )
```

- [ ] **Step 4: Run the entire synthetic suite and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/nma-decoder-venv/bin/python \
  -m pytest -p no:cacheprovider tests/test_cross_session_area_decoder.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit the wrapper and test**

```bash
git add tests/test_cross_session_area_decoder.py Group_B_working_code.ipynb
git commit -m "feat: evaluate decoder transfer in both directions"
```

---

### Task 2: Report and plot both transfer directions

**Files:**
- Modify: `tests/test_cross_session_area_decoder.py`
- Modify: `Group_B_working_code.ipynb` analysis and plotting cells
- Test: `tests/test_cross_session_area_decoder.py`

**Interfaces:**
- Consumes: `evaluate_bidirectional_transfer` and the existing `X_bef`, `X_aft`, `y_bef`, `y_aft` per area.
- Produces: five `after_to_before_*` columns and nested artifacts for both directions.

- [ ] **Step 1: Add a failing notebook-integration test**

Append:

```python
def test_notebook_reports_both_transfer_directions():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    required = [
        "evaluate_bidirectional_transfer(",
        '"after_to_before_balanced_accuracy"',
        '"after_to_before_roc_auc"',
        '"after_to_before_bootstrap_ci_low"',
        '"after_to_before_bootstrap_ci_high"',
        '"after_to_before_permutation_p"',
        '"After → before"',
    ]
    for value in required:
        assert value in source
```

- [ ] **Step 2: Run the integration test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/nma-decoder-venv/bin/python \
  -m pytest -p no:cacheprovider \
  tests/test_cross_session_area_decoder.py::test_notebook_reports_both_transfer_directions -v
```

Expected: FAIL because the analysis cell has no reverse result columns.

- [ ] **Step 3: Replace the per-area single-direction call**

Use:

```python
direction_metrics, direction_artifacts = evaluate_bidirectional_transfer(
    X_bef,
    y_bef,
    X_aft,
    y_aft,
    n_bootstrap=2000,
    n_permutations=1000,
)
metrics = direction_metrics["before_to_after"]
reverse_metrics = direction_metrics["after_to_before"]
```

Add these result fields to the row while retaining the existing forward fields:

```python
"after_to_before_balanced_accuracy": reverse_metrics[
    "transfer_balanced_accuracy"
],
"after_to_before_roc_auc": reverse_metrics["transfer_roc_auc"],
"after_to_before_bootstrap_ci_low": reverse_metrics[
    "bootstrap_ci_low"
],
"after_to_before_bootstrap_ci_high": reverse_metrics[
    "bootstrap_ci_high"
],
"after_to_before_permutation_p": reverse_metrics["permutation_p"],
```

Store artifacts with:

```python
cross_session_artifacts[area] = {
    **direction_artifacts,
    "X_before": X_bef,
    "X_after": X_aft,
    "y_before": y_bef,
    "y_after": y_aft,
}
```

- [ ] **Step 4: Extend the table, progress line, and plots**

Add these fields to the displayed table after the existing forward fields:

```python
"after_to_before_balanced_accuracy",
"after_to_before_roc_auc",
"after_to_before_bootstrap_ci_low",
"after_to_before_bootstrap_ci_high",
"after_to_before_permutation_p",
```

Use this progress line inside the area loop:

```python
print(
    f"{area:>10s} | "
    f"before CV={metrics['before_cv_balanced_accuracy']:.3f} | "
    f"after CV={metrics['after_cv_balanced_accuracy']:.3f} | "
    f"before→after={metrics['transfer_balanced_accuracy']:.3f} "
    f"(p={metrics['permutation_p']:.4f}) | "
    f"after→before={reverse_metrics['transfer_balanced_accuracy']:.3f} "
    f"(p={reverse_metrics['permutation_p']:.4f})"
)
```

Replace the one-direction transfer figure with:

```python
forward_transfer = cross_session_results[
    "transfer_balanced_accuracy"
].to_numpy()
forward_low = cross_session_results["bootstrap_ci_low"].to_numpy()
forward_high = cross_session_results["bootstrap_ci_high"].to_numpy()
reverse_transfer = cross_session_results[
    "after_to_before_balanced_accuracy"
].to_numpy()
reverse_low = cross_session_results[
    "after_to_before_bootstrap_ci_low"
].to_numpy()
reverse_high = cross_session_results[
    "after_to_before_bootstrap_ci_high"
].to_numpy()

fig, ax = plt.subplots(figsize=(8, 4.5))
offset = 0.08
ax.errorbar(
    x - offset,
    forward_transfer,
    yerr=np.vstack(
        [
            forward_transfer - forward_low,
            forward_high - forward_transfer,
        ]
    ),
    fmt="o",
    capsize=4,
    linewidth=1.5,
    label="Before → after",
)
ax.errorbar(
    x + offset,
    reverse_transfer,
    yerr=np.vstack(
        [
            reverse_transfer - reverse_low,
            reverse_high - reverse_transfer,
        ]
    ),
    fmt="o",
    capsize=4,
    linewidth=1.5,
    label="After → before",
)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
ax.set_xticks(x, area_labels, rotation=20)
ax.set_ylim(0, 1)
ax.set_ylabel("Balanced accuracy")
ax.set_title("Bidirectional cross-session decoding")
ax.legend()
plt.tight_layout()
plt.show()
```

Use four bars per area:

```python
fig, ax = plt.subplots(figsize=(9, 4.5))
width = 0.2
metric_specs = [
    ("before_cv_balanced_accuracy", "Before CV"),
    ("after_cv_balanced_accuracy", "After CV"),
    ("transfer_balanced_accuracy", "Before → after"),
    ("after_to_before_balanced_accuracy", "After → before"),
]
offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
for offset_value, (column, label) in zip(offsets, metric_specs):
    ax.bar(
        x + offset_value,
        cross_session_results[column],
        width=width,
        label=label,
    )
ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
ax.set_xticks(x, area_labels, rotation=20)
ax.set_ylim(0, 1)
ax.set_ylabel("Balanced accuracy")
ax.set_title("Within-session and bidirectional cross-session decoding")
ax.legend()
plt.tight_layout()
plt.show()
```

For confusion matrices and permutation distributions, define:

```python
direction_specs = [
    ("before_to_after", "Before → after"),
    ("after_to_before", "After → before"),
]
```

Then draw the confusion matrices with:

```python
fig, axes = plt.subplots(
    len(direction_specs),
    len(area_labels),
    figsize=(3.2 * len(area_labels), 6),
)
for row_index, (direction_key, direction_label) in enumerate(direction_specs):
    for column_index, area in enumerate(area_labels):
        ax = axes[row_index, column_index]
        matrix = cross_session_artifacts[area][direction_key][
            "confusion_matrix"
        ]
        image = ax.imshow(matrix, cmap="Blues")
        for true_class in range(2):
            for predicted_class in range(2):
                ax.text(
                    predicted_class,
                    true_class,
                    matrix[true_class, predicted_class],
                    ha="center",
                    va="center",
                    color="black",
                )
        ax.set_xticks([0, 1], ["circle1", "leaf1"], rotation=30)
        ax.set_yticks([0, 1], ["circle1", "leaf1"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{area}\n{direction_label}")
fig.colorbar(image, ax=axes, shrink=0.75)
plt.show()
```

Draw the permutation distributions with:

```python
fig, axes = plt.subplots(
    len(direction_specs),
    len(area_labels),
    figsize=(3.2 * len(area_labels), 6),
)
for row_index, (direction_key, direction_label) in enumerate(direction_specs):
    for column_index, area in enumerate(area_labels):
        ax = axes[row_index, column_index]
        direction_artifacts = cross_session_artifacts[area][direction_key]
        null_scores = direction_artifacts["permutation_scores"]
        if direction_key == "before_to_after":
            observed_column = "transfer_balanced_accuracy"
            p_column = "permutation_p"
        else:
            observed_column = "after_to_before_balanced_accuracy"
            p_column = "after_to_before_permutation_p"
        observed = cross_session_results.loc[
            cross_session_results["area"] == area,
            observed_column,
        ].iloc[0]
        p_value = cross_session_results.loc[
            cross_session_results["area"] == area,
            p_column,
        ].iloc[0]
        ax.hist(null_scores, bins=25, color="0.75", edgecolor="white")
        ax.axvline(observed, color="tab:red", linewidth=2)
        ax.set_title(f"{area}\n{direction_label}: p={p_value:.4f}")
        ax.set_xlabel("Permuted balanced accuracy")
        ax.set_ylabel("Count")
plt.tight_layout()
plt.show()
```

- [ ] **Step 5: Run the full tests and notebook validation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/nma-decoder-venv/bin/python \
  -m pytest -p no:cacheprovider tests/test_cross_session_area_decoder.py -v
/tmp/nma-decoder-venv/bin/python - <<'PY'
import nbformat
notebook = nbformat.read("Group_B_working_code.ipynb", as_version=4)
nbformat.validate(notebook)
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"cell_{index}", "exec")
print("notebook_validation=PASS")
PY
git diff --check
```

Expected: all tests PASS, notebook validation prints `PASS`, and `git diff --check` exits 0.

- [ ] **Step 6: Commit the integrated notebook output**

```bash
git add tests/test_cross_session_area_decoder.py Group_B_working_code.ipynb
git commit -m "feat: report reverse decoder transfer"
```

---

### Task 3: Final reproducibility verification

**Files:**
- Verify: `Group_B_working_code.ipynb`
- Verify: `tests/test_cross_session_area_decoder.py`
- Verify: `docs/superpowers/specs/2026-07-20-cross-session-area-decoder-design.md`

**Interfaces:**
- Consumes: completed bidirectional notebook and tests.
- Produces: evidence that both source-only scalers, statistics, tables, and plots are present without stale decoder code.

- [ ] **Step 1: Run fresh tests without generated cache files**

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/nma-decoder-venv/bin/python \
  -m pytest -p no:cacheprovider tests/test_cross_session_area_decoder.py -v
```

- [ ] **Step 2: Scan direction-specific requirements**

```bash
/tmp/nma-decoder-venv/bin/python - <<'PY'
import nbformat
notebook = nbformat.read("Group_B_working_code.ipynb", as_version=4)
source = "\n".join(cell.source for cell in notebook.cells)
assert source.count("# CROSS_SESSION_AREA_DECODER_UTILS") == 1
assert '"before_to_after"' in source
assert '"after_to_before"' in source
assert '"after_to_before_balanced_accuracy"' in source
assert "%%writefile decoding.py" not in source
assert "from decoding import" not in source
print("bidirectional_transfer_scan=PASS")
PY
```

- [ ] **Step 3: Record the execution gap and inspect Git state**

Do not claim real-data execution locally because `/content/Zhong_et_al_2025` is absent. Run:

```bash
git status --short --branch
git log --oneline -7
```

The full-data validation remains a Colab top-to-bottom run after the Figshare download cells complete.
