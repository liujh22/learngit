from pathlib import Path

import nbformat
import numpy as np
import pytest


NOTEBOOK = Path(__file__).parents[1] / "Group_B_working_code.ipynb"
MARKER = "# CROSS_SESSION_AREA_DECODER_UTILS"


def load_decoder_namespace():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    matches = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and MARKER in cell.source
    ]
    assert len(matches) == 1, (
        f"expected one decoder utility cell, found {len(matches)}"
    )
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
    assert callable(namespace["evaluate_bidirectional_transfer"])


def test_area_masks_use_exact_visual_area_mapping():
    namespace = load_decoder_namespace()
    iarea = np.array([8, 0, 1, 2, 9, 5, 6, 3, 4, -1, 7])
    masks = namespace["make_area_masks"](iarea)

    assert masks["V1"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert masks["medial"].sum() == 4
    assert masks["lateral"].sum() == 2
    assert masks["anterior"].sum() == 2
    assert masks["visual_all"].sum() == 9


def test_area_position_features_equal_explicit_neuron_reconstruction():
    namespace = load_decoder_namespace()
    rng = np.random.default_rng(4)
    U = rng.normal(size=(3, 5))
    component_activity = rng.normal(size=(3, 7, 60))
    component_activity[:, :, 40:60] += np.linspace(-1, 1, 20)
    iarea = np.array([8, 8, 0, 5, 3])

    features, metadata = namespace["area_position_features"](
        U, component_activity, iarea, "V1"
    )

    reconstructed = np.einsum(
        "cn,ctp->ntp", U[:, iarea == 8], component_activity
    )
    area_activity = reconstructed.mean(axis=0)
    gray = area_activity[:, 40:60]
    expected = (area_activity - gray.mean()) / gray.std()

    np.testing.assert_allclose(features, expected[:, :40])
    assert metadata["n_neurons"] == 2


def test_area_position_features_reject_invalid_shapes_and_constant_gray():
    namespace = load_decoder_namespace()
    U = np.ones((2, 3))
    activity = np.ones((2, 4, 60))

    with pytest.raises(ValueError, match="gray_std"):
        namespace["area_position_features"](
            U, activity, np.array([8, 8, 8]), "V1"
        )
    with pytest.raises(ValueError, match="component"):
        namespace["area_position_features"](
            U, np.ones((3, 4, 60)), np.array([8, 8, 8]), "V1"
        )


def test_label_encoding_keeps_only_circle1_and_leaf1():
    namespace = load_decoder_namespace()
    behavior = {
        "WallName": np.array(["circle1", "leaf1", "leaf2", "circle1"])
    }

    labels, keep = namespace["encode_leaf_circle_labels"](behavior)

    assert keep.tolist() == [True, True, False, True]
    assert labels.tolist() == [0, 1, 0]


def test_transfer_fits_scaler_only_on_before_features():
    namespace = load_decoder_namespace()
    rng = np.random.default_rng(5)
    labels_before = np.tile([0, 1], 30)
    labels_after = np.tile([0, 1], 35)
    features_before = (
        rng.normal(scale=0.25, size=(60, 40))
        + labels_before[:, None] * 2.0
    )
    features_after = (
        rng.normal(scale=0.25, size=(70, 40))
        + labels_after[:, None] * 2.0
        + 0.2
    )

    metrics, artifacts = namespace["evaluate_transfer"](
        features_before,
        labels_before,
        features_after,
        labels_after,
        n_bootstrap=20,
        n_permutations=20,
    )

    scaler = artifacts["pipeline"].named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, features_before.mean(axis=0))
    assert metrics["transfer_balanced_accuracy"] > 0.9
    assert 0.0 <= metrics["permutation_p"] <= 1.0
    assert len(artifacts["bootstrap_scores"]) == 20
    assert len(artifacts["permutation_scores"]) == 20


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
    np.testing.assert_allclose(
        forward_scaler.mean_, features_before.mean(axis=0)
    )
    np.testing.assert_allclose(
        reverse_scaler.mean_, features_after.mean(axis=0)
    )
    assert set(metrics) == {"before_to_after", "after_to_before"}
    assert len(artifacts["before_to_after"]["bootstrap_scores"]) == 10
    assert len(artifacts["after_to_before"]["permutation_scores"]) == 10
