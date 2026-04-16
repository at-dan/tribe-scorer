"""
Brain region definitions and vertex-to-ROI mapping.

Maps fsaverage5 cortical vertices to functional metric groups using the
Destrieux atlas (aparc.a2009s). Each metric corresponds to a set of
brain regions whose activation pattern is relevant to a creative-evaluation
dimension (attention, emotion, memorability, etc.).
"""

import numpy as np

# ── Metric definitions ─────────────────────────────────────────────────────
# Each entry maps a scoring dimension to the Destrieux atlas regions whose
# cortical activation is most informative for that dimension.

METRICS = {
    "attention": {
        "label": "Attention",
        "description": "Visual processing depth and attentional capture",
        "weight": 1.2,
        "regions": [
            # Occipital — primary and secondary visual cortex
            "S_calcarine",                  # V1
            "G_cuneus",                     # V2 / V3
            "Pole_occipital",               # Occipital pole
            "G_occipital_middle",           # V3 / V4
            "G_occipital_sup",              # Dorsal visual stream
            "G_oc-temp_med-Lingual",        # Lingual gyrus
            "S_oc_middle_and_Lunatus",
            "S_oc_sup_and_transversal",
            # Parietal — top-down attentional control
            "G_parietal_sup",               # Superior parietal lobule
            "S_intrapariet_and_P_trans",    # Intraparietal sulcus
        ],
    },
    "emotional_impact": {
        "label": "Emotion",
        "description": "Emotional arousal, valence, and salience processing",
        "weight": 1.0,
        "regions": [
            # Cingulate — conflict monitoring, emotional processing
            "G_and_S_cingul-Ant",
            "G_and_S_cingul-Mid-Ant",
            # Insula — emotional awareness, interoception
            "G_insular_short",
            "G_Ins_lg_and_S_cent_ins",
            "S_circular_insula_ant",
            "S_circular_insula_inf",
            # Orbitofrontal — emotional valuation, reward
            "G_orbital",
            "S_orbital_lateral",
            "S_orbital-H_Shaped",
            "G_rectus",
            "G_subcallosal",
            "S_suborbital",
        ],
    },
    "memorability": {
        "label": "Memorability",
        "description": "Memory encoding and narrative retention likelihood",
        "weight": 1.2,
        "regions": [
            # Medial temporal — memory encoding
            "G_oc-temp_med-Parahip",        # Parahippocampal gyrus
            "Pole_temporal",                # Temporal pole
            "S_collat_transv_ant",          # Near hippocampus
            # Precuneus / posterior cingulate — episodic memory, self-reference
            "G_precuneus",
            "S_subparietal",
            "G_cingul-Post-dorsal",
            "G_cingul-Post-ventral",
        ],
    },
    "cognitive_engagement": {
        "label": "Cognition",
        "description": "Active evaluation, reasoning, and decision processing",
        "weight": 0.9,
        "regions": [
            # Dorsolateral prefrontal — executive function
            "G_front_middle",
            "S_front_middle",
            "S_front_sup",
            # Inferior frontal — cognitive control
            "G_front_inf-Opercular",
            "G_front_inf-Triangul",
            # Parietal — working memory
            "G_pariet_inf-Angular",
            "G_pariet_inf-Supramar",
        ],
    },
    "social_resonance": {
        "label": "Social",
        "description": "Social cognition, face processing, and voice sensitivity",
        "weight": 1.0,
        "regions": [
            # Superior temporal — social brain, voice processing
            "G_temp_sup-Lateral",
            "S_temporal_sup",
            "G_temp_sup-Plan_tempo",
            "G_temp_sup-Plan_polar",
            # Fusiform — face processing
            "G_oc-temp_lat-fusifor",
            # Middle temporal — theory of mind, social inference
            "G_temporal_middle",
            "S_oc-temp_lat",
        ],
    },
    "language_clarity": {
        "label": "Language",
        "description": "Speech comprehension and linguistic engagement",
        "weight": 0.8,
        "regions": [
            # Broca's area
            "G_front_inf-Opercular",
            "G_front_inf-Triangul",
            "S_front_inf",
            # Auditory cortex
            "G_temp_sup-G_T_transv",        # Heschl's gyrus
            "S_temporal_transverse",
            # Semantic processing
            "G_pariet_inf-Angular",
        ],
    },
}

# Number of vertices per hemisphere in fsaverage5
FSAVERAGE5_VERTICES_PER_HEMI = 10242


def load_atlas() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the Destrieux atlas for fsaverage5.

    Returns:
        (labels_lh, labels_rh, label_names) where labels are integer arrays
        of shape (10242,) and label_names maps index → region name.
    """
    from nilearn.datasets import fetch_atlas_surf_destrieux

    atlas = fetch_atlas_surf_destrieux()

    # Handle different nilearn API versions for label maps
    if hasattr(atlas, "map_left"):
        labels_lh = np.asarray(atlas.map_left)
        labels_rh = np.asarray(atlas.map_right)
    elif hasattr(atlas, "labels_left"):
        labels_lh = np.asarray(atlas.labels_left)
        labels_rh = np.asarray(atlas.labels_right)
    else:
        raise RuntimeError(
            f"Unexpected Destrieux atlas format. Available keys: {list(atlas.keys())}"
        )

    # Handle different nilearn API versions for label names
    raw_labels = None
    for attr in ("labels", "label_names", "region_names"):
        if hasattr(atlas, attr):
            raw_labels = getattr(atlas, attr)
            break
    if raw_labels is None:
        raise RuntimeError(
            f"Cannot find label names in atlas. Available keys: {list(atlas.keys())}"
        )

    # Normalize: could be list of strings, list of tuples, or list of bytes
    label_names = []
    for entry in raw_labels:
        if isinstance(entry, bytes):
            label_names.append(entry.decode("utf-8"))
        elif isinstance(entry, (tuple, list)):
            label_names.append(str(entry[-1]))  # (index, name) tuple
        else:
            label_names.append(str(entry))

    return labels_lh, labels_rh, label_names


def build_roi_masks(n_vertices: int) -> dict[str, np.ndarray]:
    """
    Build boolean masks mapping each metric to vertex indices.

    Args:
        n_vertices: total vertex count from TRIBE V2 prediction output
                    (expected ~20484 for fsaverage5)

    Returns:
        {metric_name: bool array of shape (n_vertices,)}
    """
    labels_lh, labels_rh, label_names = load_atlas()
    labels_combined = np.concatenate([labels_lh, labels_rh])

    # Handle size mismatch between model output and atlas
    if len(labels_combined) != n_vertices:
        if n_vertices < len(labels_combined):
            labels_combined = labels_combined[:n_vertices]
        else:
            labels_combined = np.pad(
                labels_combined, (0, n_vertices - len(labels_combined))
            )

    # Build name → atlas-index lookup
    name_to_idx: dict[str, int] = {}
    for i, name in enumerate(label_names):
        clean = name.strip()
        name_to_idx[clean] = i

    masks = {}
    for metric_name, defn in METRICS.items():
        mask = np.zeros(n_vertices, dtype=bool)
        matched = 0
        for region in defn["regions"]:
            if region in name_to_idx:
                idx = name_to_idx[region]
                mask |= labels_combined == idx
                matched += 1
        if matched == 0:
            import warnings
            warnings.warn(f"No atlas regions matched for metric '{metric_name}'")
        masks[metric_name] = mask

    return masks
