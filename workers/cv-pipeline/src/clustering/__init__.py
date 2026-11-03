"""Clustering package for team and goalkeeper classification."""

from clustering.team_classifier import (
    DEFAULT_SIGLIP_MODEL_NAME,
    SigLIPEmbedder,
    TeamClassifier,
    crop_upper_torso,
)

__all__ = [
    "DEFAULT_SIGLIP_MODEL_NAME",
    "SigLIPEmbedder",
    "TeamClassifier",
    "crop_upper_torso",
]
