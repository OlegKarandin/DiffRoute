"""Trajectory visualization: the frame dump and its layout.

Phase 1d, demo only. Nothing here is imported by the training path unless
`viz.dump_frames` is enabled in the config.
"""
from diffopt.viz.frames import FrameWriter  # noqa: F401
from diffopt.viz.layout import frozen_layout  # noqa: F401

__all__ = ["FrameWriter", "frozen_layout"]
