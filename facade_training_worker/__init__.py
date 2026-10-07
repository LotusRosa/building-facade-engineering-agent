"""Version-locked local GPU Worker for Facade Agent.

Production pipelines are enabled one at a time after their protocol, model,
determinism, and result-packaging paths are implemented and tested.
"""

from __future__ import annotations

WORKER_PROTOCOL_VERSION = 1
__version__ = "1.0.0"

__all__ = ["WORKER_PROTOCOL_VERSION", "__version__"]
