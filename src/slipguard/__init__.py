"""slipguard — reimbursement slip / invoice fraud detection.

Each detection *approach* is an independent ``Detector`` producing a calibrated
``Signal``. Approaches are not chosen a priori: the eval harness ranks them on a
labelled benchmark (overall + per fraud-subtype, and later under image
laundering) so selection is driven by measured real-world performance.
"""

__version__ = "0.1.0"
