"""Export production 12-D DFMGNN checkpoint to ONNX (matches inference.py)."""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.ml.gnn_model import DFMGNN

MODEL_DIR = "F:/Varroc/data/models"
CHECKPOINT = os.path.join(MODEL_DIR, "gnn_model_real_cpu.pt")
ONNX_OUT = os.path.join(MODEL_DIR, "gnn_model_real.onnx")

if not os.path.exists(CHECKPOINT):
    print(f"ERROR: checkpoint not found: {CHECKPOINT}")
    print("Run scripts/retrain_12d.py first.")
    sys.exit(1)

model = DFMGNN(node_in=12, hidden=64, out_dim=1, heads=2)
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
model.eval()

dummy_x = torch.randn(8, 12)
dummy_edge = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
dummy_batch = torch.zeros(8, dtype=torch.long)

torch.onnx.export(
    model,
    (dummy_x, dummy_edge, dummy_batch),
    ONNX_OUT,
    input_names=["x", "edge_index", "batch"],
    output_names=["risk_score"],
    dynamic_axes={
        "x": {0: "num_nodes"},
        "edge_index": {1: "num_edges"},
        "batch": {0: "num_nodes"},
    },
    opset_version=17,
)
print(f"ONNX exported: {ONNX_OUT}  (12-D DFMGNN)")
print(f"Size: {os.path.getsize(ONNX_OUT)} bytes")

try:
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(ONNX_OUT)
    inp_shape = session.get_inputs()[0].shape
    print(f"ONNX input shape: {inp_shape}")
    result = session.run(None, {
        "x": dummy_x.numpy(),
        "edge_index": dummy_edge.numpy(),
        "batch": dummy_batch.numpy(),
    })
    print(f"ONNX Runtime OK: risk_score={result[0]}")
except ImportError:
    print("onnxruntime not installed — skip verify")
except Exception as e:
    print(f"ONNX verify error: {e}")
