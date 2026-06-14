import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

class DFMGNN(torch.nn.Module):
    """
    3-layer GATv2 with residual skip connections.

    Each GATv2 block:  out = BN(conv(x)) + linear_proj(x)
    This lets face context propagate further across the graph
    without vanishing gradients.
    """
    def __init__(self, node_in=12, hidden=128, out_dim=1, heads=4, dropout=0.3):
        super().__init__()

        # --- Layer 1 ---
        self.conv1 = GATv2Conv(node_in, hidden // heads, heads=heads, concat=True)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.skip1 = torch.nn.Linear(node_in, hidden)  # residual projection

        # --- Layer 2 ---
        self.conv2 = GATv2Conv(hidden, hidden // heads, heads=heads, concat=True)
        self.bn2 = torch.nn.BatchNorm1d(hidden)
        self.skip2 = torch.nn.Linear(hidden, hidden)  # residual projection

        # --- Layer 3 ---
        layer3_out = hidden // 2
        self.conv3 = GATv2Conv(hidden, layer3_out // heads, heads=heads, concat=True)
        self.bn3_conv = torch.nn.BatchNorm1d(layer3_out)
        self.skip3 = torch.nn.Linear(hidden, layer3_out)  # residual projection

        # --- Readout MLP ---
        self.lin1 = torch.nn.Linear(layer3_out * 2, 64)  # mean_pool + max_pool
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.lin2 = torch.nn.Linear(64, 32)
        self.lin3 = torch.nn.Linear(32, out_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        # Block 1: conv + residual skip
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.leaky_relu(h + self.skip1(x), negative_slope=0.01)
        h = self.dropout(h)

        # Block 2: conv + residual skip
        h_in = h
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.leaky_relu(h + self.skip2(h_in), negative_slope=0.01)
        h = self.dropout(h)

        # Block 3: conv + residual skip
        h_in = h
        h = self.conv3(h, edge_index)
        h = self.bn3_conv(h)
        h = F.leaky_relu(h + self.skip3(h_in), negative_slope=0.01)

        # Pooling: concat(mean, max)
        if torch.onnx.is_in_onnx_export():
            mean_pool = h.mean(dim=0, keepdim=True) + 0.0 * batch.sum()
            max_pool = h.max(dim=0, keepdim=True)[0]
        else:
            mean_pool = global_mean_pool(h, batch)
            max_pool = global_max_pool(h, batch)
        x_pool = torch.cat([mean_pool, max_pool], dim=1)

        # MLP head
        x_pool = self.lin1(x_pool)
        x_pool = self.bn3(x_pool)
        x_pool = F.leaky_relu(x_pool, negative_slope=0.01)
        x_pool = self.dropout(x_pool)
        x_pool = self.lin2(x_pool)
        x_pool = F.relu(x_pool)
        x_pool = self.lin3(x_pool)
        return x_pool.squeeze(-1)

    def get_face_embeddings_and_scores(self, x, edge_index):
        # Block 1
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.leaky_relu(h + self.skip1(x), negative_slope=0.01)
        h = self.dropout(h)

        # Block 2
        h_in = h
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.leaky_relu(h + self.skip2(h_in), negative_slope=0.01)
        h = self.dropout(h)

        # Block 3
        h_in = h
        h = self.conv3(h, edge_index)
        h = self.bn3_conv(h)
        embeddings = F.leaky_relu(h + self.skip3(h_in), negative_slope=0.01)

        # Per-node scoring via MLP (duplicate to match 2*dim expected by lin1)
        node_x = torch.cat([embeddings, embeddings], dim=1)
        node_x = self.lin1(node_x)
        node_x = self.bn3(node_x)
        node_x = F.leaky_relu(node_x, negative_slope=0.01)
        node_x = self.dropout(node_x)
        node_x = self.lin2(node_x)
        node_x = F.relu(node_x)
        node_x = self.lin3(node_x)
        face_scores = torch.sigmoid(node_x).squeeze(-1)

        return embeddings, face_scores

    def get_node_features(self, face):
        """Extract 12-D feature vector (width and curv_max dropped)."""
        # Clean None values so that .get("key", default) correctly falls back
        face = {k: v for k, v in face.items() if v is not None}

        type_map = {
            "Plane": 0, "Cylinder": 1, "Sphere": 2, "Cone": 3,
            "Torus": 4, "BSpline": 5, "Other": 6
        }
        ft = type_map.get(face.get("face_type", "Other"), 6)
        t = min(face.get("thickness_mm", 3.0) / 5.0, 1.0)
        r = min(face.get("radius_mm", 5.0) / 10.0, 1.0) if face.get("radius_mm") else 0.5
        a = min(face.get("area_mm2", 1000) / 10000.0, 1.0)
        d = min(face.get("depth_mm", 10.0) / 50.0, 1.0) if face.get("depth_mm") else 0.3
        # width (old dim5) — DROPPED (r=0.91 with depth)
        cm = min(abs(face.get("curvature_min", 0)) / 10.0, 1.0)
        # curv_max (old dim7) — DROPPED (r=0.97 with curv_min)

        draft = min(abs(face.get("draft_angle_deg", 0)) / 45.0, 1.0) if face.get("draft_angle_deg") else 0.0
        pwt = min(face.get("parent_wall_thickness_mm", 3.0) / 5.0, 1.0) if face.get("parent_wall_thickness_mm") else 0.5
        cx_c = abs(face.get("centroid_x", 0)) / 200.0
        cy_c = abs(face.get("centroid_y", 0)) / 200.0
        cz_c = abs(face.get("centroid_z", 0)) / 200.0
        centroid_dist = min((cx_c**2 + cy_c**2 + cz_c**2) ** 0.5, 1.0)
        aspect = min(d / (min(face.get("width_mm", 10.0) / 50.0, 1.0) + 0.001), 1.0) if face.get("width_mm", 10.0) > 0 else 0.0

        # 12-D: [face_type, thickness, radius, area, depth, curv_min,
        #         draft_angle, parent_wall, centroid_dist, aspect_ratio,
        #         min_plane_draft, frac_plane]  — last 2 added by inference.py
        return [ft / 7.0, t, r, a, d, cm, draft, pwt, centroid_dist, aspect]
