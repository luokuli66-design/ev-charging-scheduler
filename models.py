"""
两个独立模型定义（注意力机制完全不同）：
1. MH_ResGAT_Model  (论文1)
   - 无物理先验 Sij
   - 注意力：缩放点积 (Scaled Dot-Product Attention)
2. HAPPO_GNN_RL_Model (论文2)
   - 有物理先验 Sij
   - 注意力：LeakyReLU + 拼接 (Concatenation + LeakyReLU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================
# 物理先验计算 (仅 HAPPO_GNN_RL 使用)
# S_ij = MI(i,j) + Granger(i->j)
# ============================================================
def compute_mutual_info(x, y, bins=10):
    x_d = pd.cut(x, bins=bins, labels=False, duplicates='drop')
    y_d = pd.cut(y, bins=bins, labels=False, duplicates='drop')
    min_len = min(len(x_d), len(y_d))
    x_d, y_d = x_d[:min_len], y_d[:min_len]
    try:
        joint = pd.crosstab(x_d, y_d)
        jp = joint / joint.sum().sum()
        px = jp.sum(axis=1)
        py = jp.sum(axis=0)
        mi = 0.0
        for i in range(len(px)):
            for j in range(len(py)):
                if jp.iloc[i, j] > 0 and px.iloc[i] > 0 and py.iloc[j] > 0:
                    mi += jp.iloc[i, j] * np.log(jp.iloc[i, j] / (px.iloc[i] * py.iloc[j] + 1e-10))
        return max(0, mi / 2.0)
    except:
        return 0.0


def compute_granger_simple(y_cause, y_effect, max_lag=3):
    n = min(len(y_cause), len(y_effect), 100)
    if n < 10:
        return 0.0
    x, y = y_cause[:n], y_effect[:n]
    try:
        X_full = []
        Y_arr = []
        for t in range(max_lag, n):
            row = []
            for lag in range(1, max_lag + 1):
                row.extend([y[t - lag], x[t - lag]])
            X_full.append(row)
            Y_arr.append(y[t])
        X_full = np.array(X_full)
        Y_arr = np.array(Y_arr)
        if X_full.shape[0] < 5:
            return 0.0
        from numpy.linalg import lstsq
        beta_f = lstsq(X_full, Y_arr, rcond=None)[0]
        rss_f = np.sum((Y_arr - X_full @ beta_f) ** 2)
        X_red = []
        for t in range(max_lag, n):
            row = [y[t - lag] for lag in range(1, max_lag + 1)]
            X_red.append(row)
        X_red = np.array(X_red)
        beta_r = lstsq(X_red, Y_arr, rcond=None)[0]
        rss_r = np.sum((Y_arr - X_red @ beta_r) ** 2)
        k1, k0 = X_full.shape[1], X_red.shape[1]
        f = ((rss_r - rss_f) / (k1 - k0 + 1e-10)) / (rss_f / (n - k1 - 1) + 1e-10)
        return max(0, min(1.0, f / 10.0))
    except:
        return 0.0


def compute_physics_prior(df, station_ids):
    n = len(station_ids)
    S = np.zeros((n, n))
    station_series = {}
    for sid in station_ids:
        sub = df[df['station_id'] == sid]
        if 'timestamp' in sub.columns:
            try:
                ts = pd.to_datetime(sub['timestamp'], errors='coerce')
                hourly = pd.DataFrame({'ts': ts, 'demand': sub['charging_demand'].values})
                hourly = hourly.dropna(subset=['ts'])
                hourly['hour'] = hourly['ts'].dt.floor('h')
                station_series[sid] = hourly.groupby('hour')['demand'].mean().sort_index().values
            except:
                station_series[sid] = sub['charging_demand'].values[:100]
        else:
            station_series[sid] = sub['charging_demand'].values[:100]

    for i, si in enumerate(station_ids):
        for j, sj in enumerate(station_ids):
            if i == j:
                S[i][j] = 1.0
                continue
            x = station_series.get(si, np.random.randn(50))
            y = station_series.get(sj, np.random.randn(50))
            min_len = min(len(x), len(y), 100)
            if min_len < 10:
                S[i][j] = 0.0
                continue
            mi = compute_mutual_info(x[:min_len], y[:min_len])
            granger = compute_granger_simple(x[:min_len], y[:min_len])
            S[i][j] = 0.5 * mi + 0.5 * granger

    for i in range(n):
        row = S[i]
        row_exp = np.exp(row - np.max(row))
        S[i] = row_exp / (row_exp.sum() + 1e-8)
    return S


# ============================================================
# 注意力头 1：缩放点积 + LeakyReLU（MH-Res-GAT 使用，对应论文描述）
# 公式：omega_ij = LeakyReLU((W_q h_i)^T (W_k h_j) / sqrt(d))
# ============================================================
class ScaledDotProductAttentionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.1)  # 论文：负轴斜率α=0.1
        self.scale = max(out_dim, 1)

    def forward(self, h, adj):
        """
        MH-Res-GAT：缩放点积 + LeakyReLU（论文公式15）
        omega_ij = LeakyReLU((W_q h_i)^T (W_k h_j) / sqrt(d))
        """
        n = h.shape[0]
        Q = self.W_q(h)   # (n, out_dim)
        K = self.W_k(h)   # (n, out_dim)
        V = self.W_v(h)   # (n, out_dim)

        # 缩放点积注意力分数
        scores = (Q @ K.T) / np.sqrt(self.scale)   # (n, n)
        
        # 【关键修复】添加 LeakyReLU 激活（论文公式15）
        scores = self.leaky_relu(scores)

        # 只保留有边连接的位置
        scores = scores * (adj > 0).float()
        # 自连接
        scores = scores + torch.eye(n, device=h.device) * 10.0
        # Softmax归一化（论文公式16）
        alpha = F.softmax(scores, dim=-1)
        # 聚合
        h_new = alpha @ V
        return h_new, alpha


# ============================================================
# 注意力头 2：LeakyReLU + 拼接（HAPPO-GNN-RL 使用，对应论文2）
# 公式：omega_ij = LeakyReLU(W_a · [h_i ⊕ h_j])
# 融合：e_ij = omega_ij + gamma * S_ij
# ============================================================
class LeakyReLUConcatAttentionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)
        self.W_a = nn.Linear(in_dim * 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, h, adj, S=None, gamma=0.3):
        """
        HAPPO-GNN-RL：LeakyReLU+拼接，可融合物理先验
        """
        n = h.shape[0]
        h_r = h.unsqueeze(1).repeat(1, n, 1)
        h_t = h.unsqueeze(0).repeat(n, 1, 1)
        concat = torch.cat([h_r, h_t], dim=-1)
        omega = self.leaky_relu(self.W_a(concat)).squeeze(-1)

        # 融合物理先验
        if S is not None:
            S_tensor = S if isinstance(S, torch.Tensor) else torch.tensor(S, dtype=torch.float32, device=h.device)
            e = (1 - gamma) * omega + gamma * S_tensor
        else:
            e = omega

        e = e * (adj > 0).float()
        e = e + torch.eye(n, device=h.device) * 1.0
        alpha = F.softmax(e, dim=-1)
        h_new = alpha @ self.W_v(h)
        return h_new, alpha


# ============================================================
# GAT 编码器 1：缩放点积（MH-Res-GAT）
# ============================================================
class ScaledDotProductGATEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.layers = nn.ModuleList([
            ScaledDotProductGATLayer(hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, adj, S=None, gamma=0.0):
        h = F.elu(self.input_proj(x))
        final_attn = None
        for layer in self.layers:
            h, attn = layer(h, adj)
            final_attn = attn
        h_out = self.output_proj(h)
        return h_out, final_attn


class ScaledDotProductGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads  # 添加 n_heads 属性（与 LeakyReLUConcatGATLayer 保持一致）
        self.heads = nn.ModuleList([
            ScaledDotProductAttentionHead(in_dim, out_dim)
            for _ in range(n_heads)
        ])
        self.W_out = nn.Linear(out_dim * n_heads, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(out_dim)

    def forward(self, h, adj):
        head_outs = []
        head_attns = []
        for head in self.heads:
            h_h, alpha_h = head(h, adj)
            head_outs.append(h_h)
            head_attns.append(alpha_h)
        h_c = torch.cat(head_outs, dim=-1)
        h_t = self.dropout(F.elu(self.W_out(h_c)))
        if h.shape[-1] == h_t.shape[-1]:
            h_new = h_t + h
        else:
            h_new = h_t
        h_new = self.batch_norm(h_new)
        attn = torch.stack(head_attns, dim=0).mean(dim=0)
        return h_new, attn


# ============================================================
# GAT 编码器 2：LeakyReLU + 拼接（HAPPO-GNN-RL）
# ============================================================
class LeakyReLUConcatGATEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.layers = nn.ModuleList([
            LeakyReLUConcatGATLayer(hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, adj, S=None, gamma=0.3):
        h = F.elu(self.input_proj(x))
        final_attn = None
        for layer in self.layers:
            h, attn = layer(h, adj, S, gamma)
            final_attn = attn
        h_out = self.output_proj(h)
        return h_out, final_attn


class LeakyReLUConcatGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads  # 添加 n_heads 属性（与 ScaledDotProductGATLayer 保持一致）
        self.heads = nn.ModuleList([
            LeakyReLUConcatAttentionHead(in_dim, out_dim)
            for _ in range(n_heads)
        ])
        self.W_out = nn.Linear(out_dim * n_heads, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(out_dim)

    def forward(self, h, adj, S=None, gamma=0.3):
        head_outs = []
        head_attns = []
        for head in self.heads:
            h_h, alpha_h = head(h, adj, S, gamma)
            head_outs.append(h_h)
            head_attns.append(alpha_h)
        h_c = torch.cat(head_outs, dim=-1)
        h_t = self.dropout(F.elu(self.W_out(h_c)))
        if h.shape[-1] == h_t.shape[-1]:
            h_new = h_t + h
        else:
            h_new = h_t
        h_new = self.batch_norm(h_new)
        attn = torch.stack(head_attns, dim=0).mean(dim=0)
        return h_new, attn


# ============================================================
# 公共策略网络
# ============================================================
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, n_actions=3, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
        self.n_actions = n_actions

    def forward(self, state, mask=None):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        z1 = F.leaky_relu(self.fc1(state), 0.01)
        logits = self.fc2(z1)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs, logits


# ============================================================
# 模型1: MH_ResGAT (论文1 —— 缩放点积注意力，无物理先验)
# ============================================================
class MH_ResGAT_Model(nn.Module):
    def __init__(self, feat_dim, n_stations, n_actions=3,
                 gat_hidden=64, gat_layers=2, gat_heads=4,
                 policy_hidden=128):
        super().__init__()
        self.n_stations = n_stations
        self.n_actions = n_actions
        # 使用缩放点积注意力编码器
        self.gat = ScaledDotProductGATEncoder(
            feat_dim, gat_hidden, gat_layers, gat_heads
        )
        policy_input_dim = gat_hidden + 4
        self.policy = PolicyNetwork(policy_input_dim, n_actions, policy_hidden)
        self.value_net = nn.Sequential(
            nn.Linear(policy_input_dim, policy_hidden),
            nn.LeakyReLU(0.01),
            nn.Linear(policy_hidden, 1)
        )

    def forward(self, x, adj, S, gamma, station_states, action_masks=None):
        # MH-Res-GAT：S和gamma被忽略，只用缩放点积注意力
        h_gat, attn = self.gat(x, adj, S=None, gamma=0.0)
        policy_input = torch.cat([h_gat, station_states], dim=-1)
        action_probs_list = []
        for i in range(self.n_stations):
            mask = action_masks[i:i + 1] if action_masks is not None else None
            probs, logits = self.policy(policy_input[i:i + 1], mask)
            action_probs_list.append(probs)
        action_probs = torch.cat(action_probs_list, dim=0)
        values = self.value_net(policy_input)
        return action_probs, attn, values


# ============================================================
# 模型2: HAPPO_GNN_RL (论文2 —— LeakyReLU拼接注意力，有物理先验融合)
# ============================================================
class HAPPO_GNN_RL_Model(nn.Module):
    def __init__(self, feat_dim, n_stations, n_actions=3,
                 gat_hidden=64, gat_layers=2, gat_heads=4,
                 policy_hidden=128):
        super().__init__()
        self.n_stations = n_stations
        self.n_actions = n_actions
        # 使用 LeakyReLU + 拼接注意力编码器
        self.gat = LeakyReLUConcatGATEncoder(
            feat_dim, gat_hidden, gat_layers, gat_heads
        )
        policy_input_dim = gat_hidden + 4
        self.policy = PolicyNetwork(policy_input_dim, n_actions, policy_hidden)
        self.value_net = nn.Sequential(
            nn.Linear(policy_input_dim, policy_hidden),
            nn.LeakyReLU(0.01),
            nn.Linear(policy_hidden, 1)
        )

    def forward(self, x, adj, S, gamma, station_states, action_masks=None):
        # HAPPO-GNN-RL：融合物理先验
        h_gat, attn = self.gat(x, adj, S, gamma)
        policy_input = torch.cat([h_gat, station_states], dim=-1)
        action_probs_list = []
        for i in range(self.n_stations):
            mask = action_masks[i:i + 1] if action_masks is not None else None
            probs, logits = self.policy(policy_input[i:i + 1], mask)
            action_probs_list.append(probs)
        action_probs = torch.cat(action_probs_list, dim=0)
        values = self.value_net(policy_input)
        return action_probs, attn, values


# ============================================================
# 数据加载和图构建
# ============================================================
def build_graph_from_data(df, station_ids=None, sample_size=2000):
    if station_ids is None:
        station_ids = sorted(df['station_id'].unique())
    n = len(station_ids)
    adj = np.zeros((n, n))
    station_demand = {sid: df[df['station_id'] == sid]['charging_demand'].values for sid in station_ids}
    for i, s1 in enumerate(station_ids):
        for j, s2 in enumerate(station_ids):
            if i == j:
                adj[i][j] = 1.0
                continue
            d1, d2 = station_demand[s1], station_demand[s2]
            m = min(len(d1), len(d2))
            if m < 5:
                adj[i][j] = 0.0
                continue
            c = np.corrcoef(d1[:m], d2[:m])[0, 1]
            adj[i][j] = 0.0 if np.isnan(c) else max(0, c)
    for i in range(n):
        s = adj[i].sum()
        if s > 0:
            adj[i] /= s
    feat_cols = ['waiting_time', 'energy_consumed_kWh', 'charging_power_kW',
                 'charging_duration', 'queue_length', 'station_load',
                 'electricity_price', 'renewable_energy_ratio', 'charging_demand']
    node_features = []
    for sid in station_ids:
        sub = df[df['station_id'] == sid]
        feats = [sub[c].mean() if c in df.columns else 0.0 for c in feat_cols]
        feats.append(len(sub))
        node_features.append(feats)
    node_features = np.array(node_features)
    scaler = StandardScaler()
    node_features_scaled = scaler.fit_transform(node_features)
    return node_features_scaled, adj, station_ids, scaler


def load_trained_model(model_path, model_type='mh_res_gat', n_stations=20, feat_dim=10, n_actions=3):
    """
    加载训练好的模型
    model_type: 'mh_res_gat' 或 'happo_gnn_rl'
    """
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    if model_type == 'mh_res_gat':
        model = MH_ResGAT_Model(
            feat_dim=feat_dim,
            n_stations=n_stations,
            n_actions=n_actions
        )
    elif model_type == 'happo_gnn_rl':
        model = HAPPO_GNN_RL_Model(
            feat_dim=feat_dim,
            n_stations=n_stations,
            n_actions=n_actions
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 尝试直接加载（如果架构匹配）
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # 架构不匹配，尝试权重映射
        print(f"[WARN] 权重映射: {e}")
        mapped = _map_state_dict(state_dict, model)
        model.load_state_dict(mapped)

    model = model.to(device)
    model.eval()
    return model, checkpoint


def _map_state_dict(old_state_dict, model):
    """将旧checkpoint权重映射到新模型结构"""
    new_state_dict = model.state_dict()
    mapped = {}

    for key in new_state_dict:
        if key in old_state_dict:
            old_tensor = old_state_dict[key]
            new_tensor = new_state_dict[key]

            if old_tensor.shape == new_tensor.shape:
                mapped[key] = old_tensor
            else:
                print(f"  Shape mismatch for {key}: old={old_tensor.shape}, new={new_tensor.shape}, skipping")
        else:
            print(f"  Missing key in checkpoint: {key}, using random init")

    for key in mapped:
        new_state_dict[key] = mapped[key]

    return new_state_dict


def run_model_inference(model, df, station_ids, gamma, model_type,
                        sample_size=2000):
    """
    用训练好的模型进行推理
    model_type: 'mh_res_gat' 或 'happo_gnn_rl'
    """
    device = next(model.parameters()).device
    node_features, adj, _, _ = build_graph_from_data(df, station_ids, sample_size)
    n = len(station_ids)

    S = None
    if model_type == 'happo_gnn_rl' and gamma > 0:
        S = compute_physics_prior(df, station_ids)

    x = torch.tensor(node_features, dtype=torch.float32).to(device)
    adj_t = torch.tensor(adj, dtype=torch.float32).to(device)
    S_t = torch.tensor(S, dtype=torch.float32).to(device) if S is not None else None

    station_states = []
    action_masks = []
    for i, sid in enumerate(station_ids):
        sub = df[df['station_id'] == sid]
        sw = float(sub['waiting_time'].mean())
        sp = float(sub['charging_power_kW'].mean())
        sq = float(sub['queue_length'].mean())
        sg = float(sub['renewable_energy_ratio'].mean())
        station_states.append([sw, sp, sq, sg])
        mask = [1.0, 1.0, 1.0]
        if sq < 2:
            mask[0] = 0.0
        if sq > 8:
            mask[2] = 0.0
        action_masks.append(mask)

    station_states = torch.tensor(station_states, dtype=torch.float32).to(device)
    action_masks = torch.tensor(action_masks, dtype=torch.float32).to(device)

    with torch.no_grad():
        action_probs, attn, values = model(x, adj_t, S_t, gamma, station_states, action_masks)

    results = []
    actions = ['降低功率(-20%)', '维持当前功率', '提升功率(+20%)']
    attn_np = attn.cpu().numpy()
    values_np = values.squeeze(-1).cpu().numpy()
    probs_np = action_probs.cpu().numpy()

    for i, sid in enumerate(station_ids):
        sub = df[df['station_id'] == sid]
        probs = probs_np[i]
        best_action_idx = int(np.argmax(probs))
        top_neighbors = []
        for j in range(n):
            if i != j and attn_np[i][j] > 0.01:
                top_neighbors.append({'station': station_ids[j], 'weight': float(attn_np[i][j])})
        top_neighbors.sort(key=lambda x: x['weight'], reverse=True)

        w1, w2, w3, w4 = 1.0, 0.1, 2.0, 5.0
        T_t = float(sub['waiting_time'].mean())
        P_t = float(sub['charging_power_kW'].mean()) * float(sub['electricity_price'].mean())
        Q_t = float(sub['queue_length'].mean())
        G_t = float(sub['renewable_energy_ratio'].mean())

        if best_action_idx == 0:
            T_new = T_t * 1.15
            P_new = P_t * 0.80
            reward_action = -1.0
        elif best_action_idx == 1:
            T_new = T_t * 1.00
            P_new = P_t * 1.00
            reward_action = 0.0
        else:
            T_new = T_t * 0.80
            P_new = P_t * 1.20
            reward_action = 1.0

        wait_penalty = max(0, Q_t - 5) * 0.5
        reward = -w1 * T_new + w2 * P_new - w3 * wait_penalty + w4 * G_t + reward_action

        results.append({
            'station_id': sid,
            'location_type': sub.iloc[0]['location_type'],
            'avg_wait': round(float(sub['waiting_time'].mean()), 2),
            'wait_after_action': round(float(T_new), 2),  # 动作后的预测等待时间
            'avg_power': round(float(sub['charging_power_kW'].mean()), 2),
            'avg_queue': round(float(sub['queue_length'].mean()), 2),
            'avg_energy': round(float(sub['energy_consumed_kWh'].mean()), 2),
            'avg_green': round(float(sub['renewable_energy_ratio'].mean()), 3),
            'total_sessions': int(len(sub)),
            'action': actions[best_action_idx],
            'action_probs': {actions[k]: round(float(probs[k]), 4) for k in range(3)},
            'value': round(float(values_np[i]), 4),
            'reward': round(float(reward), 4),
            'top_neighbors': top_neighbors[:5],
            'model_type': model_type,
        })

    return results, attn_np, node_features, adj
