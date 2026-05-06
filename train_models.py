"""
HAPPO-GNN-RL 真实模型实现
包含：
1. MH-Res-GAT (多头残差图注意力网络, gamma=0)
2. 物理先验+动态学习融合模型 (gamma>0)

严格对应公式 (1)~(14)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

# ============================================================
# 工具函数：计算物理先验 S_ij (公式1)
# S_ij = MI(i,j) + Granger(i->j)
# ============================================================

def compute_mutual_info(x, y, bins=10):
    """计算互信息 MI(i,j)"""
    x_discrete = pd.cut(x, bins=bins, labels=False)
    y_discrete = pd.cut(y, bins=bins, labels=False)
    # 联合分布
    joint = pd.crosstab(x_discrete, y_discrete)
    joint_prob = joint / joint.sum()
    # 边缘分布
    px = joint.sum(axis=1) / joint.sum()
    py = joint.sum(axis=0) / joint.sum()
    # MI
    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if joint_prob.iloc[i, j] > 0:
                mi += joint_prob.iloc[i, j] * np.log(joint_prob.iloc[i, j] / (px.iloc[i] * py.iloc[j] + 1e-10))
    return mi


def compute_granger_causality(y_cause, y_effect, max_lag=3):
    """
    简化Granger因果检验
    返回 F-statistic (越大说明因果关系越强)
    """
    from numpy.linalg import lstsq
    n = min(len(y_cause), len(y_effect))
    y_c = y_cause[:n]
    y_e = y_effect[:n]
    
    # 全模型：用y_e的滞后 + y_c的滞后来预测y_e
    X_full = []
    Y = y_e[max_lag:]
    for t in range(max_lag, n):
        row = []
        for lag in range(1, max_lag+1):
            row.append(y_e[t-lag])      # y_e的滞后
            row.append(y_c[t-lag])      # y_c的滞后（因果项）
        X_full.append(row)
    X_full = np.array(X_full)
    
    # 简化模型：只用y_e的滞后
    X_reduced = []
    for t in range(max_lag, n):
        row = []
        for lag in range(1, max_lag+1):
            row.append(y_e[t-lag])
        X_reduced.append(row)
    X_reduced = np.array(X_reduced)
    Y = np.array(Y)
    
    # RSS
    beta_full = lstsq(X_full, Y, rcond=None)[0]
    rss_full = np.sum((Y - X_full @ beta_full) ** 2)
    beta_reduced = lstsq(X_reduced, Y, rcond=None)[0]
    rss_reduced = np.sum((Y - X_reduced @ beta_reduced) ** 2)
    
    # F统计量
    k1 = X_full.shape[1]
    k0 = X_reduced.shape[1]
    n_samples = len(Y)
    if rss_full < 1e-10:
        return 0.0
    f_stat = ((rss_reduced - rss_full) / (k1 - k0)) / (rss_full / (n_samples - k1 - 1) + 1e-10)
    return max(0, f_stat)


def compute_physics_prior(df, station_ids):
    """
    计算所有站点对的物理先验 S_ij
    返回: n_stations x n_stations 矩阵
    """
    n = len(station_ids)
    S = np.zeros((n, n))
    station_idx = {s: i for i, s in enumerate(station_ids)}
    
    # 对每个站点计算充电需求时间序列
    station_series = {}
    for sid in station_ids:
        sub = df[df['station_id'] == sid]
        # 按时间聚合充电需求
        if 'timestamp' in sub.columns:
            # 解析时间戳
            ts = pd.to_datetime(sub['timestamp'], errors='coerce')
            sub = sub.copy()
            sub['hour'] = ts.dt.floor('h')
            hourly = sub.groupby('hour')['charging_demand'].mean().sort_index()
            station_series[sid] = hourly.values
        else:
            station_series[sid] = sub['charging_demand'].values
    
    print(f"  计算物理先验 S_ij ({n}x{n} 矩阵)...")
    for i, si in enumerate(station_ids):
        for j, sj in enumerate(station_ids):
            if i == j:
                S[i][j] = 1.0
                continue
            x = station_series.get(si, np.random.randn(50))
            y = station_series.get(sj, np.random.randn(50))
            # 对齐长度
            min_len = min(len(x), len(y), 100)
            if min_len < 10:
                S[i][j] = 0.0
                continue
            x = x[:min_len]
            y = y[:min_len]
            
            # MI (归一化到[0,1])
            mi = compute_mutual_info(x, y, bins=10)
            mi_norm = min(1.0, mi / 2.0)  # MI 通常 < 2
            
            # Granger (归一化)
            granger = compute_granger_causality(x, y, max_lag=3)
            granger_norm = min(1.0, granger / 10.0)
            
            S[i][j] = 0.5 * mi_norm + 0.5 * granger_norm
    
    # 每行归一化 (softmax)
    for i in range(n):
        row = S[i]
        row_exp = np.exp(row - np.max(row))
        S[i] = row_exp / (row_exp.sum() + 1e-8)
    
    print(f"  物理先验计算完成")
    return S


# ============================================================
# 多头图注意力层 (公式2~5)
# 每个head独立计算注意力，最后拼接
# ============================================================

class GraphAttentionHead(nn.Module):
    """单头图注意力 (公式2~5)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)   # 值投影 (公式5的W_V)
        self.W_a = nn.Linear(in_dim * 2, 1, bias=False)   # 注意力投影 (公式2的W_a)
        self.leaky_relu = nn.LeakyReLU(0.01)
        
    def forward(self, h, adj, S=None, gamma=0.3):
        """
        h: (n_stations, in_dim)
        adj: (n_stations, n_stations) 邻接矩阵
        S: (n_stations, n_stations) 物理先验
        返回: (n_stations, out_dim)
        """
        n = h.shape[0]
        
        # (公式2) 计算原始注意力分数 omega_ij
        # 用批量计算提高效率
        h_repeat = h.unsqueeze(1).repeat(1, n, 1)  # (n, n, in_dim)
        h_tile = h.unsqueeze(0).repeat(n, 1, 1)      # (n, n, in_dim)
        concat = torch.cat([h_repeat, h_tile], dim=-1)   # (n, n, 2*in_dim)
        omega = self.leaky_relu(self.W_a(concat).squeeze(-1))  # (n, n)
        
        # (公式3) 融合物理先验 e_ij = (1-gamma)*omega_ij + gamma*S_ij
        if S is not None:
            S_t = torch.tensor(S, dtype=torch.float32, device=h.device)
            e = (1 - gamma) * omega + gamma * S_t
        else:
            e = omega
        
        # 用邻接矩阵mask（只保留有连接的边）
        e = e * (adj > 0).float()
        # 自连接
        e = e + torch.eye(n, device=h.device) * 1.0
        
        # (公式4) Softmax归一化 alpha_ij
        alpha = F.softmax(e, dim=-1)  # (n, n)
        
        # (公式5) 特征聚合 h_i' = sum_j alpha_ij * W_V * h_j
        h_trans = self.W_v(h)  # (n, out_dim)
        h_new = alpha @ h_trans  # (n, out_dim)
        
        return h_new, alpha


class MultiHeadResGAT(nn.Module):
    """
    多头残差图注意力层 (公式2~6)
    MH-Res-GAT: 多头 + 残差连接
    """
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([
            GraphAttentionHead(in_dim, out_dim) for _ in range(n_heads)
        ])
        self.W_out = nn.Linear(out_dim * n_heads, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        
    def forward(self, h, adj, S=None, gamma=0.3):
        """
        返回:
          h_new: (n_stations, out_dim)  # 聚合后特征
          attn: (n_stations, n_stations)  # 平均注意力
        """
        head_outputs = []
        head_attns = []
        for head in self.heads:
            h_head, alpha_head = head(h, adj, S, gamma)
            head_outputs.append(h_head)
            head_attns.append(alpha_head)
        
        # 拼接多头 (公式: 多头输出拼接)
        h_concat = torch.cat(head_outputs, dim=-1)  # (n, out_dim * n_heads)
        h_trans = self.W_out(h_concat)               # (n, out_dim)
        h_trans = self.dropout(F.elu(h_trans))       # ELU激活
        
        # (公式6) 残差连接 h_i^(l+1) = sigma(h_i') + h_i^(l)
        if h.shape[-1] == h_trans.shape[-1]:
            h_new = h_trans + h   # 残差 (跳跃连接)
        else:
            h_new = h_trans  # 第一层维度不同，不加残差
        
        h_new = self.batch_norm(h_new)
        
        # 平均多头注意力
        attn = torch.stack(head_attns, dim=0).mean(dim=0)
        
        return h_new, attn


# ============================================================
# 完整的 GAT 编码器 (多层GAT)
# ============================================================

class GATEncoder(nn.Module):
    """
    GAT编码器: 多层MH-Res-GAT
    输出每个站点的图增强特征
    """
    def __init__(self, feat_dim, hidden_dim=64, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.layers = nn.ModuleList([
            MultiHeadResGAT(hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, adj, S=None, gamma=0.3):
        """
        x: (n_stations, feat_dim)
        adj: (n_stations, n_stations)
        S: (n_stations, n_stations) 物理先验 (可为None)
        返回:
          h_out: (n_stations, hidden_dim)
          attn_final: (n_stations, n_stations)
        """
        h = F.elu(self.input_proj(x))  # (n, hidden_dim)
        final_attn = None
        for layer in self.layers:
            h, attn = layer(h, adj, S, gamma)
            final_attn = attn
        h_out = self.output_proj(h)
        return h_out, final_attn


# ============================================================
# 策略网络 (公式7~11)
# MLP: 输入GAT特征+状态 → 动作logits
# ============================================================

class PolicyNetwork(nn.Module):
    """
    策略网络 (公式7)
    输入: 站点特征 (GAT输出 + 实时状态)
    输出: 动作概率分布
    """
    def __init__(self, state_dim, n_actions=3, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
        self.n_actions = n_actions
        
    def forward(self, state, mask=None):
        """
        state: (batch, state_dim) 或 (state_dim,)
        mask: (batch, n_actions) 动作掩码 (0=禁止, 1=允许)
        返回:
          action_probs: (batch, n_actions)
          logits: (batch, n_actions)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        z1 = F.leaky_relu(self.fc1(state), 0.01)  # (公式7: LeakyReLU)
        logits = self.fc2(z1)                      # (公式7: z3)
        
        # (公式8) Softmax 概率分布
        if mask is not None:
            # (公式9) 动作掩码
            mask = mask.to(logits.device)
            logits = logits.masked_fill(mask == 0, -1e9)
        
        action_probs = F.softmax(logits, dim=-1)
        
        # (公式10) 掩码后重归一化 (softmax已经做了)
        
        return action_probs, logits


# ============================================================
# 完整的 HAPPO-GNN-RL 模型
# 包含GAT编码器 + 策略网络
# ============================================================

class HAPPO_GNN_RL(nn.Module):
    """
    HAPPO-GNN-RL 完整模型
    
    两种模式:
    1. MH-Res-GAT (gamma=0): 纯数据驱动，无物理先验
    2. 物理先验+动态学习 (gamma>0): 融合模型
    """
    def __init__(self, feat_dim, n_stations, n_actions=3,
                 gat_hidden=64, gat_layers=2, gat_heads=4,
                 policy_hidden=128):
        super().__init__()
        self.n_stations = n_stations
        self.n_actions = n_actions
        
        # GAT编码器
        self.gat = GATEncoder(feat_dim, gat_hidden, gat_layers, gat_heads)
        
        # 策略网络输入维度 = GAT输出 + 实时状态维度(4个: wait, power, queue, green)
        policy_input_dim = gat_hidden + 4
        self.policy = PolicyNetwork(policy_input_dim, n_actions, policy_hidden)
        
        # 价值网络 (用于PPO的value function)
        self.value_net = nn.Sequential(
            nn.Linear(policy_input_dim, policy_hidden),
            nn.LeakyReLU(0.01),
            nn.Linear(policy_hidden, 1)
        )
        
    def forward(self, x, adj, S, gamma, station_states, action_masks=None):
        """
        x: (n_stations, feat_dim) 节点特征
        adj: (n_stations, n_stations) 邻接矩阵
        S: (n_stations, n_stations) 物理先验
        gamma: 融合系数
        station_states: (n_stations, 4) 每个站点的实时状态
        action_masks: (n_stations, n_actions) 动作掩码
        
        返回:
          action_probs: (n_stations, n_actions)
          attn: (n_stations, n_stations)
          values: (n_stations, 1)
        """
        # GAT编码
        h_gat, attn = self.gat(x, adj, S, gamma)
        
        # 拼接GAT特征 + 实时状态 (公式7输入)
        policy_input = torch.cat([h_gat, station_states], dim=-1)
        
        # 策略网络
        action_probs_list = []
        logits_list = []
        for i in range(self.n_stations):
            mask = action_masks[i:i+1] if action_masks is not None else None
            probs, logits = self.policy(policy_input[i:i+1], mask)
            action_probs_list.append(probs)
            logits_list.append(logits)
        action_probs = torch.cat(action_probs_list, dim=0)
        logits = torch.cat(logits_list, dim=0)
        
        # 价值网络
        values = self.value_net(policy_input)
        
        return action_probs, logits, attn, values


# ============================================================
# 数据预处理：从data.csv构建图
# ============================================================

def build_graph_from_data(df, station_ids=None, sample_size=2000):
    """
    从data.csv构建图结构
    返回:
      node_features: (n_stations, feat_dim)
      adj: (n_stations, n_stations) 邻接矩阵
      station_ids: list
      scaler: 用于反归一化的StandardScaler
    """
    if station_ids is None:
        station_ids = sorted(df['station_id'].unique())
    n = len(station_ids)
    station_idx = {s: i for i, s in enumerate(station_ids)}
    
    # 构建邻接矩阵：基于站点间的充电需求相关性
    # 用charging_demand的相关系数作为边权重
    adj = np.zeros((n, n))
    station_demand = {}
    for sid in station_ids:
        sub = df[df['station_id'] == sid]
        station_demand[sid] = sub['charging_demand'].values
    
    for i, s1 in enumerate(station_ids):
        for j, s2 in enumerate(station_ids):
            if i == j:
                adj[i][j] = 1.0
                continue
            d1 = station_demand[s1]
            d2 = station_demand[s2]
            min_len = min(len(d1), len(d2))
            if min_len < 5:
                adj[i][j] = 0.0
                continue
            corr = np.corrcoef(d1[:min_len], d2[:min_len])[0, 1]
            if np.isnan(corr):
                adj[i][j] = 0.0
            else:
                adj[i][j] = max(0, corr)
    
    # 行归一化
    for i in range(n):
        row_sum = adj[i].sum()
        if row_sum > 0:
            adj[i] /= row_sum
    
    # 构建节点特征 (每个站点的统计特征)
    feat_cols = [
        'waiting_time', 'energy_consumed_kWh', 'charging_power_kW',
        'charging_duration', 'queue_length', 'station_load',
        'electricity_price', 'renewable_energy_ratio', 'charging_demand'
    ]
    
    node_features = []
    for sid in station_ids:
        sub = df[df['station_id'] == sid]
        feats = []
        for c in feat_cols:
            if c in df.columns:
                feats.append(sub[c].mean())
            else:
                feats.append(0.0)
        # 额外特征：记录数量（流量）
        feats.append(len(sub))
        node_features.append(feats)
    
    node_features = np.array(node_features)
    
    # 归一化
    scaler = StandardScaler()
    node_features_scaled = scaler.fit_transform(node_features)
    
    return node_features_scaled, adj, station_ids, scaler, feat_cols + ['flow_count']


# ============================================================
# PPO裁剪损失 (公式12~13)
# ============================================================

def ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_ratio=0.2):
    """
    PPO裁剪损失 (公式13)
    L^CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
    
    old_log_probs: (batch,) 旧策略的log概率
    new_log_probs: (batch,) 新策略的log概率
    advantages: (batch,) 优势函数 A_t
    """
    # 概率比 r_t = pi_new / pi_old
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # 裁剪
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    
    # PPO损失
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    loss = -torch.min(surr1, surr2).mean()
    
    return loss, ratio.mean().item()


def entropy_bonus(logits):
    """熵正则化 (公式13中的H(pi))"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return entropy


# ============================================================
# 训练函数
# ============================================================

def train_model(df, model_type='mh_res_gat', epochs=200, lr=0.001, 
                gamma_val=0.0, n_heads=4, sample_size=2000, save_path='model_checkpoint.pth'):
    """
    训练HAPPO-GNN-RL模型
    
    model_type: 'mh_res_gat' (MH-Res-GAT, gamma=0) 
                或 'fusion' (物理先验+动态学习, gamma>0)
    """
    device = torch.device('cpu')
    
    # 构建图
    print(f"\n[训练] 模型类型: {'MH-Res-GAT' if model_type=='mh_res_gat' else '物理先验+动态学习融合模型'}")
    print(f"[训练] 数据量: {min(sample_size, len(df))} 条")
    
    station_ids = sorted(df['station_id'].unique())
    node_features, adj, station_ids, scaler, feat_names = build_graph_from_data(
        df, station_ids, sample_size
    )
    n_stations = len(station_ids)
    feat_dim = node_features.shape[1]
    
    print(f"[训练] 站点数: {n_stations}, 特征维度: {feat_dim}")
    
    # 计算物理先验 S_ij (仅在 fusion 模式下需要)
    S = None
    if model_type == 'fusion':
        print(f"[训练] 计算物理先验 S_ij...")
        S = compute_physics_prior(df, station_ids)
        gamma = gamma_val
    else:
        gamma = 0.0
        print(f"[训练] MH-Res-GAT模式: 不使用物理先验 (gamma=0)")
    
    # 初始化模型
    model = HAPPO_GNN_RL(
        feat_dim=feat_dim,
        n_stations=n_stations,
        n_actions=3,
        gat_hidden=64,
        gat_layers=2,
        gat_heads=n_heads,
        policy_hidden=128
    )
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    
    # 训练循环
    x = torch.tensor(node_features, dtype=torch.float32).to(device)
    adj_t = torch.tensor(adj, dtype=torch.float32).to(device)
    S_t = torch.tensor(S, dtype=torch.float32).to(device) if S is not None else None
    
    print(f"[训练] 开始训练 {epochs} 轮...")
    rewards_history = []
    loss_history = []
    
    for epoch in range(epochs):
        # 构建每个站点的实时状态 (wait, power, queue, green)
        station_states = []
        action_masks = []
        for i, sid in enumerate(station_ids):
            sub = df[df['station_id'] == sid]
            avg_wait = float(sub['waiting_time'].mean())
            avg_power = float(sub['charging_power_kW'].mean())
            avg_queue = float(sub['queue_length'].mean())
            avg_green = float(sub['renewable_energy_ratio'].mean())
            station_states.append([avg_wait, avg_power, avg_queue, avg_green])
            
            # 动作掩码 (示例：根据队列长度决定哪些动作合法)
            mask = [1.0, 1.0, 1.0]  # 默认全允许
            if avg_queue < 2:
                mask[0] = 0.0  # 队列太短，不允许降功率
            if avg_queue > 8:
                mask[2] = 0.0  # 队列太长，不允许升功率
            action_masks.append(mask)
        
        station_states = torch.tensor(station_states, dtype=torch.float32).to(device)
        action_masks = torch.tensor(action_masks, dtype=torch.float32).to(device)
        
        # 前向传播
        action_probs, logits, attn, values = model(x, adj_t, S_t, gamma, station_states, action_masks)
        
        # ========== 修复：奖励必须根据模型动作来计算 ==========
        # 从概率分布采样动作（允许梯度流动）
        dist = torch.distributions.Categorical(action_probs)
        selected_actions = dist.sample()  # 采样而非argmax，保证梯度传递
        selected_action_np = selected_actions.cpu().numpy()
        
        # 根据模型选择的动作，计算模拟后的新状态和奖励
        rewards = []
        for i, sid in enumerate(station_ids):
            sub = df[df['station_id'] == sid]
            # 原始状态（来自历史数据）
            T_t = float(sub['waiting_time'].mean())      # 等待时间(min)
            P_t = float(sub['charging_power_kW'].mean())  # 充电功率(kW)
            L_t = float(sub['station_load'].mean())       # 负荷率
            G_t = float(sub['renewable_energy_ratio'].mean())  # 绿能占比
            Q_t = float(sub['queue_length'].mean())       # 队列长度
            
            action = selected_action_np[i]
            
            # === 根据动作模拟下一个状态 ===
            # 动作 0: 降低功率(-20%) -> 队列消化慢，等待时间增加
            # 动作 1: 维持功率 -> 保持当前状态
            # 动作 2: 提升功率(+20%) -> 队列消化快，等待时间减少
            if action == 0:  # 降低功率
                T_new = T_t * 1.15    # 等待时间 +15%
                P_new = P_t * 0.80    # 功率 -20%
                reward_action = -1.0   # 动作惩罚（不应该轻易降功率）
            elif action == 1:  # 维持功率
                T_new = T_t * 1.00    # 等待时间不变
                P_new = P_t * 1.00
                reward_action = 0.0    # 无惩罚无奖励
            else:  # 动作 == 2, 提升功率
                T_new = T_t * 0.80    # 等待时间 -20%（目标效果）
                P_new = P_t * 1.20    # 功率 +20%
                reward_action = 1.0    # 动作奖励（鼓励提升功率缓解等待）
            
            # === 奖励函数：等待时间↓ 好，绿能↑ 好，降功率惩罚 ===
            # R_t = -(w1*T_t) + w2*P_t - w3*wait_penalty + w4*G_t + reward_action
            w1, w2, w3, w4 = 1.0, 0.1, 2.0, 5.0
            wait_penalty = max(0, Q_t - 5) * 0.5  # 队列过长惩罚
            reward = -w1 * T_new + w2 * P_new - w3 * wait_penalty + w4 * G_t + reward_action
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards_history.append(rewards.mean().item())
        
        # 优势函数 (简化: A = R - V)
        advantages = rewards - values.squeeze(-1)
        
        # 计算策略损失 (PPO)
        # 使用采样动作的log概率（有梯度）
        log_probs = torch.log(action_probs + 1e-8)
        selected_log_probs = log_probs.gather(1, selected_actions.unsqueeze(-1)).squeeze(-1)
        
        # PPO loss: 最大化 log_prob * advantage，同时保持一定探索（entropy bonus）
        policy_loss = -(selected_log_probs * advantages.detach()).mean()
        entropy = entropy_bonus(logits)
        loss = policy_loss - 0.01 * entropy
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_reward = rewards_history[-1]
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"Loss={loss.item():.4f}, Avg_Reward={avg_reward:.4f}, "
                  f"Entropy={entropy.item():.4f}")
    
    # 保存模型
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'station_ids': station_ids,
        'feat_dim': feat_dim,
        'n_stations': n_stations,
        'gamma': gamma,
        'model_type': model_type,
        'scaler_params': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        },
        'epoch': epochs,
        'rewards_history': rewards_history,
        'loss_history': loss_history,
    }
    torch.save(checkpoint, save_path)
    print(f"\n[训练] 模型已保存到: {save_path}")
    
    # 保存训练曲线
    curve_path = save_path.replace('.pth', '_curve.csv')
    import csv
    with open(curve_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'avg_reward', 'loss'])
        for ep in range(epochs):
            writer.writerow([ep+1, rewards_history[ep], loss_history[ep]])
    print(f"[训练] 训练曲线已保存到: {curve_path}")
    
    return model, rewards_history, loss_history


# ============================================================
# 模型加载与推理
# ============================================================

def load_model(model_path, device='cpu'):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = HAPPO_GNN_RL(
        feat_dim=checkpoint['feat_dim'],
        n_stations=checkpoint['n_stations'],
        n_actions=3,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def run_inference(model, checkpoint, df, station_ids=None, gamma=None):
    """
    用训练好的模型进行推理
    返回每个站点的调度决策
    """
    device = next(model.parameters()).device
    
    if station_ids is None:
        station_ids = checkpoint['station_ids']
    
    # 重新构建图
    node_features, adj, _, _, _ = build_graph_from_data(df, station_ids)
    n = len(station_ids)
    
    # 物理先验
    S = None
    if gamma is None:
        gamma = checkpoint.get('gamma', 0.0)
    if gamma > 0:
        S = compute_physics_prior(df, station_ids)
    
    x = torch.tensor(node_features, dtype=torch.float32).to(device)
    adj_t = torch.tensor(adj, dtype=torch.float32).to(device)
    S_t = torch.tensor(S, dtype=torch.float32).to(device) if S is not None else None
    
    # 构建状态
    station_states = []
    action_masks = []
    for i, sid in enumerate(station_ids):
        sub = df[df['station_id'] == sid]
        avg_wait = float(sub['waiting_time'].mean())
        avg_power = float(sub['charging_power_kW'].mean())
        avg_queue = float(sub['queue_length'].mean())
        avg_green = float(sub['renewable_energy_ratio'].mean())
        station_states.append([avg_wait, avg_power, avg_queue, avg_green])
        action_masks.append([1.0, 1.0, 1.0])
    
    station_states = torch.tensor(station_states, dtype=torch.float32).to(device)
    action_masks = torch.tensor(action_masks, dtype=torch.float32).to(device)
    
    # 推理
    with torch.no_grad():
        action_probs, logits, attn, values = model(x, adj_t, S_t, gamma, station_states, action_masks)
    
    # 解析结果
    results = []
    actions = ['降低功率(-20%)', '维持当前功率', '提升功率(+20%)']
    attn_np = attn.cpu().numpy()
    
    for i, sid in enumerate(station_ids):
        sub = df[df['station_id'] == sid]
        probs = action_probs[i].cpu().numpy()
        best_action_idx = int(np.argmax(probs))
        
        # Top相关站点 (注意力权重)
        top_neighbors = []
        for j in range(n):
            if i != j and attn_np[i][j] > 0.01:
                top_neighbors.append({'station': station_ids[j], 'weight': float(attn_np[i][j])})
        top_neighbors.sort(key=lambda x: x['weight'], reverse=True)
        
        results.append({
            'station_id': sid,
            'location_type': sub.iloc[0]['location_type'],
            'avg_wait': round(float(sub['waiting_time'].mean()), 2),
            'avg_power': round(float(sub['charging_power_kW'].mean()), 2),
            'avg_queue': round(float(sub['queue_length'].mean()), 2),
            'avg_energy': round(float(sub['energy_consumed_kWh'].mean()), 2),
            'avg_green': round(float(sub['renewable_energy_ratio'].mean()), 3),
            'total_sessions': int(len(sub)),
            'action': actions[best_action_idx],
            'action_probs': {actions[k]: round(float(probs[k]), 4) for k in range(3)},
            'top_neighbors': top_neighbors[:5],
        })
    
    return results, attn_np


# ============================================================
# 主程序：训练两个模型
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='both', 
                        choices=['mh_res_gat', 'fusion', 'both'],
                        help='训练哪个模型')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--gamma', type=float, default=0.3,
                        help='融合系数')
    parser.add_argument('--heads', type=int, default=4,
                        help='GAT多头数')
    parser.add_argument('--sample', type=int, default=5000,
                        help='采样数据量')
    args = parser.parse_args()
    
    # 加载数据
    print("="*60)
    print("HAPPO-GNN-RL 模型训练")
    print("="*60)
    df = pd.read_csv('data.csv')
    print(f"数据加载完成: {df.shape[0]} 条记录, {len(df['station_id'].unique())} 个充电站")
    
    if args.model_type in ['mh_res_gat', 'both']:
        # 模型1: MH-Res-GAT (gamma=0, 无物理先验)
        print("\n" + "="*60)
        print("训练模型 1: MH-Res-GAT (多头残差注意力)")
        print("="*60)
        model1, r1, l1 = train_model(
            df, model_type='mh_res_gat', epochs=args.epochs, lr=args.lr,
            gamma_val=0.0, n_heads=args.heads, sample_size=args.sample,
            save_path='model_mh_res_gat.pth'
        )
        print(f"模型1训练完成. 最终平均奖励: {r1[-1]:.4f}")
    
    if args.model_type in ['fusion', 'both']:
        # 模型2: 物理先验+动态学习融合模型
        print("\n" + "="*60)
        print(f"训练模型 2: 物理先验+动态学习融合模型 (gamma={args.gamma})")
        print("="*60)
        model2, r2, l2 = train_model(
            df, model_type='fusion', epochs=args.epochs, lr=args.lr,
            gamma_val=args.gamma, n_heads=args.heads, sample_size=args.sample,
            save_path=f'model_fusion_gamma{int(args.gamma*100):03d}.pth'
        )
        print(f"模型2训练完成. 最终平均奖励: {r2[-1]:.4f}")
    
    print("\n" + "="*60)
    print("✅ 所有模型训练完成!")
    print("="*60)
