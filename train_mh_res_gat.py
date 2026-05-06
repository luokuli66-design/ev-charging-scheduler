"""
重新训练 MH-Res-GAT 模型（使用新的 ScaledDotProduct 注意力架构）
基于 data.csv 真实数据
"""
# 必须在最开头设置环境变量，修复 Windows 上 PyTorch 的问题
import os
os.environ['USERNAME'] = 'user'
os.environ['USER'] = 'user'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '.'

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from models import MH_ResGAT_Model, build_graph_from_data

# ============================================================
# 配置
# ============================================================
DATA_PATH = 'data.csv'
MODEL_SAVE_PATH = 'model_mh_res_gat_new.pth'
DEVICE = torch.device('cpu')

# 训练超参数
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4

# 模型参数
FEAT_DIM = 10
N_STATIONS = 20
N_ACTIONS = 3
GAT_HIDDEN = 64
GAT_LAYERS = 2
GAT_HEADS = 4
POLICY_HIDDEN = 128


# ============================================================
# 数据预处理
# ============================================================
def prepare_training_data(df, station_ids):
    """准备训练数据 - 从 data.csv 提取状态转移"""
    print(f"[数据] 处理 {len(df)} 条记录, {len(station_ids)} 个站点")
    
    episodes = []
    for station_id in station_ids:
        sub = df[df['station_id'] == station_id].reset_index(drop=True)
        if len(sub) < 20:
            continue
        
        # 使用滑动窗口创建训练样本
        n_samples = min(len(sub) - 1, 100)  # 每个站点最多100个样本
        for i in range(n_samples):
            s_row = sub.iloc[i]
            s_next_row = sub.iloc[i + 1]
            
            # 状态特征（4维）
            state = np.array([
                float(s_row['waiting_time']),
                float(s_row['charging_power_kW']),
                float(s_row['queue_length']),
                float(s_row['renewable_energy_ratio']),
            ], dtype=np.float32)
            
            next_state = np.array([
                float(s_next_row['waiting_time']),
                float(s_next_row['charging_power_kW']),
                float(s_next_row['queue_length']),
                float(s_next_row['renewable_energy_ratio']),
            ], dtype=np.float32)
            
            # 计算奖励
            reward = compute_reward(state, next_state)
            
            episodes.append({
                'station_id': station_id,
                'state': state,
                'next_state': next_state,
                'reward': reward,
            })
    
    print(f"[数据] 生成 {len(episodes)} 个训练样本")
    return episodes


def compute_reward(state, next_state):
    """计算奖励函数"""
    w1, w2, w3, w4 = 1.0, 0.1, 2.0, 5.0
    
    T_new = next_state[0]  # waiting_time
    P_new = next_state[1]  # charging_power_kW
    Q_t = state[2]  # queue_length
    G_t = next_state[3]  # renewable_energy_ratio
    
    wait_penalty = max(0, Q_t - 5) * 0.5
    reward = -w1 * T_new + w2 * P_new - w3 * wait_penalty + w4 * G_t
    
    return float(reward)


# ============================================================
# 简化的 PPO 训练
# ============================================================
def train():
    print("=" * 70)
    print("开始训练 MH-Res-GAT 模型（ScaledDotProduct 注意力架构）")
    print("=" * 70)
    
    # 1. 加载数据
    df = pd.read_csv(DATA_PATH)
    station_ids = sorted(df['station_id'].unique())[:N_STATIONS]
    print(f"\n[配置] 站点数: {len(station_ids)}, 动作数: {N_ACTIONS}")
    print(f"[配置] 特征维度: {FEAT_DIM}, GAT隐藏层: {GAT_HIDDEN}")
    
    # 2. 准备训练数据
    episodes = prepare_training_data(df, station_ids)
    
    if len(episodes) == 0:
        print("[错误] 没有生成训练样本！")
        return None
    
    # 3. 构建图数据（用于GAT）
    node_features, adj, _, scaler = build_graph_from_data(df, station_ids)
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32).to(DEVICE)
    adj_tensor = torch.tensor(adj, dtype=torch.float32).to(DEVICE)
    
    # 4. 初始化模型
    model = MH_ResGAT_Model(
        feat_dim=FEAT_DIM,
        n_stations=N_STATIONS,
        n_actions=N_ACTIONS,
        gat_hidden=GAT_HIDDEN,
        gat_layers=GAT_LAYERS,
        gat_heads=GAT_HEADS,
        policy_hidden=POLICY_HIDDEN,
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 计算参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 参数量: {n_params:,}")
    print(f"[训练] Epochs={NUM_EPOCHS}, Batch={BATCH_SIZE}, LR={LEARNING_RATE}")
    print("-" * 70)
    
    # 5. 训练循环（简化的 PPO）
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        
        # 将样本分成 batch
        n_samples = len(episodes)
        indices = np.random.permutation(n_samples)
        
        for start in range(0, n_samples, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_samples)
            batch_indices = indices[start:end]
            batch = [episodes[i] for i in batch_indices]
            
            # 准备 batch 数据
            states = []
            actions = []
            log_probs_old = []
            rewards = []
            
            for ep in batch:
                state = ep['state']
                next_state = ep['next_state']
                reward = ep['reward']
                
                # 拼接：节点特征 (GAT输出) + 站点状态 (4维)
                # 简化：使用 station_states 作为策略网络输入的一部分
                station_state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
                
                # 前向传播获取动作概率
                with torch.no_grad():
                    # GAT 编码（所有站点共享）
                    h_gat, attn = model.gat(node_features_tensor, adj_tensor, S=None, gamma=0.0)
                    
                    # 为当前站点构造策略输入
                    # 找到当前站点在 node_features 中的索引
                    station_idx = station_ids.index(ep['station_id'])
                    policy_input = torch.cat([
                        h_gat[station_idx].unsqueeze(0),
                        station_state.unsqueeze(0)
                    ], dim=-1)
                    
                    # 策略网络
                    action_probs, logits = model.policy(policy_input)
                    
                    # 采样动作
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                
                states.append(station_state.numpy())
                actions.append(action.item())
                log_probs_old.append(log_prob.item())
                rewards.append(reward)
            
            # 转换为 tensor
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
            actions_tensor = torch.tensor(actions, dtype=torch.long).to(DEVICE)
            log_probs_old_tensor = torch.tensor(log_probs_old, dtype=torch.float32).to(DEVICE)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
            
            # 归一化奖励
            rewards_norm = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
            
            # PPO 更新（K_EPOCHS 次）
            for _ in range(K_EPOCHS):
                # 重新前向传播
                h_gat, attn = model.gat(node_features_tensor, adj_tensor, S=None, gamma=0.0)
                
                # 构造策略输入（简化：使用 batch 中第一个站点的索引）
                policy_inputs = []
                for i, ep in enumerate(batch):
                    station_idx = station_ids.index(ep['station_id'])
                    policy_input = torch.cat([
                        h_gat[station_idx].unsqueeze(0),
                        states_tensor[i].unsqueeze(0)
                    ], dim=-1)
                    policy_inputs.append(policy_input)
                
                policy_input_cat = torch.cat(policy_inputs, dim=0)
                action_probs_new, logits_new = model.policy(policy_input_cat)
                
                # 计算新 log probabilities
                dist_new = torch.distributions.Categorical(action_probs_new)
                log_probs_new_tensor = dist_new.log_prob(actions_tensor)
                
                # 计算 ratio
                ratio = torch.exp(log_probs_new_tensor - log_probs_old_tensor)
                
                # PPO 损失
                surr1 = ratio * rewards_norm
                surr2 = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * rewards_norm
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 熵奖励
                entropy = dist_new.entropy().mean()
                
                # 总损失
                loss = policy_loss - 0.01 * entropy
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                epoch_losses.append(loss.item())
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = np.mean(epoch_losses)
            avg_reward = np.mean([ep['reward'] for ep in episodes])
            print(f"[Epoch {epoch+1:3d}/{NUM_EPOCHS}] Loss={avg_loss:.4f}, Avg_Reward={avg_reward:.4f}")
    
    # 6. 保存模型
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss if 'avg_loss' in locals() else 0.0,
        'model_type': 'mh_res_gat',
        'config': {
            'feat_dim': FEAT_DIM,
            'n_stations': N_STATIONS,
            'n_actions': N_ACTIONS,
            'gat_hidden': GAT_HIDDEN,
            'gat_layers': GAT_LAYERS,
            'gat_heads': GAT_HEADS,
            'policy_hidden': POLICY_HIDDEN,
        }
    }, MODEL_SAVE_PATH)
    
    print("-" * 70)
    print(f"[完成] 模型已保存到: {MODEL_SAVE_PATH}")
    print("=" * 70)
    
    return model


if __name__ == '__main__':
    trained_model = train()
