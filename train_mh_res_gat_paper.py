"""
训练 MH-Res-GAT 模型 - 符合论文描述的架构
使用缩放点积 + LeakyReLU 注意力（无物理先验）
"""
import os
os.environ['USERNAME'] = 'user'
os.environ['USER'] = 'user'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '.'

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from models import MH_ResGAT_Model, HAPPO_GNN_RL_Model, build_graph_from_data, compute_physics_prior
import time

# ============================================================
# 配置
# ============================================================
DATA_PATH = 'data.csv'
MODEL_SAVE_PATH = 'model_mh_res_gat.pth'
DEVICE = torch.device('cpu')

# 训练参数
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

# 模型参数
FEAT_DIM = 10
N_STATIONS = 20
N_ACTIONS = 3
GAT_HIDDEN = 64
GAT_LAYERS = 2
GAT_HEADS = 4


def compute_reward(station_states, action_idx, adj_matrix):
    """
    计算基于真实状态的奖励函数
    奖励 = 绿电奖励 - 等待惩罚 - 负荷惩罚
    """
    wait, power, queue, green = station_states[:, 0], station_states[:, 1], station_states[:, 2], station_states[:, 3]
    
    # 动作效果模拟
    if action_idx == 0:  # 降功率
        power_adj = 0.8
        wait_adj = 1.1
    elif action_idx == 1:  # 维持
        power_adj = 1.0
        wait_adj = 1.0
    else:  # 升功率
        power_adj = 1.2
        wait_adj = 0.85
    
    new_wait = wait * wait_adj
    new_power = power * power_adj
    
    # 奖励组成
    green_reward = green.mean() * 10.0  # 绿电利用
    wait_penalty = (new_wait.mean() / 20.0) * 5.0  # 等待惩罚
    power_penalty = (new_power.mean() / 100.0) * 3.0  # 功率惩罚
    queue_penalty = (queue.mean() / 10.0) * 2.0  # 排队惩罚
    
    reward = green_reward - wait_penalty - power_penalty - queue_penalty
    return reward


def train_with_policy_gradient(model, node_features, adj, optimizer, epoch):
    """策略梯度训练"""
    model.train()
    n = node_features.shape[0]
    
    # 随机生成站点状态
    station_states = torch.randn(n, 4) * 0.5 + 0.5
    station_states[:, 3] = torch.rand(n) * 0.5 + 0.2  # 绿电比例 0.2-0.7
    
    action_masks = torch.ones(n, 3)
    action_masks[station_states[:, 2] < 2, 0] = 0  # 排队少不降功率
    action_masks[station_states[:, 2] > 8, 2] = 0  # 排队多不升功率
    
    # 前向传播
    action_probs, attn, values = model(
        node_features, adj, S=None, gamma=0.0, 
        station_states=station_states, action_masks=action_masks
    )
    
    # 采样动作
    dist = torch.distributions.Categorical(action_probs)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    
    # 计算奖励
    rewards = []
    for i in range(n):
        r = compute_reward(station_states, actions[i].item(), adj)
        rewards.append(r)
    rewards = torch.tensor(rewards, dtype=torch.float32).mean()
    
    # 策略梯度损失
    advantage = rewards - values.mean().detach()
    policy_loss = -(log_probs * advantage.detach()).mean()
    
    # 熵正则化（鼓励探索）
    entropy = dist.entropy().mean()
    
    # 价值损失
    value_loss = F.mse_loss(values.mean(), rewards.detach().unsqueeze(0))
    
    # 注意力正则化（鼓励邻接节点间的注意力）
    attn_loss = 0.0
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] > 0.01:
                attn_loss -= torch.log(attn[i, j] + 1e-8) * adj[i, j]
    attn_loss = attn_loss / (n * n)
    
    # 总损失
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy + 0.1 * attn_loss
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item(), rewards.item(), entropy.item()


def main():
    print("=" * 70)
    print("训练 MH-Res-GAT 模型（论文架构：缩放点积 + LeakyReLU）")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    df = pd.read_csv(DATA_PATH)
    station_ids = sorted(df['station_id'].unique())[:N_STATIONS]
    print(f"      使用 {len(station_ids)} 个站点，共 {len(df)} 条记录")
    
    # 2. 构建图数据
    print("\n[2/5] 构建图数据...")
    node_features, adj, _, _ = build_graph_from_data(df, station_ids)
    print(f"      节点特征形状: {node_features.shape}")
    print(f"      邻接矩阵形状: {adj.shape}")
    
    x_tensor = torch.tensor(node_features, dtype=torch.float32).to(DEVICE)
    adj_tensor = torch.tensor(adj, dtype=torch.float32).to(DEVICE)
    
    # 3. 初始化模型
    print("\n[3/5] 初始化模型...")
    model = MH_ResGAT_Model(
        feat_dim=FEAT_DIM,
        n_stations=N_STATIONS,
        n_actions=N_ACTIONS,
        gat_hidden=GAT_HIDDEN,
        gat_layers=GAT_LAYERS,
        gat_heads=GAT_HEADS,
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      参数量: {n_params:,}")
    
    # 4. 训练
    print("\n[4/5] 开始训练（策略梯度）...")
    print("-" * 70)
    
    start_time = time.time()
    best_reward = float('-inf')
    
    for epoch in range(NUM_EPOCHS):
        loss, reward, entropy = train_with_policy_gradient(
            model, x_tensor, adj_tensor, optimizer, epoch
        )
        
        if reward > best_reward:
            best_reward = reward
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"[Epoch {epoch+1:3d}/{NUM_EPOCHS}] "
                  f"Loss={loss:.4f} | Reward={reward:.4f} | "
                  f"Best={best_reward:.4f} | Entropy={entropy:.4f} | "
                  f"Time={elapsed:.1f}s")
    
    print("-" * 70)
    
    # 5. 保存模型
    print("\n[5/5] 保存模型...")
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_reward': best_reward,
        'model_type': 'mh_res_gat',
        'config': {
            'feat_dim': FEAT_DIM,
            'n_stations': N_STATIONS,
            'n_actions': N_ACTIONS,
            'gat_hidden': GAT_HIDDEN,
            'gat_layers': GAT_LAYERS,
            'gat_heads': GAT_HEADS,
        }
    }, MODEL_SAVE_PATH)
    
    total_time = time.time() - start_time
    print(f"      模型已保存到: {MODEL_SAVE_PATH}")
    print(f"      最佳奖励: {best_reward:.4f}")
    print(f"      总训练时间: {total_time:.1f}秒")
    print("=" * 70)
    
    return model


if __name__ == '__main__':
    trained_model = main()
