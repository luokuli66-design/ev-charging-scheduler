"""
快速训练 MH-Res-GAT 模型 - 简化版
使用正确的 ScaledDotProduct 注意力架构
"""
# 修复 Windows PyTorch 问题
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
import time

# ============================================================
# 配置
# ============================================================
DATA_PATH = 'data.csv'
MODEL_SAVE_PATH = 'model_mh_res_gat_new.pth'
DEVICE = torch.device('cpu')

# 训练参数（简化）
NUM_EPOCHS = 30  # 减少epoch数，加快训练
BATCH_SIZE = 64
LEARNING_RATE = 1e-3  # 提高学习率

# 模型参数
FEAT_DIM = 10
N_STATIONS = 20
N_ACTIONS = 3
GAT_HIDDEN = 64
GAT_LAYERS = 2
GAT_HEADS = 4


def main():
    print("=" * 70)
    print("训练 MH-Res-GAT 模型（ScaledDotProduct 注意力）")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    df = pd.read_csv(DATA_PATH)
    station_ids = sorted(df['station_id'].unique())[:N_STATIONS]
    print(f"      使用 {len(station_ids)} 个站点")
    
    # 2. 构建图数据
    print("\n[2/5] 构建图数据...")
    node_features, adj, _, scaler = build_graph_from_data(df, station_ids)
    print(f"      节点特征形状: {node_features.shape}")
    print(f"      邻接矩阵形状: {adj.shape}")
    
    # 转换为 tensor
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
    
    # 4. 训练（简化的目标函数）
    print("\n[4/5] 开始训练...")
    print("-" * 70)
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        
        # 简化的训练：使用随机动作 + 策略梯度
        model.train()
        
        # 前向传播
        station_states = torch.randn(N_STATIONS, 4).to(DEVICE)  # 简化
        
        action_probs, attn, values = model(
            x_tensor, adj_tensor, S=None, gamma=0.0, station_states=station_states
        )
        
        # 简化的损失函数：
        # 1. 鼓励探索（熵最大化）
        dist = torch.distributions.Categorical(action_probs)
        entropy = dist.entropy().mean()
        
        # 2. 伪奖励（基于注意力权重的一致性）
        attn_target = torch.eye(N_STATIONS).to(DEVICE) * 0.5 + attn.detach() * 0.5
        attn_loss = F.mse_loss(attn, attn_target)
        
        # 3. 价值函数的伪目标（归一化）
        value_target = torch.randn(N_STATIONS, 1).to(DEVICE)
        value_loss = F.mse_loss(values, value_target)
        
        # 总损失
        loss = attn_loss + 0.5 * value_loss - 0.01 * entropy
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_losses.append(loss.item())
        
        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = np.mean(epoch_losses)
            elapsed = time.time() - start_time
            print(f"[Epoch {epoch+1:3d}/{NUM_EPOCHS}] "
                  f"Loss={avg_loss:.4f} | "
                  f"Entropy={entropy.item():.4f} | "
                  f"Time={elapsed:.1f}s")
    
    print("-" * 70)
    
    # 5. 保存模型
    print("\n[5/5] 保存模型...")
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': np.mean(epoch_losses) if 'epoch_losses' in locals() else 0.0,
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
    print(f"      总训练时间: {total_time:.1f}秒")
    print("=" * 70)
    
    return model


if __name__ == '__main__':
    trained_model = main()
