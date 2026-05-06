"""
验证两个独立模型是否产生不同的 action 输出
测试重点：确保 MH-Res-GAT 和 HAPPO-GNN-RL 的注意力机制差异导致 action 差异
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import pandas as pd
from models import (
    MH_ResGAT_Model, HAPPO_GNN_RL_Model,
    load_trained_model, run_model_inference,
    build_graph_from_data, compute_physics_prior,
    ScaledDotProductGATEncoder, LeakyReLUConcatGATEncoder
)

def verify_models_loaded():
    """验证两个模型类能正确实例化"""
    print("\n" + "="*60)
    print("【测试1】验证两个模型类实例化")
    print("="*60)

    n_stations = 20
    feat_dim = 10
    n_actions = 3

    try:
        model1 = MH_ResGAT_Model(feat_dim=feat_dim, n_stations=n_stations, n_actions=n_actions)
        model2 = HAPPO_GNN_RL_Model(feat_dim=feat_dim, n_stations=n_stations, n_actions=n_actions)

        print(f"✓ MH_ResGAT_Model 实例化成功")
        print(f"  GAT类型: {type(model1.gat).__name__}")
        print(f"  GAT层数: {len(model1.gat.layers)}")
        print(f"  注意力头数: {model1.gat.layers[0].n_heads}")

        print(f"\n✓ HAPPO_GNN_RL_Model 实例化成功")
        print(f"  GAT类型: {type(model2.gat).__name__}")
        print(f"  GAT层数: {len(model2.gat.layers)}")
        print(f"  注意力头数: {model2.gat.layers[0].n_heads}")

        return True
    except Exception as e:
        print(f"✗ 模型实例化失败: {e}")
        return False


def verify_attention_difference():
    """验证两种注意力机制的数学公式差异"""
    print("\n" + "="*60)
    print("【测试2】验证注意力机制数学公式差异")
    print("="*60)

    # 创建一个简单的测试输入
    torch.manual_seed(42)
    n = 5  # 5个节点
    d = 8  # 特征维度
    hidden = 16  # 隐藏层维度

    # 创建测试数据
    h = torch.randn(n, d)  # 节点特征
    adj = torch.rand(n, n)  # 邻接矩阵
    adj = (adj + adj.T) / 2  # 对称化
    adj = adj * (adj > 0.3).float()  # 稀疏化
    adj = adj + torch.eye(n)  # 加入自环

    # 测试缩放点积注意力
    print("\n--- 缩放点积注意力 (MH-Res-GAT) ---")
    attn1 = ScaledDotProductGATEncoder(d, hidden, n_layers=1, n_heads=2)
    h_out1, alpha1 = attn1(h, adj, S=None, gamma=0.0)
    print(f"输出形状: {h_out1.shape}")
    print(f"注意力矩阵形状: {alpha1.shape}")
    print(f"注意力矩阵 (第一行前5列):")
    print(alpha1[0].detach().numpy()[:5].round(4))

    # 测试 LeakyReLU+拼接注意力
    print("\n--- LeakyReLU+拼接注意力 (HAPPO-GNN-RL) ---")
    attn2 = LeakyReLUConcatGATEncoder(d, hidden, n_layers=1, n_heads=2)
    # 创建简单的物理先验矩阵
    S = torch.rand(n, n) * 0.5
    S = S + S.T + torch.eye(n)
    h_out2, alpha2 = attn2(h, adj, S=S, gamma=0.3)
    print(f"输出形状: {h_out2.shape}")
    print(f"注意力矩阵形状: {alpha2.shape}")
    print(f"注意力矩阵 (第一行前5列):")
    print(alpha2[0].detach().numpy()[:5].round(4))

    # 计算注意力差异
    diff = torch.abs(alpha1 - alpha2).mean().item()
    print(f"\n注意力矩阵差异 (L1距离): {diff:.4f}")

    if diff > 0.01:
        print("✓ 两种注意力机制产生不同的注意力权重分布")
    else:
        print("✗ 注意力机制差异不够明显")

    return diff > 0.01


def verify_forward_output_difference():
    """验证两个模型的 forward 输出确实不同"""
    print("\n" + "="*60)
    print("【测试3】验证模型 forward 输出差异")
    print("="*60)

    torch.manual_seed(123)
    n_stations = 20
    feat_dim = 10
    n_actions = 3

    # 实例化两个模型（随机初始化权重）
    model1 = MH_ResGAT_Model(feat_dim=feat_dim, n_stations=n_stations, n_actions=n_actions)
    model2 = HAPPO_GNN_RL_Model(feat_dim=feat_dim, n_stations=n_stations, n_actions=n_actions)

    # 创建测试输入
    x = torch.randn(n_stations, feat_dim)
    adj = torch.rand(n_stations, n_stations)
    adj = (adj + adj.T) / 2 + torch.eye(n_stations)
    S = torch.rand(n_stations, n_stations) * 0.5
    S = S + S.T + torch.eye(n_stations)
    station_states = torch.randn(n_stations, 4)
    action_masks = torch.ones(n_stations, n_actions)

    # Forward pass
    with torch.no_grad():
        probs1, attn1, values1 = model1(x, adj, S, gamma=0.0, station_states=station_states, action_masks=action_masks)
        probs2, attn2, values2 = model2(x, adj, S, gamma=0.3, station_states=station_states, action_masks=action_masks)

    print(f"\n模型1 (MH-Res-GAT) 输出:")
    print(f"  action_probs 形状: {probs1.shape}")
    print(f"  注意力矩阵形状: {attn1.shape}")
    print(f"  第一站 action_probs: {probs1[0].numpy().round(4)}")
    print(f"  argmax action: {probs1.argmax(dim=1)[0].item()}")

    print(f"\n模型2 (HAPPO-GNN-RL) 输出:")
    print(f"  action_probs 形状: {probs2.shape}")
    print(f"  注意力矩阵形状: {attn2.shape}")
    print(f"  第一站 action_probs: {probs2[0].numpy().round(4)}")
    print(f"  argmax action: {probs2.argmax(dim=1)[0].item()}")

    # 计算差异
    probs_diff = torch.abs(probs1 - probs2).mean().item()
    attn_diff = torch.abs(attn1 - attn2).mean().item()
    values_diff = torch.abs(values1 - values2).mean().item()

    print(f"\n输出差异统计:")
    print(f"  action_probs 差异 (L1): {probs_diff:.4f}")
    print(f"  注意力矩阵差异 (L1): {attn_diff:.4f}")
    print(f"  value 差异 (L1): {values_diff:.4f}")

    # 统计 action 差异
    actions1 = probs1.argmax(dim=1).numpy()
    actions2 = probs2.argmax(dim=1).numpy()
    diff_count = (actions1 != actions2).sum()
    diff_pct = diff_count / n_stations * 100

    print(f"\n动作选择差异:")
    print(f"  不同的站点数: {diff_count}/{n_stations} ({diff_pct:.1f}%)")
    for i in range(n_stations):
        if actions1[i] != actions2[i]:
            action_names = ['降低功率(-20%)', '维持当前功率', '提升功率(+20%)']
            print(f"    站点{i}: MH={action_names[actions1[i]]}, HAPPO={action_names[actions2[i]]}")

    return probs_diff > 0.001, diff_count


def verify_architecture_independence():
    """
    验证两个架构在独立随机初始化下产生不同的输出
    使用不同的随机种子，确保权重不同
    """
    print("\n" + "="*60)
    print("【测试4b】独立随机初始化架构验证")
    print("="*60)

    torch.manual_seed(9999)
    n_stations = 20
    feat_dim = 10
    n_actions = 3

    # 模型1：MH-Res-GAT (种子9999)
    model1 = MH_ResGAT_Model(feat_dim=feat_dim, n_stations=n_stations, n_actions=n_actions)
    # 模型2：HAPPO-GNN-RL (种子42)
    torch.manual_seed(42)
    model2 = HAPPO_GNN_RL_Model(feat_dim=feat_dim, n_stations=n_stations, n_actions=n_actions)

    # 验证权重确实不同（只比较浮点型tensor）
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()
    common_keys = [k for k in sd1 if k in sd2
                  and sd1[k].shape == sd2[k].shape
                  and sd1[k].dtype in (torch.float32, torch.float16, torch.float64)]
    weight_diff = sum((sd1[k] - sd2[k]).abs().mean().item() for k in common_keys) / len(common_keys)
    print(f"权重差异 (L1): {weight_diff:.4f}")

    # 创建相同的测试输入
    torch.manual_seed(12345)
    x = torch.randn(n_stations, feat_dim)
    adj = torch.rand(n_stations, n_stations)
    adj = (adj + adj.T) / 2 + torch.eye(n_stations)
    S = torch.rand(n_stations, n_stations) * 0.5
    S = S + S.T + torch.eye(n_stations)
    station_states = torch.randn(n_stations, 4)
    action_masks = torch.ones(n_stations, n_actions)

    # Forward pass
    with torch.no_grad():
        probs1, attn1, values1 = model1(x, adj, S, gamma=0.0, station_states=station_states, action_masks=action_masks)
        probs2, attn2, values2 = model2(x, adj, S, gamma=0.3, station_states=station_states, action_masks=action_masks)

    actions1 = probs1.argmax(dim=1).numpy()
    actions2 = probs2.argmax(dim=1).numpy()
    diff_count = (actions1 != actions2).sum()
    diff_pct = diff_count / n_stations * 100

    print(f"\n动作选择差异: {diff_count}/{n_stations} ({diff_pct:.1f}%)")
    probs_diff = torch.abs(probs1 - probs2).mean().item()
    attn_diff = torch.abs(attn1 - attn2).mean().item()
    print(f"action_probs 差异: {probs_diff:.4f}")
    print(f"注意力矩阵差异: {attn_diff:.4f}")

    action_names = ['降低功率(-20%)', '维持当前功率', '提升功率(+20%)']
    for i in range(n_stations):
        if actions1[i] != actions2[i]:
            print(f"  站点{i}: MH={action_names[actions1[i]]}, HAPPO={action_names[actions2[i]]}")

    return diff_count >= 5, diff_count, probs_diff, attn_diff


def verify_checkpoint_state():
    """
    诊断旧 checkpoint 的权重结构
    """
    print("\n" + "="*60)
    print("【测试5】旧 Checkpoint 权重结构诊断")
    print("="*60)

    import torch

    mh_path = os.path.join(os.path.dirname(__file__), 'model_mh_res_gat.pth')
    happo_path = os.path.join(os.path.dirname(__file__), 'model_fusion_gamma030.pth')

    # 加载两个 checkpoint（不加载模型）
    cp1 = torch.load(mh_path, map_location='cpu', weights_only=False)
    cp2 = torch.load(happo_path, map_location='cpu', weights_only=False)

    sd1 = cp1.get('model_state_dict', cp1)
    sd2 = cp2.get('model_state_dict', cp2)

    print(f"\nMH checkpoint keys: {len(sd1)}")
    print(f"HAPPO checkpoint keys: {len(sd2)}")

    # 检查 GAT 层权重结构
    gat_keys_1 = [k for k in sd1 if 'gat' in k and 'heads' in k]
    gat_keys_2 = [k for k in sd2 if 'gat' in k and 'heads' in k]

    print(f"\nMH GAT注意力相关keys (前10): {gat_keys_1[:10]}")
    print(f"HAPPO GAT注意力相关keys (前10): {gat_keys_2[:10]}")

    # 检查两个checkpoint的权重是否相同
    if set(sd1.keys()) == set(sd2.keys()):
        print("\n两个checkpoint的key完全相同!")
        # 只比较浮点型tensor
        common_keys = [k for k in sd1 if k in sd2
                      and sd1[k].shape == sd2[k].shape
                      and sd1[k].dtype in (torch.float32, torch.float16, torch.float64)]
        if common_keys:
            diff_sum = sum((sd1[k] - sd2[k]).abs().mean().item() for k in common_keys) / len(common_keys)
            print(f"共有关键字（浮点tensor）的权重差异: {diff_sum:.6f}")
            if diff_sum < 1e-5:
                print("  → 结论：两个checkpoint权重几乎完全相同！")
                print("  → 这意味着它们是用同一个架构训练的，只是保存成了两个文件")
            else:
                print(f"  → 两个checkpoint的权重有差异（可能是训练过程中的不同保存点）")
                print(f"  → 差异来源：{diff_sum:.6f} > 0 表明训练进度不同")
    else:
        print("\n两个checkpoint的key结构不同")
        only_in_1 = set(sd1.keys()) - set(sd2.keys())
        only_in_2 = set(sd2.keys()) - set(sd1.keys())
        print(f"  仅在MH中: {list(only_in_1)[:5]}")
        print(f"  仅在HAPPO中: {list(only_in_2)[:5]}")


def main():
    print("\n" + "="*60)
    print("  HAPPO-GNN-RL 双模型架构验证")
    print("  目标: 确保两个模型使用不同的注意力机制")
    print("="*60)

    results = {}

    # 测试1: 模型实例化
    results['instantiation'] = verify_models_loaded()

    # 测试2: 注意力机制差异
    results['attention_diff'] = verify_attention_difference()

    # 测试3: Forward输出差异（相同随机种子）
    probs_diff, diff_count = verify_forward_output_difference()
    results['forward_diff'] = probs_diff
    results['diff_count'] = diff_count

    # 测试4b: 独立随机初始化架构验证
    ind_diff_ok, ind_diff_count, ind_probs_diff, ind_attn_diff = verify_architecture_independence()
    results['independent_diff'] = ind_diff_ok
    results['independent_diff_count'] = ind_diff_count

    # 测试5: checkpoint 诊断
    verify_checkpoint_state()

    # 总结
    print("\n" + "="*60)
    print("  验证总结")
    print("="*60)
    print(f"✓ 模型实例化: {'通过' if results['instantiation'] else '失败'}")
    print(f"✓ 注意力机制数学公式差异: {'通过' if results['attention_diff'] else '失败'}")
    print(f"✓ 随机初始化下输出差异: {'通过' if results['forward_diff'] else '失败'}")
    print(f"  站点动作差异数: {results.get('diff_count', 0)}/20")
    print(f"✓ 独立初始化架构差异: {'通过' if results['independent_diff'] else '失败'}")
    print(f"  独立初始化站点差异数: {ind_diff_count}/20")
    print(f"  注意力矩阵差异: {ind_attn_diff:.4f}")

    if results['attention_diff'] and results['forward_diff']:
        print("\n【结论】两个模型的注意力机制已成功区分!")
        print("  - MH-Res-GAT: 缩放点积注意力 (无物理先验)")
        print("    公式: score(h_i, h_j) = (W_q h_i)^T (W_k h_j) / sqrt(d)")
        print("  - HAPPO-GNN-RL: LeakyReLU+拼接注意力 (融合物理先验)")
        print("    公式: omega_ij = LeakyReLU(W_a · [h_i ⊕ h_j])")
        print("    融合: e_ij = (1-gamma)*omega_ij + gamma*S_ij")
        print("\n  ⚠️ 注意: 旧checkpoint文件使用的是同一架构训练的结果，")
        print("     加载到新架构时大量权重被随机初始化。")
        print("     如需使用预训练模型，请用新架构重新训练。")
    else:
        print("\n【警告】模型差异不够明显，需要进一步检查!")

    return results


if __name__ == '__main__':
    main()
