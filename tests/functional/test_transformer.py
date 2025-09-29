"""
简化的Transformer测试脚本
专注于生成适合屏幕显示的可视化
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.transformer import create_model


def calculate_model_flops(model, batch_size, src_len, tgt_len):
    """计算模型的FLOPs"""
    d_model = model.d_model
    n_heads = model.n_heads
    d_k = d_model // n_heads
    d_ff = model.encoder[0].feed_forward.W_1.out_features
    
    # 词嵌入层
    embedding_flops = (src_len + tgt_len) * batch_size * d_model * 2
    
    # 编码器层
    encoder_layers = len(model.encoder)
    attention_flops_per_layer = (
        src_len * batch_size * d_model * d_model * 3 +  # QKV投影
        batch_size * n_heads * src_len * src_len * d_k +  # 注意力计算
        src_len * batch_size * d_model * d_model  # 输出投影
    )
    
    ffn_flops_per_layer = (
        src_len * batch_size * d_model * d_ff +  # 第一个线性层
        src_len * batch_size * d_ff * d_model    # 第二个线性层
    )
    
    encoder_flops = encoder_layers * (attention_flops_per_layer + ffn_flops_per_layer)
    
    # 解码器层
    decoder_layers = len(model.decoder)
    masked_attention_flops_per_layer = (
        tgt_len * batch_size * d_model * d_model * 3 +  # QKV投影
        batch_size * n_heads * tgt_len * tgt_len * d_k +  # 注意力计算
        tgt_len * batch_size * d_model * d_model  # 输出投影
    )
    
    cross_attention_flops_per_layer = (
        tgt_len * batch_size * d_model * d_model +  # Q投影
        src_len * batch_size * d_model * d_model * 2 +  # KV投影
        batch_size * n_heads * tgt_len * src_len * d_k +  # 注意力计算
        tgt_len * batch_size * d_model * d_model  # 输出投影
    )
    
    ffn_flops_per_layer = (
        tgt_len * batch_size * d_model * d_ff +  # 第一个线性层
        tgt_len * batch_size * d_ff * d_model    # 第二个线性层
    )
    
    decoder_flops = decoder_layers * (masked_attention_flops_per_layer + 
                                     cross_attention_flops_per_layer + 
                                     ffn_flops_per_layer)
    
    # 输出投影
    tgt_vocab_size = model.tgt_embedding.num_embeddings
    output_projection_flops = tgt_len * batch_size * d_model * tgt_vocab_size
    
    total_flops = embedding_flops + encoder_flops + decoder_flops + output_projection_flops
    
    return total_flops


def estimate_memory_usage(model, batch_size, src_len, tgt_len):
    """估算模型内存使用"""
    d_model = model.d_model
    n_heads = model.n_heads
    
    # 模型参数内存
    param_memory = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024  # FP32
    
    # 激活值内存 (简化估算)
    embedding_memory = (src_len + tgt_len) * batch_size * d_model * 4 / 1024 / 1024
    
    attention_weights_memory = (
        batch_size * n_heads * src_len * src_len +  # 编码器自注意力
        batch_size * n_heads * tgt_len * tgt_len +  # 解码器掩码自注意力
        batch_size * n_heads * tgt_len * src_len    # 交叉注意力
    ) * 4 / 1024 / 1024
    
    intermediate_memory = (src_len + tgt_len) * batch_size * d_model * 4 / 1024 / 1024
    
    total_memory = param_memory + embedding_memory + attention_weights_memory + intermediate_memory
    
    return total_memory


def visualize_model_structure(model, save_path=None):
    """可视化模型结构 - 适合屏幕显示"""
    print("🎨 Visualizing model structure...")
    
    # 创建图形 - 适合屏幕显示
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Transformer Model Structure Analysis', fontsize=14)
    
    # 1. 参数分布
    ax1 = axes[0, 0]
    param_counts = []
    layer_names = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_counts.append(param.numel())
            layer_names.append(name.split('.')[0])
    
    # 统计每层的参数数量
    layer_param_counts = {}
    for name, count in zip(layer_names, param_counts):
        if name not in layer_param_counts:
            layer_param_counts[name] = 0
        layer_param_counts[name] += count
    
    layers = list(layer_param_counts.keys())
    counts = [count / 1000 for count in layer_param_counts.values()]  # 转换为K
    
    bars = ax1.bar(layers, counts)
    ax1.set_title('Parameters per Layer Type')
    ax1.set_xlabel('Layer Type')
    ax1.set_ylabel('Parameter Count (K)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count:.1f}K', ha='center', va='bottom', fontsize=8)
    
    # 2. 模型深度分析
    ax2 = axes[0, 1]
    encoder_layers = len(model.encoder)
    decoder_layers = len(model.decoder)
    
    layer_types = ['Encoder\nLayers', 'Decoder\nLayers', 'Embedding\nLayers', 'Output\nProjection']
    layer_counts = [encoder_layers, decoder_layers, 2, 1]  # src_embedding, tgt_embedding
    
    bars = ax2.bar(layer_types, layer_counts, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax2.set_title('Model Architecture Overview')
    ax2.set_ylabel('Number of Layers')
    
    # 添加数值标签
    for bar, count in zip(bars, layer_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 3. 参数大小分布
    ax3 = axes[1, 0]
    param_sizes = [p.numel() / 1000 for p in model.parameters() if p.requires_grad]  # 转换为K
    ax3.hist(param_sizes, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax3.set_title('Parameter Size Distribution')
    ax3.set_xlabel('Parameter Count (K, log scale)')
    ax3.set_ylabel('Frequency')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. 模型信息总结
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算FLOPs (假设输入尺寸)
    batch_size, src_len, tgt_len = 1, 10, 8  # 示例输入尺寸
    flops = calculate_model_flops(model, batch_size, src_len, tgt_len)
    
    model_info = f"""
    Model Configuration:
    • d_model: {model.d_model}
    • n_heads: {model.n_heads}
    • Encoder layers: {len(model.encoder)}
    • Decoder layers: {len(model.decoder)}
    
    Parameter Statistics:
    • Total parameters: {total_params/1000:.1f}K
    • Trainable parameters: {trainable_params/1000:.1f}K
    • Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)
    
    Computational Cost (batch=1, src=10, tgt=8):
    • FLOPs: {flops/1e6:.2f}M
    • Memory: {estimate_memory_usage(model, batch_size, src_len, tgt_len):.2f} MB
    
    Architecture:
    • Source vocab size: {model.src_embedding.num_embeddings}
    • Target vocab size: {model.tgt_embedding.num_embeddings}
    • Positional encoding max_len: {model.positional_encoding.pe.size(1)}
    """
    
    ax4.text(0.1, 0.9, model_info, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 降低DPI适合屏幕显示
        print(f"✅ Model structure visualization saved to: {save_path}")
    
    plt.show()


def generate_architecture_diagram(model, save_dir):
    """生成模型架构图 - 适合屏幕显示"""
    print("🏗️ Generating architecture diagram...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # 定义颜色
    colors = {
        'input': '#E3F2FD',
        'embedding': '#BBDEFB',
        'attention': '#90CAF9',
        'ffn': '#64B5F6',
        'output': '#42A5F5',
        'connection': '#1976D2'
    }
    
    # 绘制模型架构
    y_pos = 16
    
    # 输入层
    ax.add_patch(Rectangle((1, y_pos), 3, 1, facecolor=colors['input'], edgecolor='black'))
    ax.text(2.5, y_pos + 0.5, 'Source Input\n[batch, src_len]', ha='center', va='center', fontsize=10)
    
    y_pos -= 1.5
    
    # 源词嵌入
    ax.add_patch(Rectangle((1, y_pos), 3, 1, facecolor=colors['embedding'], edgecolor='black'))
    ax.text(2.5, y_pos + 0.5, 'Source Embedding\n+ Positional Encoding', ha='center', va='center', fontsize=10)
    
    y_pos -= 1.5
    
    # 编码器层
    for i in range(len(model.encoder)):
        # 自注意力
        ax.add_patch(Rectangle((0.5, y_pos), 2, 1, facecolor=colors['attention'], edgecolor='black'))
        ax.text(1.5, y_pos + 0.5, f'Multi-Head\nAttention {i+1}', ha='center', va='center', fontsize=9)
        
        # 前馈网络
        ax.add_patch(Rectangle((3, y_pos), 2, 1, facecolor=colors['ffn'], edgecolor='black'))
        ax.text(4, y_pos + 0.5, f'Feed Forward\n{i+1}', ha='center', va='center', fontsize=9)
        
        # 残差连接
        ax.plot([1.5, 1.5], [y_pos + 1, y_pos - 0.5], color=colors['connection'], linewidth=2)
        ax.plot([4, 4], [y_pos + 1, y_pos - 0.5], color=colors['connection'], linewidth=2)
        
        y_pos -= 1.5
    
    # 编码器输出
    ax.add_patch(Rectangle((1, y_pos), 3, 1, facecolor=colors['output'], edgecolor='black'))
    ax.text(2.5, y_pos + 0.5, 'Encoder Output\n[batch, src_len, d_model]', ha='center', va='center', fontsize=10)
    
    y_pos -= 2
    
    # 目标输入
    ax.add_patch(Rectangle((6, y_pos + 1), 3, 1, facecolor=colors['input'], edgecolor='black'))
    ax.text(7.5, y_pos + 1.5, 'Target Input\n[batch, tgt_len]', ha='center', va='center', fontsize=10)
    
    y_pos -= 0.5
    
    # 目标词嵌入
    ax.add_patch(Rectangle((6, y_pos), 3, 1, facecolor=colors['embedding'], edgecolor='black'))
    ax.text(7.5, y_pos + 0.5, 'Target Embedding\n+ Positional Encoding', ha='center', va='center', fontsize=10)
    
    y_pos -= 1.5
    
    # 解码器层
    for i in range(len(model.decoder)):
        # 掩码自注意力
        ax.add_patch(Rectangle((5.5, y_pos), 2, 1, facecolor=colors['attention'], edgecolor='black'))
        ax.text(6.5, y_pos + 0.5, f'Masked\nAttention {i+1}', ha='center', va='center', fontsize=9)
        
        # 交叉注意力
        ax.add_patch(Rectangle((8, y_pos), 2, 1, facecolor=colors['attention'], edgecolor='black'))
        ax.text(9, y_pos + 0.5, f'Cross\nAttention {i+1}', ha='center', va='center', fontsize=9)
        
        # 前馈网络
        ax.add_patch(Rectangle((5.5, y_pos - 1), 4.5, 1, facecolor=colors['ffn'], edgecolor='black'))
        ax.text(7.75, y_pos - 0.5, f'Feed Forward {i+1}', ha='center', va='center', fontsize=9)
        
        y_pos -= 2.5
    
    # 输出投影
    ax.add_patch(Rectangle((6, y_pos), 3, 1, facecolor=colors['output'], edgecolor='black'))
    ax.text(7.5, y_pos + 0.5, 'Output Projection\n[batch, tgt_len, vocab]', ha='center', va='center', fontsize=10)
    
    # 添加连接线
    # 编码器到解码器的连接
    ax.arrow(2.5, 8, 4, 0, head_width=0.2, head_length=0.2, fc=colors['connection'], ec=colors['connection'])
    ax.text(4.5, 8.2, 'Encoder Output', ha='center', va='bottom', fontsize=8)
    
    ax.set_title('Transformer Architecture Diagram', fontsize=14, fontweight='bold', pad=20)
    
    # 保存图片
    arch_path = os.path.join(save_dir, "transformer_architecture.png")
    plt.savefig(arch_path, dpi=150, bbox_inches='tight')  # 降低DPI适合屏幕显示
    plt.show()
    
    print(f"✅ Architecture diagram saved to: {arch_path}")


def main():
    """主函数"""
    print("🚀 Starting simplified Transformer visualization...")
    
    # 创建模型
    model = create_model(
        src_vocab_size=1000, 
        tgt_vocab_size=1000, 
        d_model=512, 
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 创建可视化目录
    vis_dir = "tests/visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # 生成模型结构可视化
    structure_path = os.path.join(vis_dir, "transformer_structure_screen.png")
    visualize_model_structure(model, structure_path)
    
    # 生成架构图
    generate_architecture_diagram(model, vis_dir)
    
    # 保存模型
    model_path = os.path.join(vis_dir, "transformer_model_screen.pt")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to: {model_path}")
    
    print("🎉 Visualization completed!")


if __name__ == "__main__":
    main()
