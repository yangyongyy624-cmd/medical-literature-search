#!/usr/bin/env python3
"""
多模态因果学习 - 表征驱动的跨域泛化方法

核心思想:
1. 模型表征混杂两种信息：
   - 因果特征 (真正相关的)
   - 混淆因子 (导致新域失效)

2. 分离这两种信息：
   - 强调因果特征
   - 文本化混淆因子

3. 边际化计算：
   - 因果特征 × 混淆因子交互
   - 利用先验知识指导

4. 提高跨域泛化能力

基于因果推理和先验知识的表征学习方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class CausalRepresentationLearner(nn.Module):
    """因果表征学习器"""

    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 causal_dim: int = 256, confounder_dim: int = 128,
                 num_domains: int = 3):
        """
        初始化因果表征学习器

        Args:
            input_dim: 输入特征维度 (多模态融合后)
            hidden_dim: 隐藏层维度
            causal_dim: 因果特征维度
            confounder_dim: 混淆因子维度
            num_domains: 域的数量
        """
        super().__init__()

        # 1. 特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 2. 因果特征提取器
        self.causal_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, causal_dim),
        )

        # 3. 混淆因子提取器
        self.confounder_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, confounder_dim),
        )

        # 4. 域分类器 (用于对抗训练)
        self.domain_classifier = nn.Sequential(
            nn.Linear(causal_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_domains),
        )

        # 5. 先验知识引导器
        self.prior_guide = nn.Sequential(
            nn.Linear(causal_dim + confounder_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, causal_dim),
            nn.Sigmoid(),  # 输出权重 [0, 1]
        )

        # 6. 预测器
        self.predictor = nn.Sequential(
            nn.Linear(causal_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 7. 文本编码器 (混淆因子文本描述)
        self.text_encoder = nn.Sequential(
            nn.Linear(768, hidden_dim // 2),  # BERT base 768
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, confounder_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码输入，分离因果特征和混淆因子

        Args:
            x: 输入特征 [batch_size, input_dim]

        Returns:
            causal_features: 因果特征 [batch_size, causal_dim]
            confounder_features: 混淆因子 [batch_size, confounder_dim]
        """
        # 共享编码
        hidden = self.encoder(x)

        # 分离因果特征和混淆因子
        causal_features = self.causal_extractor(hidden)
        confounder_features = self.confounder_extractor(hidden)

        return causal_features, confounder_features

    def text_guided_disentanglement(self, causal_features: torch.Tensor,
                                     confounder_features: torch.Tensor,
                                     confounder_texts: torch.Tensor) -> torch.Tensor:
        """
        文本引导的解耦

        Args:
            causal_features: 因果特征 [batch_size, causal_dim]
            confounder_features: 混淆因子 [batch_size, confounder_dim]
            confounder_texts: 混淆因子的文本描述 [batch_size, 768] (BERT 编码)

        Returns:
            disentangled_features: 解耦后的因果特征 [batch_size, causal_dim]
        """
        # 1. 编码文本描述为混淆因子表示
        text_confounders = self.text_encoder(confounder_texts)

        # 2. 计算混淆因子相似度
        confounder_sim = F.cosine_similarity(
            confounder_features,
            text_confounders,
            dim=1
        )  # [batch_size]

        # 3. 先验知识引导
        combined = torch.cat([causal_features, confounder_features], dim=1)
        prior_weights = self.prior_guide(combined)  # [batch_size, causal_dim]

        # 4. 边际化：从因果特征中去除混淆影响
        # 使用注意力机制，根据文本描述的重要性加权
        confounder_weight = confounder_sim.unsqueeze(1)  # [batch_size, 1]
        disentangled = causal_features * prior_weights * (1 - confounder_weight)

        return disentangled

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        预测

        Args:
            features: 特征 [batch_size, causal_dim]

        Returns:
            prediction: 预测结果 [batch_size, 1]
        """
        return self.predictor(features)

    def domain_classification(self, features: torch.Tensor) -> torch.Tensor:
        """
        域分类 (用于对抗训练)

        Args:
            features: 因果特征 [batch_size, causal_dim]

        Returns:
            domain_logits: 域分类 logits [batch_size, num_domains]
        """
        return self.domain_classifier(features)

    def forward(self, x: torch.Tensor,
                confounder_texts: Optional[torch.Tensor] = None,
                domain_labels: Optional[torch.Tensor] = None) -> Dict:
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, input_dim]
            confounder_texts: 混淆因子文本描述 [batch_size, 768]
            domain_labels: 域标签 [batch_size]

        Returns:
            outputs: 包含所有输出的字典
        """
        # 编码
        causal_features, confounder_features = self.encode(x)

        outputs = {
            'causal_features': causal_features,
            'confounder_features': confounder_features,
        }

        # 文本引导解耦
        if confounder_texts is not None:
            disentangled = self.text_guided_disentanglement(
                causal_features,
                confounder_features,
                confounder_texts
            )
            outputs['disentangled_features'] = disentangled
        else:
            outputs['disentangled_features'] = causal_features

        # 预测
        predictions = self.predict(outputs['disentangled_features'])
        outputs['predictions'] = predictions

        # 域分类 (对抗训练)
        if domain_labels is not None:
            domain_logits = self.domain_classification(causal_features)
            outputs['domain_logits'] = domain_logits

        return outputs


class CausalLoss(nn.Module):
    """因果损失函数"""

    def __init__(self, lambda_adv: float = 0.1,
                 lambda_disen: float = 0.5,
                 lambda_prior: float = 0.3):
        """
        初始化因果损失

        Args:
            lambda_adv: 对抗损失权重
            lambda_disen: 解耦损失权重
            lambda_prior: 先验知识权重
        """
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_disen = lambda_disen
        self.lambda_prior = lambda_prior

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs: Dict, targets: torch.Tensor,
                domain_labels: torch.Tensor) -> Dict:
        """
        计算损失

        Args:
            outputs: 模型输出
            targets: 真实标签 [batch_size, 1]
            domain_labels: 域标签 [batch_size]

        Returns:
            losses: 各种损失的字典
        """
        losses = {}

        # 1. 主任务损失 (预测损失)
        pred_loss = self.mse_loss(
            outputs['predictions'],
            targets
        )
        losses['pred_loss'] = pred_loss

        # 2. 对抗损失 (域不变性)
        if 'domain_logits' in outputs:
            # 梯度反转层
            domain_loss = self.ce_loss(
                outputs['domain_logits'],
                domain_labels
            )
            losses['domain_loss'] = domain_loss * self.lambda_adv

        # 3. 解耦损失 (因果特征和混淆因子正交)
        causal = outputs['causal_features']
        confounder = outputs['confounder_features']
        disen_loss = torch.mean(
            torch.pow(torch.matmul(causal, confounder.t()), 2)
        )
        losses['disen_loss'] = disen_loss * self.lambda_disen

        # 总损失
        total_loss = pred_loss
        if 'domain_loss' in losses:
            total_loss += losses['domain_loss']
        if 'disen_loss' in losses:
            total_loss += losses['disen_loss']

        losses['total_loss'] = total_loss

        return losses


class MultiModalCausalLearner:
    """多模态因果学习器 (训练和推理)"""

    def __init__(self, input_dim: int, device: str = 'cuda'):
        """
        初始化多模态因果学习器

        Args:
            input_dim: 输入特征维度
            device: 设备
        """
        self.device = device
        self.model = CausalRepresentationLearner(input_dim).to(device)
        self.loss_fn = CausalLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict:
        """
        训练一个 epoch

        Args:
            dataloader: 数据加载器

        Returns:
            metrics: 训练指标
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            x = batch['features'].to(self.device)
            y = batch['targets'].to(self.device)
            domain_labels = batch['domain_labels'].to(self.device)
            confounder_texts = batch.get('confounder_texts', None)
            if confounder_texts is not None:
                confounder_texts = confounder_texts.to(self.device)

            # 前向传播
            outputs = self.model(x, confounder_texts, domain_labels)

            # 计算损失
            losses = self.loss_fn(outputs, y, domain_labels)

            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            self.optimizer.step()

            total_loss += losses['total_loss'].item()
            num_batches += 1

        return {
            'avg_loss': total_loss / num_batches,
            'pred_loss': losses['pred_loss'].item(),
            'domain_loss': losses.get('domain_loss', 0),
            'disen_loss': losses.get('disen_loss', 0),
        }

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict:
        """
        评估

        Args:
            dataloader: 数据加载器

        Returns:
            metrics: 评估指标
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch['features'].to(self.device)
                y = batch['targets'].to(self.device)
                domain_labels = batch['domain_labels'].to(self.device)

                outputs = self.model(x, None, domain_labels)
                losses = self.loss_fn(outputs, y, domain_labels)

                total_loss += losses['total_loss'].item()
                all_preds.append(outputs['predictions'].cpu())
                all_targets.append(y.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 计算指标
        mse = ((all_preds - all_targets) ** 2).mean().item()
        mae = torch.abs(all_preds - all_targets).mean().item()
        corr = torch.corrcoef(torch.stack([all_preds.squeeze(), all_targets.squeeze()]))[0, 1].item()

        return {
            'avg_loss': total_loss / len(dataloader),
            'mse': mse,
            'mae': mae,
            'correlation': corr,
        }

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"模型已保存到 {path}")

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"模型已从 {path} 加载")


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 创建模拟数据
    batch_size = 32
    input_dim = 512
    num_domains = 3

    # 模拟数据
    features = torch.randn(batch_size, input_dim)
    targets = torch.randn(batch_size, 1)
    domain_labels = torch.randint(0, num_domains, (batch_size,))
    confounder_texts = torch.randn(batch_size, 768)  # BERT 编码的文本

    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(features, targets, domain_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # 创建模型
    learner = MultiModalCausalLearner(input_dim)

    # 训练
    print("开始训练...")
    for epoch in range(10):
        metrics = learner.train_epoch(dataloader)
        print(f"Epoch {epoch+1}/10 - Loss: {metrics['avg_loss']:.4f}")

    # 评估
    print("\n评估...")
    metrics = learner.evaluate(dataloader)
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Correlation: {metrics['correlation']:.4f}")

    print("\n训练完成!")
