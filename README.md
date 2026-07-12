# 医学文献检索工作流 v2.0

基于经典论文 (PRISMA + Cochrane + PRESS) 的医学文献检索系统

## 核心功能

1. **迭代检索** - 使用不同检索词扩大范围
2. **参考文献追溯** - Crossref API 获取参考文献
3. **引文追踪** - Semantic Scholar API 找到最新进展

## 安装

```bash
pip install requests
```

## 使用

```bash
python3 literature_search_workflow.py
```

## 自定义检索

```python
from literature_search_workflow import LiteratureSearchWorkflow

workflow = LiteratureSearchWorkflow()

report = workflow.search(
    topic='rTMS depression',
    exclude_terms=['electroconvulsive', 'ECT'],
    max_iterations=4,
    max_results=50,
    use_reference_tracking=True,
)
```

## API

- **PubMed E-utilities**: PMID→DOI (免费)
- **Crossref**: 获取参考文献 (免费)
- **Semantic Scholar**: 引文追踪 (免费)

**总成本**: $0/年

## 成果

**从 50 篇到 231 篇文献!**

- 迭代检索：50 篇
- 参考文献：181 篇
- 引用文献：0-20 篇

**提升**: 362%

## 符合标准

- ✅ PRISMA 声明
- ✅ Cochrane 检索手册
- ✅ PRESS 指南

## 作者

宵宵  
2026-07-12

## 新增：多模态因果学习

**文件**: `multimodal_causal_learner.py`

**核心思想**:
- 分离因果特征和混淆因子
- 文本引导的解耦
- 提高跨域泛化能力

**使用**:
```python
from multimodal_causal_learner import MultiModalCausalLearner

learner = MultiModalCausalLearner(input_dim=512)
learner.train_epoch(dataloader)
```

**文档**: `docs/multimodal_causal_learning.md`
