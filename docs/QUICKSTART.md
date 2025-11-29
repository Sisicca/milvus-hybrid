# 快速开始指南

## 5分钟上手

### 步骤1: 安装依赖

```bash
# 克隆项目
cd milvus-hybrid

# 安装依赖
uv sync  # 或者 pip install -e .
```

### 步骤2: 准备数据和模型

确保以下文件就位：

```
data/textbooks/text/        # 医学教科书文本文件
model-hub/Qwen3-Embedding-0.6B/  # 嵌入模型
```

### 步骤3: 构建数据库

```bash
# 使用默认参数构建
python -m src.build_db.build_db_from_txt

# 大约需要几分钟，取决于数据量
```

### 步骤4: 开始检索

创建 `my_search.py`：

```python
from src.hybrid_retriever import HybridRetriever

# 初始化
retriever = HybridRetriever(
    uri="milvus_db_hub/med_corpus.db",
    collection_name="med_corpus",
    model_path="model-hub/Qwen3-Embedding-0.6B"
)

# 检索
results = retriever.hybrid_search("What is diabetes?", limit=5)

# 显示结果
print(retriever.format_results(results))

# 关闭
retriever.close()
```

运行：

```bash
python my_search.py
```

## 常用代码片段

### 1. 基本检索

```python
# 混合检索（推荐）
results = retriever.hybrid_search("查询内容", limit=5)

# BM25全文检索
results = retriever.sparse_search("查询内容", limit=5)

# 语义向量检索
results = retriever.dense_search("查询内容", limit=5)
```

### 2. 学科过滤

```python
# 在特定学科中检索
results = retriever.search_by_subject(
    query="heart anatomy",
    subject="Anatomy",
    limit=5
)
```

### 3. 自定义过滤

```python
# 多学科过滤
results = retriever.hybrid_search(
    query="treatment",
    limit=5,
    filter_expr='subject in ["Pharmacology", "InternalMed"]'
)
```

### 4. 调整融合策略

```python
# 使用RRF
results = retriever.hybrid_search(
    query="...",
    use_rrf=True,
    rrf_k=60
)

# 使用加权
results = retriever.hybrid_search(
    query="...",
    use_rrf=False,
    sparse_weight=0.4,
    dense_weight=0.6
)
```

## 运行示例程序

```bash
# 基础示例
python examples/search_example.py

# 综合测试
python examples/comprehensive_test.py

# 性能测试
python examples/benchmark.py
```

## 命令行参数

### 构建数据库

```bash
python -m src.build_db.build_db_from_txt \
    --uri milvus_db_hub/med_corpus.db \
    --collection_name med_corpus \
    --data_path data/textbooks/text \
    --model_path model-hub/Qwen3-Embedding-0.6B \
    --chunk_size 5000 \
    --chunk_overlap 500 \
    --batch_size 32
```

查看所有参数：

```bash
python -m src.build_db.build_db_from_txt --help
```

## 常见问题

### Q: 如何查看支持的学科？

A: 查看 `src/build_db/build_db_from_txt.py` 中的 `SUBJECTS` 列表

### Q: 如何自定义分块大小？

A: 使用 `--chunk_size` 参数，推荐值：
- 小文档：2000-3000
- 医学文本：5000-6000
- 长文档：8000-10000

### Q: 如何提高检索速度？

A:
1. 减小 `limit` 值
2. 使用过滤条件预筛选
3. 只返回必要的字段
4. 使用GPU加速（自动检测）

### Q: 检索结果不理想？

A:
1. 尝试不同的检索模式
2. 调整融合策略权重
3. 优化查询语句
4. 调整 `chunk_size`

## 下一步

- 📚 阅读 [完整使用文档](USAGE.md)
- 🔧 查看 [更新日志](../CHANGELOG.md)
- 💡 运行示例程序了解更多功能
- 🧪 使用基准测试评估性能

## 需要帮助？

- 查看文档：`docs/USAGE.md`
- 查看示例：`examples/`
- 查看代码注释：所有函数都有详细文档字符串

祝您使用愉快！🚀

