# Milvus混合检索使用指南

## 项目概述

这是一个基于Milvus的医学教科书混合检索系统，结合了BM25全文检索和语义向量检索，提供高质量的医学知识检索能力。

## 主要特性

- ✅ **混合检索**：结合BM25稀疏向量和语义稠密向量
- ✅ **多种检索模式**：支持稀疏检索、稠密检索、混合检索
- ✅ **学科过滤**：可以在特定医学学科中进行检索
- ✅ **可配置的排序策略**：支持RRF和加权排序
- ✅ **完善的日志和错误处理**：便于调试和监控

## 项目结构

```
milvus-hybrid/
├── src/
│   ├── __init__.py                  # 包初始化文件
│   ├── hybrid_retriever.py          # 混合检索器类
│   └── build_db/
│       ├── __init__.py
│       └── build_db_from_txt.py     # 数据库构建脚本
├── examples/
│   └── search_example.py            # 使用示例
├── data/
│   └── textbooks/
│       └── text/                    # 教科书文本文件
├── model-hub/
│   └── Qwen3-Embedding-0.6B/        # 嵌入模型
└── milvus_db_hub/
    └── med_corpus.db                # Milvus数据库文件
```

## 快速开始

### 1. 构建向量数据库

首先，从文本文件构建向量数据库：

```bash
# 使用默认参数
python -m src.build_db.build_db_from_txt

# 自定义参数
python -m src.build_db.build_db_from_txt \
    --uri milvus_db_hub/med_corpus.db \
    --collection_name med_corpus \
    --data_path data/textbooks/text \
    --chunk_size 5000 \
    --chunk_overlap 500 \
    --model_path model-hub/Qwen3-Embedding-0.6B \
    --batch_size 32
```

**参数说明：**

- `--uri`: Milvus数据库文件路径
- `--collection_name`: Collection名称
- `--data_path`: 教科书文本文件所在目录
- `--chunk_size`: 文本块的最大字符数
- `--chunk_overlap`: 相邻文本块之间的重叠字符数
- `--model_path`: SentenceTransformer模型路径
- `--batch_size`: 嵌入生成的批处理大小

### 2. 使用混合检索器

#### 基础使用

```python
from src.hybrid_retriever import HybridRetriever

# 初始化检索器
retriever = HybridRetriever(
    uri="milvus_db_hub/med_corpus.db",
    collection_name="med_corpus",
    model_path="model-hub/Qwen3-Embedding-0.6B"
)

# 执行混合检索
query = "What is positron emission tomography?"
results = retriever.hybrid_search(query, limit=5)

# 格式化并打印结果
print(retriever.format_results(results))

# 关闭连接
retriever.close()
```

#### 稀疏检索（BM25全文检索）

```python
# 基于关键词的全文检索
results = retriever.sparse_search(
    query="diabetes symptoms",
    limit=5
)
```

#### 稠密检索（语义向量检索）

```python
# 基于语义相似度的检索
results = retriever.dense_search(
    query="What are the symptoms of diabetes?",
    limit=5
)
```

#### 混合检索

```python
# 使用RRF（Reciprocal Rank Fusion）融合策略
results = retriever.hybrid_search(
    query="beta blockers mechanism",
    limit=5,
    use_rrf=True,
    rrf_k=60
)

# 使用加权融合策略
results = retriever.hybrid_search(
    query="beta blockers mechanism",
    limit=5,
    use_rrf=False,
    sparse_weight=0.4,
    dense_weight=0.6
)
```

#### 学科过滤检索

```python
# 在特定学科中检索
results = retriever.search_by_subject(
    query="heart anatomy",
    subject="Anatomy",
    limit=3,
    search_type="hybrid"  # 可选: "sparse", "dense", "hybrid"
)
```

#### 自定义过滤条件

```python
# 使用Milvus表达式进行过滤
results = retriever.hybrid_search(
    query="insulin treatment",
    limit=5,
    filter_expr='subject in ["Pharmacology", "InternalMed"]'
)
```

### 3. 运行示例程序

```bash
python examples/search_example.py
```

## API参考

### HybridRetriever类

#### 初始化参数

- `uri` (str): Milvus数据库URI
- `collection_name` (str): Collection名称
- `model_path` (str): SentenceTransformer模型路径
- `sparse_field` (str, 可选): 稀疏向量字段名，默认"sparse_vector"
- `dense_field` (str, 可选): 稠密向量字段名，默认"dense_vector"

#### 主要方法

##### sparse_search()

执行BM25全文检索

**参数：**
- `query` (str): 查询文本
- `limit` (int): 返回结果数量
- `output_fields` (List[str], 可选): 需要返回的字段
- `filter_expr` (str, 可选): 过滤表达式

##### dense_search()

执行语义向量检索

**参数：**
- `query` (str): 查询文本
- `limit` (int): 返回结果数量
- `output_fields` (List[str], 可选): 需要返回的字段
- `filter_expr` (str, 可选): 过滤表达式

##### hybrid_search()

执行混合检索

**参数：**
- `query` (str): 查询文本
- `limit` (int): 返回结果数量
- `output_fields` (List[str], 可选): 需要返回的字段
- `filter_expr` (str, 可选): 过滤表达式
- `sparse_weight` (float): 稀疏检索权重（加权模式）
- `dense_weight` (float): 稠密检索权重（加权模式）
- `use_rrf` (bool): 是否使用RRF融合，默认True
- `rrf_k` (int): RRF的k参数，默认60

##### search_by_subject()

在特定学科中检索

**参数：**
- `query` (str): 查询文本
- `subject` (str): 学科名称
- `limit` (int): 返回结果数量
- `search_type` (str): 检索类型："sparse", "dense", "hybrid"
- `output_fields` (List[str], 可选): 需要返回的字段

##### format_results()

格式化检索结果

**参数：**
- `results` (List[Dict]): 检索结果
- `show_score` (bool): 是否显示分数
- `show_subject` (bool): 是否显示学科

##### close()

关闭Milvus连接

## 支持的医学学科

- Histology (组织学)
- Physiology (生理学)
- Biochemistry (生物化学)
- First_Aid (急救)
- Psichiatry (精神病学)
- Gynecology (妇科学)
- Pediatrics (儿科学)
- Pharmacology (药理学)
- Immunology (免疫学)
- InternalMed (内科学)
- Neurology (神经学)
- Pathoma (病理学基础)
- Pathology (病理学)
- Anatomy (解剖学)
- Obstentrics (产科学)
- Cell_Biology (细胞生物学)
- Surgery (外科学)

## 最佳实践

### 1. 选择合适的检索模式

- **稀疏检索（BM25）**：适合精确关键词匹配
- **稠密检索（语义）**：适合概念性、语义相似的查询
- **混合检索**：大多数情况下的最佳选择，结合两者优势

### 2. 调整融合策略

- **RRF（Reciprocal Rank Fusion）**：
  - 优点：简单有效，无需手动调参
  - 适用：大多数场景的默认选择
  - 参数：`rrf_k`控制融合平滑度，一般60-100

- **加权融合**：
  - 优点：可以精确控制各检索模式的贡献
  - 适用：有特定偏好或优化需求的场景
  - 建议：稀疏0.3-0.5，稠密0.5-0.7

### 3. 分块策略

- `chunk_size`：
  - 较小（2000-3000）：精确性高，但可能丢失上下文
  - 较大（5000-8000）：保留更多上下文，但相关性可能降低
  - 推荐：5000（医学文本）

- `chunk_overlap`：
  - 推荐：chunk_size的10%左右
  - 防止重要信息在分块边界丢失

### 4. 批处理优化

- 构建数据库时使用合适的`batch_size`（16-64）
- 根据可用内存调整批大小

## 性能优化

### 数据库构建优化

1. **使用更大的批处理**：
   ```bash
   python -m src.build_db.build_db_from_txt --batch_size 64
   ```

2. **调整分块大小**以平衡性能和效果

3. **使用GPU加速**（如果可用）：
   模型会自动检测并使用GPU

### 检索优化

1. **限制返回字段**：
   ```python
   results = retriever.hybrid_search(
       query="...",
       output_fields=["content"]  # 只返回必要字段
   )
   ```

2. **合理设置limit**：
   不要返回过多结果

3. **使用过滤表达式**：
   在数据库层面过滤，而不是在应用层

## 常见问题

### Q: 如何更新数据库？

A: 重新运行构建脚本会自动删除旧collection并创建新的。

### Q: 如何添加新的学科？

A: 在`build_db_from_txt.py`的`SUBJECTS`列表中添加新学科名称。

### Q: 检索结果不理想怎么办？

A: 尝试以下方法：
1. 调整混合检索的权重
2. 使用不同的检索模式
3. 调整分块大小和重叠
4. 优化查询语句

### Q: 如何处理内存不足？

A: 
1. 减小batch_size
2. 减小chunk_size
3. 分批处理数据

## 许可证

请参阅项目根目录的LICENSE文件。

