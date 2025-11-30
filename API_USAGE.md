# API 服务使用指南

## 概述

医学知识检索API提供了两个检索端点，满足不同场景的需求：

| 端点 | 模式 | 文档数 | 适用场景 | 响应速度 |
|-----|------|--------|---------|---------|
| `/search` | map_reduce | 10 | 深度分析，需要全面信息 | 较慢 |
| `/search_fast` | stuff | 3 | 快速响应，高并发场景 | 快速 |

## 关键改进

### ✨ 并发处理优化

**问题**：之前多个查询需要串行处理，速度慢
- 10个查询 × 10秒/查询 = **100秒**

**解决方案**：使用 ThreadPoolExecutor + asyncio 实现并发
- 10个查询并发处理 ≈ **10-15秒**
- **性能提升：7-10倍** 🚀

### 🎯 两种端点对比

#### `/search` - 深度分析端点
```python
# 特点
- 检索10个相关文档
- 使用map_reduce模式：先分别总结每个文档，再汇总
- 更全面、更深入的答案
- 适合复杂医学问题

# 响应时间
- 单个查询：~8-12秒
- 10个查询（并发）：~10-20秒
```

#### `/search_fast` - 快速响应端点
```python
# 特点
- 仅检索3个最相关文档
- 使用stuff模式：一次性总结
- 更快的响应速度
- 降低OpenAI API流量压力

# 响应时间
- 单个查询：~3-5秒
- 10个查询（并发）：~5-10秒
```

## 启动服务

```bash
# 设置环境变量
export OPENAI_API_KEY='your-api-key'
export OPENAI_BASE_URL='your-base-url'  # 可选
export MAX_WORKERS='10'  # 可选，线程池大小

# 启动服务
python api_server.py

# 服务地址
# - API: http://localhost:8000
# - 文档: http://localhost:8000/docs
```

## API 使用示例

### 1. 健康检查

```bash
curl http://localhost:8000/health
```

### 2. 单个查询（深度分析）

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {
        "query": "What is diabetes mellitus?",
        "subject": null
      }
    ]
  }'
```

### 3. 多个查询（并发处理）

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {"query": "What is hypertension?", "subject": null},
      {"query": "What are beta blockers?", "subject": "Pharmacology"},
      {"query": "Explain cardiac anatomy", "subject": "Anatomy"}
    ]
  }'
```

### 4. 快速查询（高并发场景）

```bash
curl -X POST http://localhost:8000/search_fast \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {"query": "What is pneumonia?", "subject": null}
    ]
  }'
```

### 5. 学科过滤

```bash
curl -X POST http://localhost:8000/search_fast \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {
        "query": "How do ACE inhibitors work?",
        "subject": "Pharmacology"
      }
    ]
  }'
```

## Python 客户端示例

```python
import requests

API_URL = "http://localhost:8000"

# 单个查询
response = requests.post(
    f"{API_URL}/search_fast",
    json={
        "queries": [
            {"query": "What is diabetes?", "subject": None}
        ]
    }
)

result = response.json()
print(result['results'][0]['summary'])

# 批量查询（自动并发处理）
response = requests.post(
    f"{API_URL}/search_fast",
    json={
        "queries": [
            {"query": "What is hypertension?", "subject": None},
            {"query": "What are beta blockers?", "subject": "Pharmacology"},
            {"query": "Explain heart anatomy", "subject": "Anatomy"},
            # ... 可以添加更多查询
        ]
    },
    timeout=300  # 设置合适的超时时间
)

results = response.json()['results']
for i, result in enumerate(results, 1):
    print(f"{i}. {result['query']}")
    print(f"   {result['summary'][:100]}...\n")
```

## 性能测试

```bash
# 运行完整测试套件
python examples/test_api_service.py

# 选择测试：
# 0. 运行所有测试
# 6. 快速端点测试
# 7. 性能对比测试
# 8. 并发性能测试（10个查询）
# 9. 仅运行性能测试（6-8）
```

## 性能基准

### 单个查询

| 端点 | 响应时间 |
|-----|---------|
| `/search` | 8-12秒 |
| `/search_fast` | 3-5秒 |

### 10个查询（并发）

| 端点 | 串行时间（旧） | 并发时间（新） | 加速比 |
|-----|--------------|--------------|-------|
| `/search` | ~100秒 | ~10-20秒 | **5-10x** |
| `/search_fast` | ~50秒 | ~5-10秒 | **5-10x** |

## 建议

### 何时使用 `/search`
- ✅ 需要全面、深入的医学分析
- ✅ 复杂的临床问题
- ✅ 研究和学习场景
- ❌ 不适合高并发场景

### 何时使用 `/search_fast`
- ✅ 需要快速响应
- ✅ 高并发场景（如Web应用）
- ✅ 简单的医学问题
- ✅ 降低API流量成本
- ✅ 对OpenAI有速率限制时

### 并发处理建议

```python
# 推荐：一次请求包含多个查询（自动并发）
response = requests.post(url, json={
    "queries": [query1, query2, query3, ...]  # 自动并发
})

# 不推荐：多次单独请求
for query in queries:
    response = requests.post(url, json={"queries": [query]})  # 串行
```

## 配置优化

### 环境变量

```bash
# OpenAI配置
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=your-base-url  # 可选

# 服务配置
HOST=0.0.0.0
PORT=8000
MAX_WORKERS=10  # 线程池大小，根据服务器资源调整
```

### 线程池大小建议

| 场景 | MAX_WORKERS |
|-----|-------------|
| 开发/测试 | 5-10 |
| 生产环境（低并发） | 10-20 |
| 生产环境（高并发） | 20-50 |

⚠️ **注意**：线程池过大会增加内存开销和OpenAI API压力

## 常见问题

### Q1: 为什么并发查询还是很慢？
A: 检查以下几点：
- OpenAI API响应速度
- 网络延迟
- 线程池大小（MAX_WORKERS）
- 是否触发OpenAI速率限制

### Q2: 如何避免OpenAI速率限制？
A: 
- 使用 `/search_fast` 端点（减少API调用）
- 适当调小 MAX_WORKERS
- 实现请求队列和速率限制
- 考虑使用缓存

### Q3: 单个查询和批量查询性能一样吗？
A: 不一样。批量查询利用并发，性能更好：
- 单个查询：无并发优势
- 批量查询（10个）：并发处理，总时间接近单个查询时间

## 技术实现细节

### 并发架构

```
FastAPI (异步)
    ↓
ThreadPoolExecutor (线程池)
    ↓
HybridRetriever (同步)
    ↓ (并发)
[Query1] [Query2] [Query3] ... [QueryN]
    ↓      ↓       ↓            ↓
  Milvus 检索
    ↓      ↓       ↓            ↓
  LLM 总结
    ↓      ↓       ↓            ↓
asyncio.gather (汇总结果)
```

### 关键代码

```python
# API层：使用asyncio + 线程池
async def _process_single_query(query_item):
    loop = asyncio.get_event_loop()
    summary = await loop.run_in_executor(
        executor,
        lambda: retriever.search_results_summary_by_llm(...)
    )
    return summary

# 并发处理
tasks = [_process_single_query(q) for q in queries]
results = await asyncio.gather(*tasks)
```

## 监控和日志

服务启动后会输出日志：
```
✓ 混合检索器初始化成功
✓ 线程池初始化成功 (max_workers=10)
启动医学知识检索API服务...
服务地址: http://0.0.0.0:8000
API文档: http://0.0.0.0:8000/docs
```

每个请求都会记录：
- 检索结果数量
- LLM总结状态
- 异常信息（如果有）

## 更新日志

### v1.1.0 - 并发优化
- ✨ 添加 ThreadPoolExecutor 实现并发处理
- ✨ 新增 `/search_fast` 快速端点
- 🚀 性能提升 5-10倍
- 📝 完善测试脚本

### v1.0.0 - 初始版本
- ✨ 基础检索功能
- ✨ LLM总结功能
- ✨ 学科过滤

---

**技术支持**: 如有问题，请查看 `/docs` 自动生成的API文档

