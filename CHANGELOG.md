# 更新日志

## [未发布] - 2025-11-23

### 新增功能

- ✨ 创建了 `HybridRetriever` 类，提供统一的混合检索接口
- ✨ 支持三种检索模式：稀疏检索（BM25）、稠密检索（语义）、混合检索
- ✨ 支持两种融合策略：RRF（Reciprocal Rank Fusion）和加权融合
- ✨ 支持学科过滤检索
- ✨ 支持自定义过滤表达式
- ✨ 添加结果格式化工具
- ✨ 创建配置管理模块 (`src/config.py`)
- ✨ 添加使用示例 (`examples/search_example.py`)
- ✨ 添加综合测试脚本 (`examples/comprehensive_test.py`)
- ✨ 添加性能基准测试 (`examples/benchmark.py`)

### 改进优化

#### build_db_from_txt.py

- 🔧 修复函数命名拼写错误
  - `determin_subject` → `determine_subject`
  - `textbook_spilt_to_chunks` → `split_textbook_to_chunks`
- 🔧 修复 `get_embeddings` 函数调用问题（缺少model参数）
- 📝 添加完整的类型注解（Type Hints）
- 📝 添加详细的文档字符串（Docstrings）
- 🛡️ 添加完善的错误处理和异常捕获
- 📊 添加日志记录功能
- 🔨 重构代码结构，提高可读性
  - 将 schema 创建逻辑提取为独立函数
  - 将数据加载逻辑提取为独立函数
  - 改进主函数的流程清晰度
- ✅ 添加参数验证
- 🎯 自动获取嵌入向量维度，移除硬编码
- 📖 添加命令行参数帮助信息

#### hybrid_retriever.py

- 🏗️ 创建面向对象的检索器类
- 🛡️ 添加连接验证和错误处理
- 📊 集成日志记录
- 📝 添加详细的文档字符串
- 🔧 提供灵活的配置选项
- 🎨 实现结果格式化方法
- 🧹 实现资源清理方法（close）

### 文档

- 📚 创建详细的使用文档 (`docs/USAGE.md`)
- 📚 更新项目 README
- 📚 添加更新日志（本文件）
- 📚 添加代码注释和文档字符串

### 工程改进

- 📦 添加包初始化文件 (`src/__init__.py`, `src/build_db/__init__.py`)
- 🔧 更新 `.gitignore`，添加更多忽略规则
- 📁 创建 `.gitkeep` 文件保持目录结构
- 🔧 创建配置管理模块

### 性能优化

- ⚡ 支持批量嵌入生成
- ⚡ 优化数据插入流程
- ⚡ 添加性能基准测试工具

## 技术栈

- Python 3.10+
- Milvus (pymilvus 2.6.3+)
- Sentence Transformers 5.1.2+
- LangChain Text Splitters 1.0.0+
- Qwen3-Embedding-0.6B

## 主要特点

1. **完善的类型注解**：所有函数都有清晰的类型提示
2. **详细的文档**：每个类和函数都有文档字符串
3. **错误处理**：全面的异常捕获和处理
4. **日志记录**：完整的日志记录体系
5. **代码质量**：遵循PEP 8规范
6. **易于使用**：提供简洁的API接口
7. **灵活配置**：支持多种配置方式

## 待办事项

- [ ] 添加单元测试
- [ ] 添加集成测试
- [ ] 支持批量查询
- [ ] 添加查询结果缓存
- [ ] 支持查询分析和统计
- [ ] 添加API服务封装
- [ ] 支持分布式部署
- [ ] 添加Web界面

## 贡献者

感谢所有为本项目做出贡献的开发者！

