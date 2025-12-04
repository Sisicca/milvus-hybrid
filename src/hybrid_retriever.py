"""
混合检索器模块

该模块提供了一个封装的混合检索器类，支持：
- 稀疏检索（BM25全文检索）
- 稠密检索（语义向量检索）
- 混合检索（结合稀疏和稠密检索）
"""

import logging
from typing import List, Optional, Dict, Any
from openai import OpenAI
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker, WeightedRanker
from sentence_transformers import SentenceTransformer
from torch import futures

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    混合检索器类
    
    结合BM25全文检索和语义向量检索，提供高质量的搜索结果。
    
    Attributes:
        client: Milvus客户端实例
        collection_name: Collection名称
        model: SentenceTransformer模型用于生成查询嵌入
        sparse_field: 稀疏向量字段名
        dense_field: 稠密向量字段名
    """
    
    def __init__(
        self,
        uri: str,
        collection_name: str,
        model_path: str,
        sparse_field: str = "sparse_vector",
        dense_field: str = "dense_vector",
        openai_api_key: str = None,
        openai_base_url: str = None
    ):
        """
        初始化混合检索器
        
        Args:
            uri: Milvus数据库URI
            collection_name: Collection名称
            model_path: SentenceTransformer模型路径
            sparse_field: 稀疏向量字段名，默认为"sparse_vector"
            dense_field: 稠密向量字段名，默认为"dense_vector"
            
        Raises:
            FileNotFoundError: 如果模型路径不存在
            ConnectionError: 如果无法连接到Milvus
        """
        try:
            # 连接Milvus
            logger.info(f"正在连接到Milvus: {uri}")
            self.client = MilvusClient(uri)
            self.collection_name = collection_name
            
            # 检查collection是否存在
            if not self.client.has_collection(collection_name):
                raise ValueError(f"Collection '{collection_name}' 不存在")
            
            logger.info(f"成功连接到Collection: {collection_name}")
            
            # 加载嵌入模型
            logger.info(f"正在加载嵌入模型: {model_path}")
            self.model = SentenceTransformer(model_path)
            logger.info("模型加载成功")
            
            # 设置字段名
            self.sparse_field = sparse_field
            self.dense_field = dense_field
            
        except Exception as e:
            logger.error(f"初始化HybridRetriever时发生错误: {e}")
            raise
        
        try:
            if openai_api_key:
                self.openai_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
            else:
                self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
        except Exception as e:
            logger.error(f"初始化OpenAI客户端时发生错误: {e}")
            raise
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        生成查询的嵌入向量
        
        Args:
            query: 查询文本
            
        Returns:
            嵌入向量
        """
        embedding = self.model.encode([query], prompt_name="query")[0]
        return embedding.tolist()
    
    def sparse_search(
        self,
        query: str,
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        使用BM25进行稀疏检索（全文检索）
        
        Args:
            query: 查询文本
            limit: 返回结果数量限制
            output_fields: 需要返回的字段列表
            filter_expr: 过滤表达式
            
        Returns:
            检索结果列表，每个结果包含distance和entity信息
        """
        if output_fields is None:
            output_fields = ["content", "subject"]
        
        try:
            logger.debug(f"执行稀疏检索: {query}")
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field=self.sparse_field,
                limit=limit,
                output_fields=output_fields,
                filter=filter_expr
            )
            return results[0] if results else []
        except Exception as e:
            logger.error(f"稀疏检索失败: {e}")
            raise
    
    def dense_search(
        self,
        query: str,
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        使用语义向量进行稠密检索
        
        Args:
            query: 查询文本
            limit: 返回结果数量限制
            output_fields: 需要返回的字段列表
            filter_expr: 过滤表达式
            
        Returns:
            检索结果列表，每个结果包含distance和entity信息
        """
        if output_fields is None:
            output_fields = ["content", "subject"]
        
        try:
            logger.debug(f"执行稠密检索: {query}")
            query_embedding = self._get_query_embedding(query)
            
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                anns_field=self.dense_field,
                limit=limit,
                output_fields=output_fields,
                filter=filter_expr
            )
            return results[0] if results else []
        except Exception as e:
            logger.error(f"稠密检索失败: {e}")
            raise
    
    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        use_rrf: bool = True,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        执行混合检索，结合稀疏和稠密检索结果
        
        Args:
            query: 查询文本
            limit: 返回结果数量限制
            output_fields: 需要返回的字段列表
            filter_expr: 过滤表达式
            sparse_weight: 稀疏检索权重（仅在use_rrf=False时使用）
            dense_weight: 稠密检索权重（仅在use_rrf=False时使用）
            use_rrf: 是否使用RRF（Reciprocal Rank Fusion）排序，默认True
            rrf_k: RRF的k参数，默认60
            
        Returns:
            混合检索结果列表
        """
        if output_fields is None:
            output_fields = ["content", "subject"]
        
        try:
            logger.debug(f"执行混合检索: {query}")
            
            # 生成查询嵌入
            query_embedding = self._get_query_embedding(query)
            
            # 创建稀疏检索请求
            sparse_search_params = {"metric_type": "BM25"}
            if filter_expr:
                sparse_search_params["expr"] = filter_expr
            
            sparse_request = AnnSearchRequest(
                data=[query],
                anns_field=self.sparse_field,
                param=sparse_search_params,
                limit=limit
            )
            
            # 创建稠密检索请求
            dense_search_params = {"metric_type": "IP"}
            if filter_expr:
                dense_search_params["expr"] = filter_expr
            
            dense_request = AnnSearchRequest(
                data=[query_embedding],
                anns_field=self.dense_field,
                param=dense_search_params,
                limit=limit
            )
            
            # 选择排序策略
            if use_rrf:
                ranker = RRFRanker(k=rrf_k)
            else:
                ranker = WeightedRanker(sparse_weight, dense_weight)
            
            # 执行混合检索
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[sparse_request, dense_request],
                ranker=ranker,
                limit=limit,
                output_fields=output_fields
            )
            
            return results[0] if results else []
        
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            raise
    
    def search_by_subject(
        self,
        query: str,
        subject: str,
        limit: int = 5,
        search_type: str = "hybrid",
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        在特定学科中进行检索
        
        Args:
            query: 查询文本
            subject: 学科名称
            limit: 返回结果数量限制
            search_type: 检索类型，可选 "sparse", "dense", "hybrid"
            output_fields: 需要返回的字段列表
            
        Returns:
            检索结果列表
        """
        filter_expr = f'subject == "{subject}"'
        
        if search_type == "sparse":
            return self.sparse_search(query, limit, output_fields, filter_expr)
        elif search_type == "dense":
            return self.dense_search(query, limit, output_fields, filter_expr)
        elif search_type == "hybrid":
            return self.hybrid_search(query, limit, output_fields, filter_expr)
        else:
            raise ValueError(f"不支持的检索类型: {search_type}")
    
    def format_results(
        self,
        results: List[Dict[str, Any]],
        show_score: bool = True,
        show_subject: bool = True,
        max_content_length: int = 2000
    ) -> str:
        """
        格式化检索结果为易读的字符串
        
        Args:
            results: 检索结果列表
            show_score: 是否显示相似度分数
            show_subject: 是否显示学科信息
            
        Returns:
            格式化后的字符串
        """
        if not results:
            return "未找到结果"
        
        output = []
        for i, result in enumerate(results, 1):
            line = f"{i}. "
            
            if show_score:
                score = result.get('distance', 0)
                line += f"Score: {score:.4f} | "
            
            if show_subject and 'subject' in result.get('entity', {}):
                subject = result['entity']['subject']
                line += f"Subject: {subject} | "
            
            content = result.get('entity', {}).get('content', '')
            # 截断过长的内容
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            line += f"Content: {content}"
            
            output.append(line)
        
        return "\n".join(output)
    
    def search_results_summary_by_llm(
        self,
        query: str,
        search_type: str = "hybrid",
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
        subject: str = None,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        use_rrf: bool = True,
        rrf_k: int = 60,
        model: str = "gpt-4o-mini",
        summary_mode: str = "stuff",
        custom_prompt: Optional[str] = None,
        final_reduce: bool = True
    ) -> str:
        """
        执行检索并使用LLM总结检索结果
        
        Args:
            query: 查询文本
            search_type: 检索类型，可选 "sparse", "dense", "hybrid"
            limit: 返回结果数量限制
            output_fields: 需要返回的字段列表
            subject: 学科名称
            sparse_weight: 稀疏检索权重（仅在use_rrf=False时使用）
            dense_weight: 稠密检索权重（仅在use_rrf=False时使用）
            use_rrf: 是否使用RRF（Reciprocal Rank Fusion）排序，默认True
            rrf_k: RRF的k参数，默认60
            model: LLM模型名称
            summary_mode: 总结模式，可选 "stuff", "map_reduce"
            custom_prompt: 自定义提示词模板，如果为None则使用默认的医疗QA提示词
            final_reduce: 在map_reduce模式下，是否使用LLM对所有压缩文档进行最终汇总，默认True。
                          如果为False，则直接返回格式化的各文档总结，避免二次总结导致的信息损失。
        
        Returns:
            LLM生成的总结文本
        """
        
        assert summary_mode in ["stuff", "map_reduce"], "不支持的总结模式"
        assert search_type in ["sparse", "dense", "hybrid"], "不支持的检索类型"
        assert sparse_weight >= 0 and sparse_weight <= 1, "稀疏检索权重必须在0到1之间"
        assert dense_weight >= 0 and dense_weight <= 1, "稠密检索权重必须在0到1之间"
        assert use_rrf in [True, False], "use_rrf必须在True或False之间"
        assert rrf_k > 0, "RRF的k参数必须大于0"
        assert limit > 0, "返回结果数量限制必须大于0"
        
        if subject:
            filter_expr = f'subject == "{subject}"'
        else:
            filter_expr = None
        
        # 执行检索
        try:
            if search_type == "sparse":
                results = self.sparse_search(query, limit, output_fields, filter_expr=filter_expr)
            elif search_type == "dense":
                results = self.dense_search(query, limit, output_fields, filter_expr=filter_expr)
            elif search_type == "hybrid":
                results = self.hybrid_search(query, limit, output_fields, filter_expr=filter_expr, sparse_weight=sparse_weight, dense_weight=dense_weight, use_rrf=use_rrf, rrf_k=rrf_k)
            else:
                raise ValueError(f"不支持的检索类型: {search_type}")
        except Exception as e:
            logger.error(f"检索失败: {e}")
            raise
        
        # 检查是否有结果
        if not results:
            logger.warning(f"No results found for query: {query}")
            return "No relevant medical documents found. Please try using different keywords or adjusting search parameters."
        
        logger.info(f"检索到 {len(results)} 条结果，开始使用LLM进行总结...")

        if summary_mode == "stuff":
            formatted_results = self.format_results(results, show_score=True, show_subject=True, max_content_length=20000)
            
            # 使用医疗QA专用提示词
            if custom_prompt:
                prompt = custom_prompt.format(query=query, formatted_results=formatted_results)
            else:
                prompt = f"""You are a professional medical knowledge assistant specializing in extracting and summarizing key information from medical textbooks.

User Question:
{query}

Retrieved Medical Documents:
{formatted_results}

Based on the above search results, please provide an accurate and comprehensive medical answer. Requirements:

1. **Accuracy First**: Ensure all medical terminology, pathological mechanisms, and treatment protocols are accurately stated
2. **Highlight Key Information**: Focus on extracting and presenting the following key information (if relevant):
   - Disease definition, classification, and epidemiological characteristics
   - Etiology and pathogenesis
   - Clinical manifestations and symptoms
   - Diagnostic criteria and examination methods
   - Treatment options (pharmacotherapy, surgical treatment, etc.)
   - Medication guidance (drug names, dosages, duration, contraindications, side effects)
   - Prognosis and complications
   - Related anatomy and physiology knowledge
3. **Structured Presentation**: Use Markdown format with clear headings and lists to organize content
4. **Cite Sources**: If documents contain subject information (Subject), cite the source after relevant content
5. **Completeness**: Do not omit important medical information from the search results
6. **Objectivity**: If information is insufficient or uncertain in the search results, clearly indicate this

Please begin your summary:"""
            
            try:
                response = self._call_llm(model, [{"role": "user", "content": prompt}])
                logger.info("LLM总结完成")
            except Exception as e:
                logger.error(f"使用LLM总结检索结果时发生错误: {e}")
                return formatted_results
            return response
        
        elif summary_mode == "map_reduce":
            # 对每个result进行总结，并使用map_reduce模式进行总结
            futures = {}
            summaries = []  # 存储 (idx, summary) 元组，用于保持顺序
            
            logger.info(f"使用map_reduce模式，对 {len(results)} 个文档分别进行总结...")
            
            with ThreadPoolExecutor(max_workers=min(len(results), 10)) as executor:
                for idx, result in enumerate(results):
                    formatted_result = self.format_results([result], show_score=True, show_subject=True, max_content_length=20000)
                    
                    # 单个文档的总结提示词
                    if custom_prompt:
                        prompt = custom_prompt.format(query=query, formatted_result=formatted_result)
                    else:
                        prompt = f"""# Role & Task
You are a medical information extraction expert. Your task is to analyze the provided medical document excerpt and distill all key information fragments that are **directly relevant** to the user's question.

# Core Instruction
**Strictly extract, summarize, and structure the information from the given text. Do NOT answer the question, provide interpretations, inferences, or any information beyond the provided document.**

# Input
- **User's Question:** {query}
- **Medical Document Excerpt:** {formatted_result}

# Processing Steps
1.  **Identify Relevance:** Scrutinize the document and pinpoint every piece of information (e.g., sentences, clauses, data) that is pertinent to the user's question.
2.  **Extract & Summarize:** For each relevant information fragment, rephrase it concisely while strictly preserving the original meaning and precise medical terminology.
3.  **Categorize & Structure:** Organize the summarized points into the following logical categories. If a category has no relevant information, omit it.

# Output Format & Requirements
- **Output Purpose:** This output is a structured summary of relevant information from ONE document, for subsequent integration.
- **Format:** Use clear, nested bullet points under the following headings. Be concise.

**Structured Summary (Based on Document Content):**
- **Core Concepts & Definitions:** [Summarize relevant definitions and key concepts found.]
- **Pathophysiology & Mechanisms:** [Summarize relevant pathological processes.]
- **Clinical Manifestations:** [Summarize relevant symptoms, signs, etc.]
- **Diagnostic Criteria/Methods:** [Summarize relevant diagnostic information.]
- **Treatment Approaches & Medications:** [Summarize relevant therapeutic strategies and drug information.]
- **Other Critical Details:** [Summarize any other pertinent information, e.g., prognosis, risk factors.]"""
                    
                    future = executor.submit(self._call_llm, model, [{"role": "user", "content": prompt}])
                    futures[future] = (idx, formatted_result)
                
                # 收集所有总结结果
                for future in as_completed(futures):
                    idx, formatted_result = futures[future]
                    try:
                        summary = future.result()
                        summaries.append((idx, summary))
                        logger.debug(f"文档 {idx+1} 总结完成")
                    except Exception as e:
                        logger.error(f"总结文档 {idx+1} 时发生错误: {e}")
                        summaries.append((idx, formatted_result))
            
            if not summaries:
                logger.error("所有文档总结都失败")
                return self.format_results(results, show_score=True, show_subject=True, max_content_length=20000)
            
            # 按文档索引排序，确保顺序一致
            summaries.sort(key=lambda x: x[0])
            
            # 格式化为 [Document X] 形式
            formatted_summaries = []
            for idx, summary in summaries:
                formatted_summaries.append(f"[Document {idx+1}]\n{summary}")
            combined_summaries = "\n\n".join(formatted_summaries)
            
            # 如果不需要最终汇总，直接返回格式化的压缩文档
            if not final_reduce:
                logger.info("final_reduce=False，跳过最终汇总，直接返回各文档的压缩总结")
                return combined_summaries
            
            logger.info("开始汇总所有文档的总结...")
            
            final_prompt = f"""# Role & Task
You are a medical information synthesis specialist. Your task is to integrate several pre-existing summaries (each based on a different medical document) about the same query into one unified, coherent, and non-redundant summary.

# Core Instruction
**Synthesize the information from the provided summaries ONLY. Create a logically flowing narrative that follows standard medical exposition. Do NOT generate new facts, answer the question directly, or make inferences beyond the provided summaries.**

# Input
- **User's Question:** {query}
- **Individual Document Summaries:** {combined_summaries}

# Processing Steps
1.  **Information Fusion:** Merge information from all summaries. When multiple summaries mention the same point, present it once in its most complete and accurate form.
2.  **Logical Structuring:** Organize the fused information into a standard medical knowledge structure. Use the sequence below as a template. Deviate only if the content logically demands it.
    -   **Definition / Overview**
    -   **Etiology & Pathogenesis**
    -   **Clinical Presentation (Symptoms & Signs)**
    -   **Diagnostic Approach**
    -   **Management & Treatment (Including Medications)**
    -   **Prognosis / Other Key Considerations**
3.  **Prioritize Clarity:** Ensure the summary reads as a continuous, well-structured text, not just a list of points from different sources.

# Output Format & Requirements
- **Output Purpose:** This is the final, integrated summary for the user, representing a consolidation of all sourced information.
- **Format:** Use Markdown for clear headings and subheadings. The output should be a flowing document.

**Final Integrated Summary**

[Begin your integrated summary here. Write in full sentences and paragraphs under the logical headings mentioned above.]"""
            
            try:
                response = self._call_llm(model, [{"role": "user", "content": final_prompt}])
                logger.info("最终汇总完成")
            except Exception as e:
                logger.error(f"使用LLM汇总检索结果的总结时发生错误: {e}")
                return combined_summaries
            return response
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def _call_llm(
        self,
        model: str,
        messages: List[Dict[str, Any]]
    ) -> Any:
        """
        调用LLM模型
        
        Args:
            model: LLM模型名称
            messages: 消息列表
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0, # 用于总结文档内容，为保证训练和评估的稳定与一致性，不使用随机性
                max_tokens=8192
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"调用LLM模型时发生错误: {e}")
            raise
    
    def close(self):
        """
        关闭Milvus连接
        """
        try:
            self.client.close()
            logger.info("Milvus连接已关闭")
        except Exception as e:
            logger.error(f"关闭连接时发生错误: {e}")

    def __del__(self) -> None:
        self.close()


def main():
    """
    演示混合检索器的使用方法
    """
    # 初始化检索器
    retriever = HybridRetriever(
        uri="milvus_db_hub/med_corpus.db",
        collection_name="med_corpus",
        model_path="model-hub/Qwen3-Embedding-0.6B"
    )
    
    # 示例查询
    query = "What is positron emission tomography?"
    
    print("=" * 80)
    print(f"查询: {query}")
    print("=" * 80)
    
    # 1. 稀疏检索（BM25全文检索）
    print("\n1. 稀疏检索结果（BM25）:")
    print("-" * 80)
    sparse_results = retriever.sparse_search(query, limit=3)
    print(retriever.format_results(sparse_results))
    
    # 2. 稠密检索（语义向量检索）
    print("\n2. 稠密检索结果（语义相似度）:")
    print("-" * 80)
    dense_results = retriever.dense_search(query, limit=3)
    print(retriever.format_results(dense_results))
    
    # 3. 混合检索
    print("\n3. 混合检索结果（RRF融合）:")
    print("-" * 80)
    hybrid_results = retriever.hybrid_search(query, limit=5)
    print(retriever.format_results(hybrid_results))
    
    # 4. 特定学科检索
    print("\n4. 在解剖学中检索:")
    print("-" * 80)
    subject_results = retriever.search_by_subject(
        query="heart anatomy",
        subject="Anatomy",
        limit=3
    )
    print(retriever.format_results(subject_results))
    
    # 关闭连接
    retriever.close()


if __name__ == "__main__":
    main()

