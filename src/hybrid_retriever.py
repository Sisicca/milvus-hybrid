"""
混合检索器模块

该模块提供了一个封装的混合检索器类，支持：
- 稀疏检索（BM25全文检索）
- 稠密检索（语义向量检索）
- 混合检索（结合稀疏和稠密检索）
"""

import logging
from typing import List, Optional, Dict, Any

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker, WeightedRanker
from sentence_transformers import SentenceTransformer

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
        dense_field: str = "dense_vector"
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
        show_subject: bool = True
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
            if len(content) > 200:
                content = content[:200] + "..."
            line += f"Content: {content}"
            
            output.append(line)
        
        return "\n".join(output)
    
    def close(self):
        """
        关闭Milvus连接
        """
        try:
            self.client.close()
            logger.info("Milvus连接已关闭")
        except Exception as e:
            logger.error(f"关闭连接时发生错误: {e}")


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

