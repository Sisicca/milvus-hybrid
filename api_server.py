"""
FastAPI服务 - 医学知识混合检索API

提供基于Milvus的医学文档检索和LLM总结服务
"""

import os
import argparse
import asyncio
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv
from src.hybrid_retriever import HybridRetriever

load_dotenv()

# 初始化FastAPI应用
app = FastAPI(
    title="医学知识检索API",
    description="基于Milvus混合检索和LLM的医学知识问答服务",
    version="1.0.0"
)

# 全局检索器实例
retriever: Optional[HybridRetriever] = None

# 线程池执行器（用于并发处理）
executor: Optional[ThreadPoolExecutor] = None

# 全局配置：是否在map_reduce模式下进行最终汇总
final_reduce_enabled: bool = True


class SearchQuery(BaseModel):
    """单个搜索查询"""
    query: str = Field(..., description="查询问题")
    subject: Optional[str] = Field(None, description="指定的学科，如：Anatomy, Pharmacology等")


class SearchRequest(BaseModel):
    """搜索请求"""
    queries: List[SearchQuery] = Field(..., description="查询列表")


class SearchResult(BaseModel):
    """单个搜索结果"""
    query: str = Field(..., description="原始查询问题")
    subject: Optional[str] = Field(None, description="指定的学科")
    summary: str = Field(..., description="LLM生成的总结答案")


class SearchResponse(BaseModel):
    """搜索响应"""
    results: List[SearchResult] = Field(..., description="搜索结果列表")


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化检索器和线程池"""
    global retriever, executor
    try:
        # 初始化检索器
        retriever = HybridRetriever(
            uri="./milvus_db_hub/med_corpus.db",
            collection_name="med_corpus",
            model_path="./model-hub/Qwen3-Embedding-0.6B",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL")
        )
        print("✓ 混合检索器初始化成功")
        
        # 初始化线程池
        max_workers = int(os.getenv("MAX_WORKERS", "10"))
        executor = ThreadPoolExecutor(max_workers=max_workers)
        print(f"✓ 线程池初始化成功 (max_workers={max_workers})")
        
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global retriever, executor
    if retriever:
        retriever.close()
        print("✓ 已关闭检索器连接")
    if executor:
        executor.shutdown(wait=True)
        print("✓ 已关闭线程池")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "医学知识检索API",
        "version": "1.0.0",
        "endpoints": {
            "POST /search": "执行医学知识检索和总结（map_reduce模式，深度分析）",
            "POST /search_fast": "快速检索和总结（stuff模式，快速响应）"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    if retriever is None:
        raise HTTPException(status_code=503, detail="检索器未初始化")
    return {"status": "healthy"}


async def _process_single_query(
    query_item: SearchQuery,
    summary_mode: str = "map_reduce",
    limit: int = 10,
    final_reduce: bool = True
) -> SearchResult:
    """
    处理单个查询（在线程池中执行）
    
    Args:
        query_item: 查询项
        summary_mode: 总结模式
        limit: 检索文档数量限制
        final_reduce: 在map_reduce模式下，是否使用LLM对所有压缩文档进行最终汇总
    
    Returns:
        搜索结果
    """
    try:
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行同步方法
        summary = await loop.run_in_executor(
            executor,
            lambda: retriever.search_results_summary_by_llm(
                query=query_item.query,
                search_type="hybrid",
                limit=limit,
                subject=query_item.subject,
                use_rrf=True,
                summary_mode=summary_mode,
                model="gpt-4o-mini",
                final_reduce=final_reduce
            )
        )
        
        return SearchResult(
            query=query_item.query,
            subject=query_item.subject,
            summary=summary
        )
        
    except Exception as e:
        # 如果查询失败，返回错误信息
        return SearchResult(
            query=query_item.query,
            subject=query_item.subject,
            summary=f"检索失败: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    执行医学知识检索和总结（深度分析版）
    
    使用混合检索（BM25 + 语义向量）+ Map-Reduce模式LLM总结
    - 检索10个文档
    - 使用map_reduce模式进行深度分析
    - 并发处理多个查询，显著提升速度
    - 通过启动参数 --final-reduce 控制是否对所有压缩文档进行最终汇总
    
    适合：需要全面、深入分析的场景
    """
    if retriever is None or executor is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    if not request.queries:
        raise HTTPException(status_code=400, detail="查询列表不能为空")
    
    # 并发处理所有查询
    tasks = [
        _process_single_query(
            query_item, 
            summary_mode="map_reduce", 
            limit=10,
            final_reduce=final_reduce_enabled
        )
        for query_item in request.queries
    ]
    
    # 使用 asyncio.gather 并发执行所有任务
    results = await asyncio.gather(*tasks)
    
    return SearchResponse(results=list(results))


@app.post("/search_fast", response_model=SearchResponse)
async def search_fast(request: SearchRequest):
    """
    快速检索和总结（轻量级版）
    
    使用混合检索（BM25 + 语义向量）+ Stuff模式LLM总结
    - 仅检索3个最相关文档
    - 使用stuff模式一次性总结
    - 并发处理多个查询
    - 更快的响应速度，降低API流量压力
    
    适合：需要快速响应的场景，或高并发场景
    """
    if retriever is None or executor is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    if not request.queries:
        raise HTTPException(status_code=400, detail="查询列表不能为空")
    
    # 并发处理所有查询（使用stuff模式和更少的文档）
    tasks = [
        _process_single_query(query_item, summary_mode="stuff", limit=3)
        for query_item in request.queries
    ]
    
    # 使用 asyncio.gather 并发执行所有任务
    results = await asyncio.gather(*tasks)
    
    return SearchResponse(results=list(results))


def main():
    """启动服务"""
    global final_reduce_enabled
    
    parser = argparse.ArgumentParser(description="医学知识检索API服务")
    parser.add_argument(
        "--host", 
        type=str, 
        default=os.getenv("HOST", "0.0.0.0"),
        help="服务监听地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=int(os.getenv("PORT", "8000")),
        help="服务监听端口 (默认: 8000)"
    )
    parser.add_argument(
        "--final-reduce",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="在map_reduce模式下，是否使用LLM对所有压缩文档进行最终汇总 (默认: true)。"
             "设置为false时直接返回各文档的压缩总结（[Document1]...形式），避免二次总结导致的信息损失。"
    )
    
    args = parser.parse_args()
    
    # 设置全局配置
    final_reduce_enabled = args.final_reduce
    
    print(f"启动医学知识检索API服务...")
    print(f"服务地址: http://{args.host}:{args.port}")
    print(f"API文档: http://{args.host}:{args.port}/docs")
    print(f"final_reduce: {final_reduce_enabled} ({'启用最终汇总' if final_reduce_enabled else '跳过最终汇总，直接返回各文档压缩总结'})")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=False
    )


if __name__ == "__main__":
    main()

