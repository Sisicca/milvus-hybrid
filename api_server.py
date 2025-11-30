"""
FastAPI服务 - 医学知识混合检索API

提供基于Milvus的医学文档检索和LLM总结服务
"""

import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from src.hybrid_retriever import HybridRetriever

# 初始化FastAPI应用
app = FastAPI(
    title="医学知识检索API",
    description="基于Milvus混合检索和LLM的医学知识问答服务",
    version="1.0.0"
)

# 全局检索器实例
retriever: Optional[HybridRetriever] = None


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
    """应用启动时初始化检索器"""
    global retriever
    try:
        retriever = HybridRetriever(
            uri=os.getenv("MILVUS_URI", "milvus_db_hub/med_corpus.db"),
            collection_name=os.getenv("COLLECTION_NAME", "med_corpus"),
            model_path=os.getenv("MODEL_PATH", "model-hub/Qwen3-Embedding-0.6B"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL")
        )
        print("✓ 混合检索器初始化成功")
    except Exception as e:
        print(f"✗ 初始化检索器失败: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global retriever
    if retriever:
        retriever.close()
        print("✓ 已关闭检索器连接")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "医学知识检索API",
        "version": "1.0.0",
        "endpoints": {
            "POST /search": "执行医学知识检索和总结"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    if retriever is None:
        raise HTTPException(status_code=503, detail="检索器未初始化")
    return {"status": "healthy"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    执行医学知识检索和总结
    
    使用混合检索（BM25 + 语义向量）和LLM进行智能总结
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="检索器未初始化")
    
    if not request.queries:
        raise HTTPException(status_code=400, detail="查询列表不能为空")
    
    results = []
    
    for query_item in request.queries:
        try:
            # 使用混合检索 + map_reduce总结
            summary = retriever.search_results_summary_by_llm(
                query=query_item.query,
                search_type="hybrid",
                limit=10,
                subject=query_item.subject,
                use_rrf=True,
                summary_mode="map_reduce"
            )
            
            results.append(SearchResult(
                query=query_item.query,
                subject=query_item.subject,
                summary=summary
            ))
            
        except Exception as e:
            # 如果单个查询失败，返回错误信息
            results.append(SearchResult(
                query=query_item.query,
                subject=query_item.subject,
                summary=f"检索失败: {str(e)}"
            ))
    
    return SearchResponse(results=results)


def main():
    """启动服务"""
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"启动医学知识检索API服务...")
    print(f"服务地址: http://{host}:{port}")
    print(f"API文档: http://{host}:{port}/docs")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False
    )


if __name__ == "__main__":
    main()

