"""
构建Milvus向量数据库的脚本

该脚本从文本文件中读取医学教科书内容，将其分块并嵌入到向量中，
然后将数据存储到Milvus数据库中，支持混合检索（稠密向量+稀疏向量）。
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import DataType, Function, FunctionType, MilvusClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 支持的医学学科列表
SUBJECTS = [
    "Histology",
    "Physiology",
    "Biochemistry",
    "First_Aid",
    "Psichiatry",
    "Gynecology",
    "Pediatrics",
    "Pharmacology",
    "Immunology",
    "InternalMed",
    "Neurology",
    "Pathoma",
    "Pathology",
    "Anatomy",
    "Obstentrics",
    "Cell_Biology",
    "Surgery"
]


def determine_subject(file_name: str) -> str:
    """
    根据文件名确定医学学科
    
    Args:
        file_name: 文件名
        
    Returns:
        学科名称，如果无法识别则返回 "Unknown"
    """
    for subject in SUBJECTS:
        if file_name.startswith(subject):
            return subject
    return "Unknown"


def split_textbook_to_chunks(
    textbook: str,
    chunk_size: int = 5000,
    chunk_overlap: int = 500
) -> List[str]:
    """
    将教科书文本分割成较小的块
    
    Args:
        textbook: 教科书文本内容
        chunk_size: 每个块的最大字符数
        chunk_overlap: 相邻块之间的重叠字符数
        
    Returns:
        文本块列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(textbook)
    return chunks


def get_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    is_query: bool = False,
    batch_size: int = 32
) -> List:
    """
    使用嵌入模型将文本转换为向量
    
    Args:
        model: SentenceTransformer 模型实例
        texts: 要嵌入的文本列表
        is_query: 是否为查询文本（使用不同的prompt）
        batch_size: 批处理大小
        
    Returns:
        嵌入向量列表
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="生成嵌入向量"):
        batch_texts = texts[i:i+batch_size]
        if is_query:
            embeddings.extend(model.encode(batch_texts, prompt_name="query"))
        else:
            embeddings.extend(model.encode(batch_texts))
    return embeddings

def load_textbooks(data_path: str) -> Dict[str, str]:
    """
    从指定路径加载所有教科书文件
    
    Args:
        data_path: 教科书文件所在目录路径
        
    Returns:
        学科名称到教科书内容的字典
        
    Raises:
        FileNotFoundError: 如果指定路径不存在
        IOError: 如果文件读取失败
    """
    folder_path = Path(data_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"数据路径不存在: {data_path}")
    
    if not folder_path.is_dir():
        raise NotADirectoryError(f"路径不是目录: {data_path}")
    
    med_corpus_dict = {}
    
    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix == '.txt':
            subject = determine_subject(file_path.name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    textbook = f.read()
                    if subject not in med_corpus_dict:
                        med_corpus_dict[subject] = ""
                    med_corpus_dict[subject] += textbook
                logger.info(f"已加载: {file_path.name} -> {subject}")
            except Exception as e:
                logger.error(f"读取文件失败 {file_path.name}: {e}")
                continue
    
    if not med_corpus_dict:
        raise ValueError(f"未找到任何有效的文本文件: {data_path}")
    
    return med_corpus_dict


def create_milvus_schema(embedding_dim: int) -> tuple:
    """
    创建Milvus collection的schema和索引参数
    
    Args:
        embedding_dim: 嵌入向量的维度
        
    Returns:
        (schema, index_params) 元组
    """
    analyzer_params = {"tokenizer": "standard", "filter": ["lowercase"]}
    
    schema = MilvusClient.create_schema()
    
    # 主键字段
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        auto_id=True,
        max_length=100,
    )
    
    # 文本内容字段
    schema.add_field(
        field_name="content",
        datatype=DataType.VARCHAR,
        max_length=65535,
        analyzer_params=analyzer_params,
        enable_match=True,
        enable_analyzer=True,
    )
    
    # 学科字段
    schema.add_field(
        field_name="subject",
        datatype=DataType.VARCHAR,
        max_length=100
    )
    
    # 稀疏向量字段（用于BM25全文检索）
    schema.add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR
    )
    
    # 稠密向量字段（用于语义检索）
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=embedding_dim,
    )
    
    # 添加BM25函数用于生成稀疏向量
    bm25_function = Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["content"],
        output_field_names="sparse_vector",
    )
    schema.add_function(bm25_function)
    
    # 创建索引参数
    index_params = MilvusClient.prepare_index_params()
    
    # 稀疏向量索引（BM25）
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )
    
    # 稠密向量索引（语义相似度）
    index_params.add_index(
        field_name="dense_vector",
        index_type="FLAT",
        metric_type="IP"
    )
    
    return schema, index_params


def main(args):
    """
    主函数：构建Milvus向量数据库
    
    Args:
        args: 命令行参数
    """
    try:
        # Step 1: 加载嵌入模型
        logger.info(f"Step 1: 从路径加载嵌入模型: {args.model_path}")
        model = SentenceTransformer(args.model_path)
        embedding_dim = model.get_sentence_embedding_dimension()
        logger.info(f"模型加载成功，嵌入维度: {embedding_dim}")
        
        # Step 2: 加载教科书数据
        logger.info(f"Step 2: 从路径加载教科书数据: {args.data_path}")
        med_corpus_dict = load_textbooks(args.data_path)
        logger.info(f"加载了 {len(med_corpus_dict)} 个学科的教科书")
        
        # Step 3: 分块处理
        logger.info("Step 3: 将教科书分割成文本块")
        med_corpus_chunks_dict = {}
        total_chunks = 0
        
        for subject, textbook in tqdm(med_corpus_dict.items(), desc="分割文本"):
            chunks = split_textbook_to_chunks(
                textbook,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            med_corpus_chunks_dict[subject] = chunks
            total_chunks += len(chunks)
            logger.info(f"{subject}: {len(chunks)} 个文本块")
        
        logger.info(f"总共生成 {total_chunks} 个文本块")
        
        # Step 4: 创建Milvus collection
        logger.info("Step 4: 创建Milvus数据库")
        client = MilvusClient(args.uri)
        
        # 如果collection已存在，删除它
        if client.has_collection(args.collection_name):
            logger.warning(f"Collection '{args.collection_name}' 已存在，将被删除")
            client.drop_collection(args.collection_name)
        
        # 创建schema和索引
        schema, index_params = create_milvus_schema(embedding_dim)
        
        # 创建collection
        client.create_collection(
            collection_name=args.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Collection '{args.collection_name}' 创建成功")
        
        # Step 5: 生成嵌入并插入数据
        logger.info("Step 5: 生成嵌入向量并插入数据")
        entities = []
        
        for subject, chunks in tqdm(
            med_corpus_chunks_dict.items(),
            desc="处理学科"
        ):
            # 批量生成嵌入
            embeddings = get_embeddings(
                model,
                chunks,
                is_query=False,
                batch_size=args.batch_size
            )
            
            # 构建实体列表
            for i, chunk in enumerate(chunks):
                entities.append({
                    "content": chunk,
                    "dense_vector": embeddings[i],
                    "subject": subject
                })
        
        # 批量插入数据
        logger.info(f"向数据库插入 {len(entities)} 条记录")
        client.insert(args.collection_name, entities)
        logger.info("数据插入完成")
        
        # 显示统计信息
        stats = client.get_collection_stats(args.collection_name)
        logger.info(f"Collection统计信息: {stats}")
        logger.info("数据库构建完成！")
        
    except Exception as e:
        logger.error(f"构建数据库时发生错误: {e}", exc_info=True)
        raise



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构建医学教科书的Milvus向量数据库",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--uri",
        type=str,
        default="milvus_db_hub/med_corpus.db",
        help="Milvus数据库文件路径"
    )
    
    parser.add_argument(
        "--collection_name",
        type=str,
        default="med_corpus",
        help="Collection名称"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/textbooks/text",
        help="教科书文本文件所在目录"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000,
        help="文本块的最大字符数"
    )
    
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=500,
        help="相邻文本块之间的重叠字符数"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="model-hub/Qwen3-Embedding-0.6B",
        help="SentenceTransformer模型路径"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="嵌入生成的批处理大小"
    )
    
    args = parser.parse_args()
    
    main(args)

