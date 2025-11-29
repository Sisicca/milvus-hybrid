"""
配置管理模块

提供项目的配置管理功能
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class DatabaseConfig:
    """数据库配置"""
    uri: str = "milvus_db_hub/med_corpus.db"
    collection_name: str = "med_corpus"
    sparse_field: str = "sparse_vector"
    dense_field: str = "dense_vector"
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """从环境变量加载配置"""
        return cls(
            uri=os.getenv("MILVUS_URI", cls.uri),
            collection_name=os.getenv("COLLECTION_NAME", cls.collection_name),
            sparse_field=os.getenv("SPARSE_FIELD", cls.sparse_field),
            dense_field=os.getenv("DENSE_FIELD", cls.dense_field),
        )


@dataclass
class ModelConfig:
    """模型配置"""
    model_path: str = "model-hub/Qwen3-Embedding-0.6B"
    batch_size: int = 32
    
    @classmethod
    def from_env(cls) -> "ModelConfig":
        """从环境变量加载配置"""
        return cls(
            model_path=os.getenv("MODEL_PATH", cls.model_path),
            batch_size=int(os.getenv("BATCH_SIZE", str(cls.batch_size))),
        )


@dataclass
class DataConfig:
    """数据配置"""
    data_path: str = "data/textbooks/text"
    chunk_size: int = 5000
    chunk_overlap: int = 500
    
    @classmethod
    def from_env(cls) -> "DataConfig":
        """从环境变量加载配置"""
        return cls(
            data_path=os.getenv("DATA_PATH", cls.data_path),
            chunk_size=int(os.getenv("CHUNK_SIZE", str(cls.chunk_size))),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(cls.chunk_overlap))),
        )


@dataclass
class AppConfig:
    """应用配置"""
    database: DatabaseConfig
    model: ModelConfig
    data: DataConfig
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """从环境变量加载配置"""
        return cls(
            database=DatabaseConfig.from_env(),
            model=ModelConfig.from_env(),
            data=DataConfig.from_env(),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
    
    @classmethod
    def default(cls) -> "AppConfig":
        """获取默认配置"""
        return cls(
            database=DatabaseConfig(),
            model=ModelConfig(),
            data=DataConfig(),
        )


def get_config() -> AppConfig:
    """
    获取应用配置
    
    优先从环境变量读取，如果环境变量不存在则使用默认值
    
    Returns:
        AppConfig实例
    """
    return AppConfig.from_env()


# 全局配置实例
config = get_config()

