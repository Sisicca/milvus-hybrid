"""
API服务测试脚本

测试医学知识检索API服务
"""

import requests
import json
from typing import List, Dict

# API服务地址
API_BASE_URL = "http://localhost:8000"


def test_health_check():
    """测试健康检查"""
    print("=" * 80)
    print("测试健康检查")
    print("=" * 80)
    
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}\n")


def test_single_query():
    """测试单个查询"""
    print("=" * 80)
    print("测试单个查询")
    print("=" * 80)
    
    payload = {
        "queries": [
            {
                "query": "What is diabetes mellitus?",
                "subject": None
            }
        ]
    }
    
    print(f"查询: {payload['queries'][0]['query']}")
    response = requests.post(f"{API_BASE_URL}/search", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n答案:\n{result['results'][0]['summary']}\n")
    else:
        print(f"错误: {response.status_code} - {response.text}\n")


def test_multiple_queries():
    """测试多个查询"""
    print("=" * 80)
    print("测试多个查询")
    print("=" * 80)
    
    payload = {
        "queries": [
            {
                "query": "What are beta blockers?",
                "subject": "Pharmacology"
            },
            {
                "query": "Describe the anatomy of the heart",
                "subject": "Anatomy"
            },
            {
                "query": "What are the symptoms of myocardial infarction?",
                "subject": "InternalMed"
            }
        ]
    }
    
    response = requests.post(f"{API_BASE_URL}/search", json=payload)
    
    if response.status_code == 200:
        results = response.json()['results']
        for i, result in enumerate(results, 1):
            print(f"\n问题 {i}: {result['query']}")
            print(f"学科: {result['subject']}")
            print(f"答案:\n{result['summary']}")
            print("-" * 80)
    else:
        print(f"错误: {response.status_code} - {response.text}\n")


def test_subject_filtering():
    """测试学科过滤"""
    print("=" * 80)
    print("测试学科过滤")
    print("=" * 80)
    
    subjects_tests = [
        ("Anatomy", "What is the structure of the kidney?"),
        ("Pharmacology", "How do ACE inhibitors work?"),
        ("Pathology", "What is necrosis?")
    ]
    
    for subject, query in subjects_tests:
        print(f"\n学科: {subject}")
        print(f"问题: {query}")
        
        payload = {
            "queries": [
                {
                    "query": query,
                    "subject": subject
                }
            ]
        }
        
        response = requests.post(f"{API_BASE_URL}/search", json=payload)
        
        if response.status_code == 200:
            result = response.json()['results'][0]
            print(f"答案: {result['summary'][:200]}...")
        else:
            print(f"错误: {response.status_code}")
        print("-" * 40)


def test_batch_queries():
    """测试批量查询"""
    print("=" * 80)
    print("测试批量查询（5个问题）")
    print("=" * 80)
    
    payload = {
        "queries": [
            {"query": "What is hypertension?", "subject": None},
            {"query": "Define sepsis", "subject": "InternalMed"},
            {"query": "Explain wound healing", "subject": "Surgery"},
            {"query": "What are NSAIDs?", "subject": "Pharmacology"},
            {"query": "Describe the cerebral cortex", "subject": "Anatomy"}
        ]
    }
    
    print("发送5个查询请求...")
    response = requests.post(f"{API_BASE_URL}/search", json=payload)
    
    if response.status_code == 200:
        results = response.json()['results']
        print(f"成功返回 {len(results)} 个结果\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['query']}")
            print(f"   学科: {result['subject'] or '全部'}")
            print(f"   答案: {result['summary'][:100]}...\n")
    else:
        print(f"错误: {response.status_code} - {response.text}\n")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print(" 医学知识检索API服务测试")
    print("=" * 80)
    print(f"API地址: {API_BASE_URL}\n")
    
    try:
        # 测试1: 健康检查
        test_health_check()
        
        # 测试2: 单个查询
        test_single_query()
        
        # 测试3: 多个查询
        test_multiple_queries()
        
        # 测试4: 学科过滤
        test_subject_filtering()
        
        # 测试5: 批量查询
        test_batch_queries()
        
        print("\n" + "=" * 80)
        print(" 所有测试完成")
        print("=" * 80 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n✗ 无法连接到API服务")
        print("请确保API服务已启动: python api_server.py\n")
    except Exception as e:
        print(f"\n✗ 测试失败: {e}\n")


if __name__ == "__main__":
    main()

