"""
API服务测试脚本

测试医学知识检索API服务
"""

import requests
import json
import time
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


def test_fast_endpoint():
    """测试快速端点"""
    print("=" * 80)
    print("测试快速端点 /search_fast")
    print("=" * 80)
    
    payload = {
        "queries": [
            {
                "query": "What is pneumonia and how is it treated?",
                "subject": None
            }
        ]
    }
    
    print(f"查询: {payload['queries'][0]['query']}")
    print("使用 /search_fast 端点（stuff模式，limit=3）\n")
    
    start_time = time.time()
    response = requests.post(f"{API_BASE_URL}/search_fast", json=payload)
    elapsed_time = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ 响应时间: {elapsed_time:.2f}秒")
        print(f"\n答案:\n{result['results'][0]['summary']}\n")
    else:
        print(f"✗ 错误: {response.status_code} - {response.text}\n")


def test_performance_comparison():
    """测试性能对比：/search vs /search_fast"""
    print("=" * 80)
    print("性能对比测试：/search vs /search_fast")
    print("=" * 80)
    
    test_queries = [
        {"query": "What is coronary artery disease?", "subject": None},
        {"query": "Explain the mechanism of aspirin", "subject": "Pharmacology"},
        {"query": "What is the anatomy of the liver?", "subject": "Anatomy"}
    ]
    
    payload = {"queries": test_queries}
    
    # 测试 /search (map_reduce模式)
    print("\n🔄 测试 /search 端点 (map_reduce模式, limit=10)...")
    start_time = time.time()
    response_search = requests.post(f"{API_BASE_URL}/search", json=payload, timeout=300)
    time_search = time.time() - start_time
    
    if response_search.status_code == 200:
        print(f"✓ /search 完成: {time_search:.2f}秒")
    else:
        print(f"✗ /search 失败: {response_search.status_code}")
    
    # 测试 /search_fast (stuff模式)
    print("\n🔄 测试 /search_fast 端点 (stuff模式, limit=3)...")
    start_time = time.time()
    response_fast = requests.post(f"{API_BASE_URL}/search_fast", json=payload, timeout=300)
    time_fast = time.time() - start_time
    
    if response_fast.status_code == 200:
        print(f"✓ /search_fast 完成: {time_fast:.2f}秒")
    else:
        print(f"✗ /search_fast 失败: {response_fast.status_code}")
    
    # 性能总结
    if response_search.status_code == 200 and response_fast.status_code == 200:
        print(f"\n📊 性能总结:")
        print(f"  - /search (深度分析):  {time_search:.2f}秒")
        print(f"  - /search_fast (快速): {time_fast:.2f}秒")
        print(f"  - 速度提升: {(time_search / time_fast):.2f}x")
        print(f"\n💡 建议:")
        print(f"  - 需要全面、深入分析时，使用 /search")
        print(f"  - 需要快速响应或高并发时，使用 /search_fast")


def test_concurrent_performance():
    """测试并发性能"""
    print("\n" + "=" * 80)
    print("并发性能测试")
    print("=" * 80)
    
    # 准备10个查询
    queries_10 = [
        {"query": "What is diabetes mellitus?", "subject": None},
        {"query": "Define hypertension", "subject": None},
        {"query": "What are beta blockers?", "subject": "Pharmacology"},
        {"query": "Explain cardiac anatomy", "subject": "Anatomy"},
        {"query": "What is pneumonia?", "subject": None},
        {"query": "Define sepsis", "subject": "InternalMed"},
        {"query": "What is a CT scan?", "subject": None},
        {"query": "Explain MRI imaging", "subject": None},
        {"query": "What are antibiotics?", "subject": "Pharmacology"},
        {"query": "Define inflammation", "subject": "Pathology"}
    ]
    
    payload = {"queries": queries_10}
    
    print(f"\n测试场景: 10个查询并发处理")
    print(f"使用 /search_fast 端点（推荐用于高并发）\n")
    
    print("🔄 发送请求...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/search_fast",
            json=payload,
            timeout=600  # 10分钟超时
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            results = response.json()['results']
            print(f"✓ 请求完成！")
            print(f"\n📊 性能指标:")
            print(f"  - 查询数量: {len(results)}")
            print(f"  - 总耗时: {elapsed_time:.2f}秒")
            print(f"  - 平均每个查询: {elapsed_time / len(results):.2f}秒")
            
            # 估算串行处理时间（假设每个查询5秒）
            estimated_serial_time = len(results) * 5
            print(f"  - 估算串行耗时: ~{estimated_serial_time}秒")
            print(f"  - 并发加速比: ~{estimated_serial_time / elapsed_time:.1f}x")
            
            print(f"\n✨ 成功返回结果:")
            for i, result in enumerate(results[:3], 1):  # 只显示前3个
                print(f"  {i}. {result['query']}")
                print(f"     答案: {result['summary'][:80]}...")
            if len(results) > 3:
                print(f"  ... 还有 {len(results) - 3} 个结果")
        else:
            print(f"✗ 请求失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            
    except requests.exceptions.Timeout:
        print(f"✗ 请求超时（>600秒）")
    except Exception as e:
        print(f"✗ 请求异常: {e}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print(" 医学知识检索API服务测试")
    print("=" * 80)
    print(f"API地址: {API_BASE_URL}\n")
    
    # 测试列表
    tests = [
        ("健康检查", test_health_check),
        ("单个查询", test_single_query),
        ("多个查询", test_multiple_queries),
        ("学科过滤", test_subject_filtering),
        ("批量查询", test_batch_queries),
        ("快速端点测试", test_fast_endpoint),
        ("性能对比测试", test_performance_comparison),
        ("并发性能测试", test_concurrent_performance),
    ]
    
    print("可用的测试:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"  {i}. {name}")
    print(f"  0. 运行所有测试")
    print(f"  9. 仅运行性能测试（6-8）")
    
    try:
        choice = input("\n选择要运行的测试 (0-9): ").strip()
        
        if choice == "0":
            # 运行所有测试
            for name, test_func in tests:
                try:
                    test_func()
                except KeyboardInterrupt:
                    print("\n\n⚠️ 测试被用户中断")
                    break
                except Exception as e:
                    print(f"\n✗ 测试失败: {e}\n")
                    continue
        elif choice == "9":
            # 仅运行性能测试
            print("\n运行性能测试套件...\n")
            for name, test_func in tests[5:]:  # 第6-8个测试
                try:
                    test_func()
                except Exception as e:
                    print(f"\n✗ 测试失败: {e}\n")
                    continue
        elif choice.isdigit() and 1 <= int(choice) <= len(tests):
            # 运行选定的测试
            name, test_func = tests[int(choice) - 1]
            test_func()
        else:
            print("✗ 无效的选择")
            return
        
        print("\n" + "=" * 80)
        print(" 测试完成")
        print("=" * 80 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n✗ 无法连接到API服务")
        print("请确保API服务已启动: python api_server.py\n")
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断\n")
    except Exception as e:
        print(f"\n✗ 测试失败: {e}\n")


if __name__ == "__main__":
    main()

