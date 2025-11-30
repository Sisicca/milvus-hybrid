"""
LLM总结功能测试脚本

演示如何使用search_results_summary_by_llm方法对医疗文档检索结果进行智能总结
"""

import sys
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_retriever import HybridRetriever


def print_section(title: str):
    """打印分隔线和标题"""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100 + "\n")


def test_stuff_mode(retriever: HybridRetriever):
    """测试stuff模式（一次性总结）"""
    print_section("Test 1: Stuff Mode - Medical QA")
    
    # 医疗相关查询（英文）
    queries = [
        "What is hypertension? How is it diagnosed and treated?",
        "What are the causes and symptoms of diabetes mellitus?",
        "Explain the mechanism of action and side effects of aspirin",
        "What is the emergency management of myocardial infarction?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 100}")
        print(f"Query {i}: {query}")
        print('─' * 100)
        
        try:
            # 使用stuff模式进行总结
            summary = retriever.search_results_summary_by_llm(
                query=query,
                search_type="hybrid",
                limit=5,
                summary_mode="stuff",
                model="gpt-4o-mini"
            )
            
            print("\n【LLM Summary Result】\n")
            print(summary)
            print("\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


def test_map_reduce_mode(retriever: HybridRetriever):
    """测试map_reduce模式（分步总结后汇总）"""
    print_section("Test 2: Map-Reduce Mode - Complex Medical Questions")
    
    # 复杂的医疗查询，需要综合多个文档
    query = "Explain in detail the pathogenesis, clinical manifestations, diagnostic methods, and treatment options for coronary heart disease"
    
    print(f"Query: {query}\n")
    print("Processing multiple documents using Map-Reduce mode...\n")
    
    try:
        summary = retriever.search_results_summary_by_llm(
            query=query,
            search_type="hybrid",
            limit=8,  # 检索更多文档
            summary_mode="map_reduce",
            model="gpt-4o-mini"
        )
        
        print("\n【LLM Summary Result】\n")
        print(summary)
        print("\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def test_subject_filtering(retriever: HybridRetriever):
    """测试学科过滤的LLM总结"""
    print_section("Test 3: Subject Filtering - Pharmacology Focus")
    
    query = "What are the pharmacological actions and clinical applications of beta-blockers?"
    subject = "Pharmacology"
    
    print(f"Query: {query}")
    print(f"Subject: {subject}\n")
    
    try:
        summary = retriever.search_results_summary_by_llm(
            query=query,
            search_type="hybrid",
            limit=5,
            subject=subject,
            summary_mode="stuff",
            model="gpt-4o-mini"
        )
        
        print("\n【LLM Summary Result】\n")
        print(summary)
        print("\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def test_custom_prompt(retriever: HybridRetriever):
    """测试自定义提示词"""
    print_section("Test 4: Custom Prompt - Targeted Information Extraction")
    
    query = "What are the medication guidelines for treating hypertension?"
    
    # 自定义提示词，专注于用药指导
    custom_prompt = """You are a clinical pharmacist. Please extract key medication-related information from the following medical documents.

Patient Question:
{query}

Retrieved Medical Documents:
{formatted_results}

Please focus on providing the following information:
1. **Recommended Medications**: List commonly used therapeutic drugs
2. **Dosage and Administration**: Specific dosage and administration methods for each drug
3. **Precautions**: Contraindications, drug interactions
4. **Side Effects**: Common and serious adverse reactions
5. **Patient Education**: Medication adherence recommendations

Please present in clear Markdown format for easy patient understanding."""
    
    print(f"Query: {query}\n")
    print("Using custom prompt (focused on medication guidance)...\n")
    
    try:
        summary = retriever.search_results_summary_by_llm(
            query=query,
            search_type="hybrid",
            limit=5,
            summary_mode="stuff",
            model="gpt-4o-mini",
            custom_prompt=custom_prompt
        )
        
        print("\n【LLM Summary Result】\n")
        print(summary)
        print("\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def test_different_search_types(retriever: HybridRetriever):
    """测试不同检索类型的LLM总结"""
    print_section("Test 5: Comparison of Different Search Types")
    
    query = "What is computed tomography (CT) and how does it work?"
    search_types = ["sparse", "dense", "hybrid"]
    
    print(f"Query: {query}\n")
    
    for search_type in search_types:
        print(f"\n{'─' * 100}")
        print(f"Search Type: {search_type.upper()}")
        print('─' * 100)
        
        try:
            summary = retriever.search_results_summary_by_llm(
                query=query,
                search_type=search_type,
                limit=3,
                summary_mode="stuff",
                model="gpt-4o-mini"
            )
            
            print("\n【LLM Summary Result】\n")
            print(summary)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


def test_empty_results(retriever: HybridRetriever):
    """测试空结果的处理"""
    print_section("Test 6: Empty Results Handling")
    
    # 使用一个不太可能找到结果的查询
    query = "xyzabc123nonexistentmedicalterm"
    
    print(f"Query: {query}\n")
    print("Expected: Should return a friendly message\n")
    
    try:
        summary = retriever.search_results_summary_by_llm(
            query=query,
            search_type="hybrid",
            limit=5,
            summary_mode="stuff",
            model="gpt-4o-mini"
        )
        
        print("\n【Return Result】\n")
        print(summary)
        print("\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def test_performance_comparison(retriever: HybridRetriever):
    """测试性能对比：stuff vs map_reduce"""
    print_section("Test 7: Performance Comparison - Stuff vs Map-Reduce")
    
    import time
    
    query = "What are the diagnosis and treatment of pneumonia?"
    limit = 6
    
    print(f"Query: {query}")
    print(f"Number of documents to retrieve: {limit}\n")
    
    # 测试stuff模式
    print("🔄 Testing Stuff mode...")
    start_time = time.time()
    try:
        stuff_summary = retriever.search_results_summary_by_llm(
            query=query,
            search_type="hybrid",
            limit=limit,
            summary_mode="stuff",
            model="gpt-4o-mini"
        )
        stuff_time = time.time() - start_time
        print(f"✅ Stuff mode completed, time elapsed: {stuff_time:.2f}s")
    except Exception as e:
        print(f"❌ Stuff mode failed: {e}")
        stuff_time = None
    
    print()
    
    # 测试map_reduce模式
    print("🔄 Testing Map-Reduce mode...")
    start_time = time.time()
    try:
        map_reduce_summary = retriever.search_results_summary_by_llm(
            query=query,
            search_type="hybrid",
            limit=limit,
            summary_mode="map_reduce",
            model="gpt-4o-mini"
        )
        map_reduce_time = time.time() - start_time
        print(f"✅ Map-Reduce mode completed, time elapsed: {map_reduce_time:.2f}s")
    except Exception as e:
        print(f"❌ Map-Reduce mode failed: {e}")
        map_reduce_time = None
    
    # 性能总结
    if stuff_time and map_reduce_time:
        print(f"\n📊 Performance Summary:")
        print(f"  - Stuff mode: {stuff_time:.2f}s")
        print(f"  - Map-Reduce mode: {map_reduce_time:.2f}s")
        print(f"  - Speed difference: Map-Reduce is {'faster' if map_reduce_time < stuff_time else 'slower'} by {abs(stuff_time - map_reduce_time):.2f}s")
        print(f"\n💡 Recommendations:")
        print(f"  - For fewer documents (<5) or shorter total content, use Stuff mode")
        print(f"  - For more documents (>5) or very long individual documents, use Map-Reduce mode")


def main():
    """主函数"""
    
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set")
        print("Please set the environment variable or pass the API key during initialization")
        print("\nUsage:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("  export OPENAI_BASE_URL='your-base-url'  # Optional\n")
    
    print("=" * 100)
    print("  LLM Summary Function Test Suite")
    print("  Testing Hybrid Retrieval + LLM Intelligent Summary for Medical QA")
    print("=" * 100)
    
    # 初始化检索器
    print("\nInitializing hybrid retriever...")
    try:
        retriever = HybridRetriever(
            uri="./milvus_db_hub/med_corpus.db",
            collection_name="med_corpus",
            model_path="./model-hub/Qwen3-Embedding-0.6B"
        )
        print("✅ Initialization successful!\n")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # 运行测试
    tests = [
        ("Basic Test - Stuff Mode", test_stuff_mode),
        ("Advanced Test - Map-Reduce Mode", test_map_reduce_mode),
        ("Subject Filtering Test", test_subject_filtering),
        ("Custom Prompt Test", test_custom_prompt),
        ("Search Type Comparison Test", test_different_search_types),
        ("Exception Handling Test - Empty Results", test_empty_results),
        ("Performance Comparison Test", test_performance_comparison),
    ]
    
    # 让用户选择要运行的测试
    print("\nAvailable tests:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"  {i}. {name}")
    print(f"  0. Run all tests")
    
    try:
        choice = input("\nSelect a test to run (0-7): ").strip()
        
        if choice == "0":
            # 运行所有测试
            for name, test_func in tests:
                try:
                    test_func(retriever)
                except KeyboardInterrupt:
                    print("\n\n⚠️  Test interrupted by user")
                    break
                except Exception as e:
                    print(f"\n❌ Test failed: {e}\n")
                    continue
        elif choice.isdigit() and 1 <= int(choice) <= len(tests):
            # 运行选定的测试
            name, test_func = tests[int(choice) - 1]
            test_func(retriever)
        else:
            print("❌ Invalid selection")
            return
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
    finally:
        # 关闭连接
        print("\n" + "=" * 100)
        print("Cleaning up resources...")
        retriever.close()
        print("✅ Tests completed!")
        print("=" * 100)


if __name__ == "__main__":
    main()

