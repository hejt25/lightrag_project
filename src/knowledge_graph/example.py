"""
Knowledge Graph Example
========================
知识图谱模块使用示例
"""

from src.knowledge_graph import (
    KnowledgeGraph,
    Entity,
    Relation,
    EntityExtractor,
    RelationExtractor,
    GraphBuilder,
    GraphRetriever,
    HybridGraphRetriever,
    GraphEnhancedRAG
)


def example_basic_graph():
    """基本图谱操作"""
    print("=" * 50)
    print("1. Basic Graph Operations")
    print("=" * 50)

    # 创建图谱
    graph = KnowledgeGraph("InsuranceGraph")

    # 添加实体
    entity1 = Entity(
        id="e1",
        name="平安福重疾险",
        entity_type="PRODUCT",
        description="平安保险公司的重疾险产品",
        frequency=5
    )
    entity2 = Entity(
        id="e2",
        name="平安保险公司",
        entity_type="COMPANY",
        description="中国平安保险集团",
        frequency=10
    )
    entity3 = Entity(
        id="e3",
        name="重疾险",
        entity_type="PRODUCT",
        description="重大疾病保险",
        frequency=20
    )

    graph.add_entity(entity1)
    graph.add_entity(entity2)
    graph.add_entity(entity3)

    # 添加关系
    relation1 = Relation(
        source_id="e1",
        target_id="e2",
        relation_type="PRODUCED_BY",
        description="平安福重疾险由平安保险公司承保"
    )
    relation2 = Relation(
        source_id="e1",
        target_id="e3",
        relation_type="IS_A",
        description="平安福重疾险是一种重疾险"
    )

    graph.add_relation(relation1)
    graph.add_relation(relation2)

    # 获取统计
    stats = graph.get_stats()
    print(f"Graph stats: {stats}")

    # 获取邻居
    neighbors = graph.get_neighbors("e1")
    print(f"Neighbors of 平安福重疾险: {[(n[0].name, n[1]) for n in neighbors]}")

    print()


def example_entity_extraction():
    """实体抽取"""
    print("=" * 50)
    print("2. Entity Extraction")
    print("=" * 50)

    extractor = EntityExtractor()

    text = """平安福重疾险是平安保险公司推出的一款重大疾病保险产品。
该产品提供100种重疾保障，保额最高可达50万。"""

    entities = extractor.extract(text, source="doc1")

    print(f"Text: {text}")
    print(f"Extracted entities:")
    for entity in entities:
        print(f"  - {entity.name} ({entity.entity_type})")

    print()


def example_graph_builder():
    """从文档构建图谱"""
    print("=" * 50)
    print("3. Build Graph from Documents")
    print("=" * 50)

    documents = [
        {
            "id": "doc1",
            "content": """平安福重疾险是平安保险公司推出的重大疾病保险。
保障100种重疾和50种轻症，等待期90天。
保费每年5000元，保额30万起。"""
        },
        {
            "id": "doc2",
            "content": """太平洋金佑人生是太平洋保险公司的分红型重疾险。
具有保额分红功能，保障终身。
支持保单贷款和年金转换。"""
        },
        {
            "id": "doc3",
            "content": """医疗险是报销型保险产品，主要用于医疗费用报销。
通常包含住院医疗、门诊医疗、特殊门诊等保障。
百万医疗险近年来非常受欢迎。"""
        }
    ]

    builder = GraphBuilder()
    graph = builder.build_from_documents(documents)

    stats = graph.get_stats()
    print(f"Built graph with {stats['entity_count']} entities and {stats['relation_count']} relations")
    print(f"Type distribution: {stats['type_distribution']}")

    print()


def example_graph_retriever():
    """图谱检索"""
    print("=" * 50)
    print("4. Graph Retrieval")
    print("=" * 50)

    # 创建图谱
    graph = KnowledgeGraph("TestGraph")

    # 添加测试实体
    products = [
        ("平安福", "PRODUCT", "平安福重疾险"),
        ("平安保险公司", "COMPANY", "平安"),
        ("重疾险", "PRODUCT", "重大疾病保险"),
        ("医疗险", "PRODUCT", "医疗保险"),
        ("保额", "CONCEPT", "保险金额"),
        ("保费", "CONCEPT", "保险费用"),
    ]

    for i, (name, etype, desc) in enumerate(products):
        graph.add_entity(Entity(
            id=f"e{i}",
            name=name,
            entity_type=etype,
            description=desc
        ))

    # 添加关系
    relations = [
        ("e0", "e1", "PRODUCED_BY"),
        ("e0", "e2", "IS_A"),
        ("e3", "e0", "RELATED_TO"),
    ]

    for s, t, rtype in relations:
        graph.add_relation(Relation(s, t, rtype))

    # 创建检索器
    retriever = GraphRetriever(graph)

    # 检索
    query = "平安福"
    results = retriever.retrieve(query, top_k=3)

    print(f"Query: {query}")
    print(f"Results:")
    for r in results:
        print(f"  - {r.entity_name} ({r.entity_type}) score: {r.score:.2f}")
        print(f"    Neighbors: {[n['name'] for n in r.neighbors]}")

    print()


def example_hybrid_retriever():
    """混合检索"""
    print("=" * 50)
    print("5. Hybrid Retrieval (Graph + Vector)")
    print("=" * 50)

    # 创建图谱
    graph = KnowledgeGraph("TestGraph")

    # 添加实体
    products = [
        ("平安福重疾险", "PRODUCT", "平安保险公司推出的重疾险"),
        ("太平洋金佑人生", "PRODUCT", "太平洋保险的分红型重疾险"),
        ("医疗险", "PRODUCT", "报销医疗费用的保险"),
    ]

    for i, (name, etype, desc) in enumerate(products):
        graph.add_entity(Entity(
            id=f"e{i}",
            name=name,
            entity_type=etype,
            description=desc
        ))

    # 创建混合检索器（无向量存储，仅演示图谱检索）
    hybrid = HybridGraphRetriever(graph=graph)

    # 检索
    query = "重疾险哪个好"
    results = hybrid.retrieve(query, top_k=3, mode="graph")

    print(f"Query: {query}")
    print(f"Mode: {results['mode']}")
    print(f"Results:")
    for r in results["graph_results"]:
        print(f"  - {r['name']} ({r['type']}) score: {r['score']:.2f}")

    print()


def example_graph_enhanced_rag():
    """图谱增强的RAG"""
    print("=" * 50)
    print("6. Graph Enhanced RAG")
    print("=" * 50)

    # 准备文档
    documents = [
        {
            "id": "doc1",
            "content": """
            平安福重疾险是平安保险公司于2015年推出的重大疾病保险产品。
            该产品保障80种重疾和20种轻症，等待期90天。
            保额分为10万、20万、30万三档，年缴保费约4000-12000元。
            支持保单贷款，贷款额度为现金价值的80%。
            """
        },
        {
            "id": "doc2",
            "content": """
            太平洋金佑人生是太平洋保险公司推出的分红型终身重疾险。
            保障100种重疾和50种轻症，等待期180天。
            具有保额分红功能，随着时间推移保额会增长。
            支持年金转换，在60岁后可选择转换为年金领取。
            """
        },
        {
            "id": "doc3",
            "content": """
            医疗险是报销型健康险，对住院医疗费用进行报销。
            百万医疗险保额通常为100-600万，免赔额1-2万。
            保障范围包括住院医疗、特殊门诊、门诊手术、住院前后门急诊。
            部分产品还包含就医绿通、专家二诊等增值服务。
            """
        }
    ]

    # 创建图谱增强的RAG
    rag = GraphEnhancedRAG()

    # 构建图谱
    graph = rag.build_graph(documents)
    stats = graph.get_stats()
    print(f"Built graph: {stats['entity_count']} entities, {stats['relation_count']} relations")

    # 检索
    queries = [
        "平安福重疾险的保额是多少？",
        "太平洋金佑人生有什么特点？",
        "医疗险和重疾险有什么区别？"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = rag.retrieve(query, top_k=3, mode="graph")

        print(f"Found {results['reasoning_info']['total_entities']} entities")

        if results["reasoning_info"]["reasoning_path"]:
            print("Reasoning path:")
            for step in results["reasoning_info"]["reasoning_path"][:3]:
                print(f"  - {step['entity']} ({step['type']})")

    print()


def example_entity_search():
    """实体搜索"""
    print("=" * 50)
    print("7. Entity Search Methods")
    print("=" * 50)

    graph = KnowledgeGraph("TestGraph")

    # 添加实体
    products = [
        ("平安福", "PRODUCT", "平安福重疾险"),
        ("平安福Pro", "PRODUCT", "平安福升级版"),
        ("太平洋金佑人生", "PRODUCT", "太平洋金佑人生"),
        ("金佑人生Pro", "PRODUCT", "金佑人生升级版"),
    ]

    for i, (name, etype, desc) in enumerate(products):
        graph.add_entity(Entity(id=f"e{i}", name=name, entity_type=etype, description=desc))

    # 按类型搜索
    print("Search by type PRODUCT:")
    results = graph.search_by_type("PRODUCT")
    print([e.name for e in results])

    # 模糊搜索
    print("\nSearch by name contains '金佑':")
    results = graph.search_by_name_contains("金佑")
    print([e.name for e in results])

    print("\nSearch by name contains '平安':")
    results = graph.search_by_name_contains("平安")
    print([e.name for e in results])

    print()


def example_export_import():
    """导出导入图谱"""
    print("=" * 50)
    print("8. Export/Import Graph")
    print("=" * 50)

    # 创建图谱
    graph = KnowledgeGraph("TestGraph")
    graph.add_entity(Entity(id="e1", name="测试产品", entity_type="PRODUCT"))
    graph.add_entity(Entity(id="e2", name="测试公司", entity_type="COMPANY"))

    # 导出
    import json
    graph_dict = graph.to_dict()
    print(f"Exported graph: {json.dumps(graph_dict, ensure_ascii=False, indent=2)[:200]}...")

    # 导入
    loaded_graph = KnowledgeGraph.from_dict(graph_dict)
    print(f"Loaded graph has {len(loaded_graph.entities)} entities")

    print()


if __name__ == "__main__":
    print("=" * 50)
    print("Knowledge Graph Module Examples")
    print("=" * 50)
    print()

    example_basic_graph()
    example_entity_extraction()
    example_graph_builder()
    example_graph_retriever()
    example_hybrid_retriever()
    example_graph_enhanced_rag()
    example_entity_search()
    example_export_import()

    print("=" * 50)
    print("All examples completed!")
    print("=" * 50)
