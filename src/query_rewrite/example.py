"""
Insurance Query Rewrite Example
================================
保险问答Query改写模块使用示例
"""

from src import (
    InsuranceQueryRewriter,
    RewriteResult,
    UserProfile,
    ConversationHistoryManager,
    UserBehaviorTracker
)


def example_basic_rewrite():
    """基本改写示例"""
    rewriter = InsuranceQueryRewriter()

    # 原始查询
    query = "重疾险哪个好"

    # 改写
    result = rewriter.rewrite(
        query=query,
        user_id="user_001",
        session_id="session_001"
    )

    print(f"Original: {result.original_query}")
    print(f"Rewritten: {result.rewritten_query}")
    print(f"Type: {result.rewrite_type}")
    print(f"Intent: {result.context_used.get('intent')}")
    print()


def example_with_user_profile():
    """带用户画像的改写示例"""
    rewriter = InsuranceQueryRewriter()

    # 创建用户画像
    profile = UserProfile(
        user_id="user_002",
        age=35,
        gender="male",
        income_level="medium",
        marital_status="married",
        has_children=True,
        children_ages=[5, 8],
        insurance_history=["医疗险"],
        risk_preference="moderate",
        budget_range="standard"
    )

    # 改写查询
    query = "给我推荐个保险"
    result = rewriter.rewrite(
        query=query,
        user_id="user_002",
        session_id="session_002",
        user_profile=profile
    )

    print(f"Original: {result.original_query}")
    print(f"Rewritten: {result.rewritten_query}")
    print(f"Suggestions: {result.suggestions}")
    print()


def example_with_conversation_history():
    """带会话历史的改写示例（指代消解）"""
    rewriter = InsuranceQueryRewriter()

    # 添加对话历史
    rewriter.add_conversation_message(
        session_id="session_003",
        user_id="user_003",
        role="user",
        content="我想了解一下重疾险"
    )
    rewriter.add_conversation_message(
        session_id="session_003",
        user_id="user_003",
        role="assistant",
        content="好的，重疾险有很多种，请问您有什么具体需求？"
    )
    rewriter.add_conversation_message(
        session_id="session_003",
        user_id="user_003",
        role="user",
        content="那它多少钱？"
    )

    # 改写"它多少钱" - 应该消解为"重疾险多少钱"
    result = rewriter.rewrite(
        query="它多少钱",
        user_id="user_003",
        session_id="session_003"
    )

    print(f"Original: {result.original_query}")
    print(f"Rewritten: {result.rewritten_query}")
    print()


def example_with_user_behavior():
    """带用户行为的改写示例"""
    rewriter = InsuranceQueryRewriter()

    # 追踪用户行为
    rewriter.add_user_behavior(
        user_id="user_004",
        behavior_type="view_product",
        data={"product_name": "平安福重疾险", "category": "重疾险"}
    )
    rewriter.add_user_behavior(
        user_id="user_004",
        behavior_type="click_product",
        data={"product_name": "平安福重疾险"}
    )
    rewriter.add_user_behavior(
        user_id="user_004",
        behavior_type="search",
        data={"keyword": "重疾险推荐"}
    )

    # 改写查询
    query = "有什么推荐的吗？"
    result = rewriter.rewrite(
        query=query,
        user_id="user_004",
        session_id="session_004"
    )

    print(f"Original: {result.original_query}")
    print(f"Rewritten: {result.rewritten_query}")
    print(f"Behaviors used: {result.context_used.get('behaviors_count')}")
    print()


def example_term_expansion():
    """术语扩展示例"""
    rewriter = InsuranceQueryRewriter()

    queries = [
        "重疾险多少钱",
        "医疗险值得买吗",
        "车险哪个好",
        "理赔需要什么材料"
    ]

    for query in queries:
        result = rewriter.rewrite(
            query=query,
            user_id="user_005",
            session_id="session_005"
        )
        print(f"Original: {query}")
        print(f"Rewritten: {result.rewritten_query}")
        print()


def example_full_scenario():
    """完整场景示例"""
    rewriter = InsuranceQueryRewriter()

    # 1. 初始化用户画像
    profile = UserProfile(
        user_id="user_006",
        age=30,
        gender="female",
        income_level="high",
        marital_status="married",
        has_children=False,
        insurance_history=["意外险"],
        risk_preference="conservative",
        budget_range="premium"
    )

    # 2. 模拟用户浏览行为
    rewriter.add_user_behavior("user_006", "view_product",
                                {"product_name": "尊享e生医疗险"})
    rewriter.add_user_behavior("user_006", "search",
                                {"keyword": "高端医疗险推荐"})

    # 3. 开始对话
    session_id = "session_006"

    # 第一轮
    rewriter.add_conversation_message(session_id, "user_006", "user",
                                       "我想买保险")
    result = rewriter.rewrite("我想买保险", "user_006", session_id, profile)
    print(f"Q: 我想买保险")
    print(f"A: 已为您推荐合适的保险方案")

    # 第二轮 - 指代消解
    rewriter.add_conversation_message(session_id, "user_006", "assistant",
                                       "请问您想了解哪种保险？")
    rewriter.add_conversation_message(session_id, "user_006", "user",
                                       "那种高端的怎么样？")

    result = rewriter.rewrite("那种高端的怎么样？", "user_006", session_id, profile)
    print(f"Q: 那种高端的怎么样？")
    print(f"Rewritten: {result.rewritten_query}")
    print(f"Intent: {result.context_used.get('intent')}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("Insurance Query Rewrite Examples")
    print("=" * 50)
    print()

    print("1. Basic Rewrite:")
    example_basic_rewrite()

    print("2. With User Profile:")
    example_with_user_profile()

    print("3. With Conversation History (Coreference Resolution):")
    example_with_conversation_history()

    print("4. With User Behavior:")
    example_with_user_behavior()

    print("5. Term Expansion:")
    example_term_expansion()

    print("6. Full Scenario:")
    example_full_scenario()
