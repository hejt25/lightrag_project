"""
Prompt Templates Example
========================
专业Prompt模板系统使用示例
"""

from src.prompt_templates import (
    # Templates
    InsuranceProductPrompt,
    InsuranceClaimPrompt,
    InsuranceComparisonPrompt,
    InsuranceRecommendationPrompt,
    GeneralRAGPrompt,
    ConversationalPrompt,
    # Builder & Manager
    PromptBuilder,
    PromptManager,
    PromptComponent,
    ComponentType
)


def example_basic_template():
    """基本模板使用"""
    print("=" * 50)
    print("1. Basic Template Usage")
    print("=" * 50)

    # 创建模板
    template = InsuranceProductPrompt()

    # 渲染Prompt
    context = """产品名称：平安福重疾险
保障范围：100种重疾+50种轻症
保额：50万
保费：6500元/年
缴费期：20年
等待期：90天"""

    question = "这款重疾险怎么样？"

    prompt = template.render(context, question)
    print(f"\nContext: {context[:50]}...")
    print(f"Question: {question}")
    print(f"\nGenerated Prompt:\n{prompt}")
    print()


def example_template_comparison():
    """模板对比"""
    print("=" * 50)
    print("2. Template Comparison")
    print("=" * 50)

    templates = {
        "产品咨询": InsuranceProductPrompt(),
        "理赔指导": InsuranceClaimPrompt(),
        "产品对比": InsuranceComparisonPrompt(),
        "产品推荐": InsuranceRecommendationPrompt()
    }

    context = "某重疾险产品..."
    question = "哪个更适合30岁的人？"

    for name, template in templates.items():
        print(f"\n{name}:")
        print(f"  Version: {template.version}")
        print(f"  Components: {len(template.components)}")
        print(f"  Component types: {[ct.value for ct in template.components.keys()]}")


def example_prompt_builder():
    """Prompt构建器使用"""
    print("\n" + "=" * 50)
    print("3. Prompt Builder")
    print("=" * 50)

    builder = PromptBuilder()

    # 注册模板
    builder.register_template(InsuranceProductPrompt(), set_default=True)

    # 构建Prompt
    context = "产品信息..."
    question = "重疾险哪个好？"

    result = builder.build(context, question)

    print(f"\nTemplate used: {result.template_used}")
    print(f"Template version: {result.template_version}")
    print(f"Context length: {result.context_length}")
    print(f"Question length: {result.question_length}")
    print(f"Estimated tokens: {builder.estimate_tokens(result.full_prompt)}")


def example_template_management():
    """模板管理"""
    print("\n" + "=" * 50)
    print("4. Template Management (Version Control)")
    print("=" * 50)

    manager = PromptManager("./prompt_versions")

    # 注册模板
    template = InsuranceProductPrompt()
    manager.register_template(template.to_dict())

    # 保存版本
    manager.save_version(
        template_name="InsuranceProductPrompt",
        version="1.0.0",
        template_dict=template.to_dict(),
        changelog="Initial version"
    )

    print(f"\nVersions saved: {manager.list_versions('InsuranceProductPrompt')}")
    print(f"Current version: {manager.get_current_version('InsuranceProductPrompt').version if manager.get_current_version('InsuranceProductPrompt') else 'None'}")

    # 检测变更
    changes = manager.detect_changes(template.to_dict())
    print(f"\nChanges detected: {len(changes)}")


def example_custom_component():
    """自定义组件"""
    print("\n" + "=" * 50)
    print("5. Custom Component")
    print("=" * 50)

    template = InsuranceProductPrompt()

    # 添加自定义约束
    custom_constraint = PromptComponent(
        name="custom_constraint",
        content="""额外要求：
1. 回答控制在200字以内
2. 使用bullet point格式
3. 适合发送给客户""",
        component_type=ComponentType.CONSTRAINT
    )
    template.add_component(custom_constraint)

    context = "产品信息..."
    question = "推荐一个保险"

    prompt = template.render(context, question)
    print(f"Custom constraint added!")
    print(f"Total components: {len(template.components)}")


def example_insurance_scenarios():
    """保险场景示例"""
    print("\n" + "=" * 50)
    print("6. Insurance Scenarios")
    print("=" * 50)

    # 场景1: 产品咨询
    print("\n[场景1: 产品咨询]")
    template = InsuranceProductPrompt()
    context = """平安e生保：
- 百万医疗险
- 保额200万
- 保费380元/年
- 免赔额1万"""
    question = "这款医疗险保障什么？"
    print(f"Template: {template.name}")
    print(f"Question: {question}")

    # 场景2: 理赔指导
    print("\n[场景2: 理赔指导]")
    template = InsuranceClaimPrompt()
    context = """理赔流程：
1. 报案
2. 准备材料
3. 提交申请
4. 审核
5. 打款"""
    question = "住院医疗险怎么理赔？"
    print(f"Template: {template.name}")
    print(f"Question: {question}")

    # 场景3: 产品对比
    print("\n[场景3: 产品对比]")
    template = InsuranceComparisonPrompt()
    context = """产品A：
- 重疾50种
- 保额30万
- 保费5000

产品B：
- 重疾100种
- 保额50万
- 保费8000"""
    question = "这两款重疾险哪个好？"
    print(f"Template: {template.name}")
    print(f"Question: {question}")

    # 场景4: 产品推荐
    print("\n[场景4: 产品推荐]")
    template = InsuranceRecommendationPrompt()
    context = """用户需求：30岁男性，预算5000元，需要重疾保障"""
    question = "有什么推荐的重疾险？"
    print(f"Template: {template.name}")
    print(f"Question: {question}")


def example_version_info():
    """版本信息"""
    print("\n" + "=" * 50)
    print("7. Version Info")
    print("=" * 50)

    template = InsuranceProductPrompt()
    version_info = template.get_version_info()

    print(f"\nTemplate: {version_info['template_name']}")
    print(f"Version: {version_info['template_version']}")
    print("\nComponent hashes:")
    for ct, h in version_info['components'].items():
        print(f"  {ct}: {h}")


def example_template_to_dict():
    """模板导出"""
    print("\n" + "=" * 50)
    print("8. Template Export")
    print("=" * 50)

    template = InsuranceProductPrompt()
    template_dict = template.to_dict()

    print(f"\nExported template: {template_dict['name']}")
    print(f"Version: {template_dict['version']}")
    print(f"Components: {list(template_dict['components'].keys())}")


if __name__ == "__main__":
    print("=" * 50)
    print("Professional Prompt Templates Demo")
    print("=" * 50)

    example_basic_template()
    example_template_comparison()
    example_prompt_builder()
    example_template_management()
    example_custom_component()
    example_insurance_scenarios()
    example_version_info()
    example_template_to_dict()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)
