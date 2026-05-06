#!/usr/bin/env python3
# 提取Word文档中的所有MathML公式
import xml.etree.ElementTree as ET
import re

xml_file = "C:/Users/罗/WorkBuddy/2026-05-04-task-5/docx_extracted_new/word/document.xml"

# 读取XML文件
with open(xml_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 使用正则表达式提取所有MathML块
# MathML在Word中通常包裹在 <m:oMath> 或 <m:oMathPara> 标签中
mathml_pattern = r'<m:oMathPara[^>]*>.*?</m:oMathPara>|<m:oMath[^>]*>.*?</m:oMath>'
mathml_blocks = re.findall(mathml_pattern, content, re.DOTALL)

print(f"找到 {len(mathml_blocks)} 个MathML公式块\n")
print("=" * 80)

for i, block in enumerate(mathml_blocks, 1):
    print(f"\n公式 {i}:")
    print("-" * 80)
    # 简化输出，只显示关键部分
    # 提取文本内容
    text_parts = re.findall(r'<m:t[^>]*>(.*?)</m:t>', block, re.DOTALL)
    if text_parts:
        formula_text = ' '.join(text_parts)
        print(f"公式内容: {formula_text}")
    print("-" * 80)

# 也尝试提取公式编号引用
print("\n\n" + "=" * 80)
print("搜索公式编号引用...")
formula_refs = re.findall(r'公式\s*[\(（]?\s*\d+\s*[\)）]?', content)
if formula_refs:
    print(f"找到 {len(set(formula_refs))} 个不同的公式引用:")
    for ref in sorted(set(formula_refs)):
        print(f"  - {ref}")
else:
    print("未找到公式编号引用")
