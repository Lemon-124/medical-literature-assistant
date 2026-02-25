"""医学文献综述助手配置文件"""

# 应用基本信息
APP_CONFIG = {
    "title": "医学文献写作助手",
    "icon": "📚",
    "layout": "wide"
}

# 文档处理参数
PROCESSING_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_workers": 4,
    "timeout": 300,  # 5分钟超时
    "max_content_length": 15000
}

# 向量存储配置
VECTOR_STORE_CONFIG = {
    "persist_directory": "./chroma_db"
}

# 参考文献标识符（中英文）
REFERENCE_INDICATORS = [
    r'\n\s*参考文献\s*\n',
    r'\n\s*References\s*\n',
    r'\n\s*Bibliography\s*\n',
    r'\n\s*Works Cited\s*\n',
    r'\n\s*Literature Cited\s*\n',
    r'^\s*参考文献',
    r'^\s*References',
    r'^\s*R E F E R E N C E S',
    r'\n\d+\.\s+[A-Z]',  # 参考文献条目格式
    r'\n\[?\d+\]?\.\s+[A-Z]',  # 带编号的参考文献
]

# 分析类型选项
ANALYSIS_TYPES = {
    "研究方法汇总": "请汇总所有文献使用的研究方法，包括实验设计、样本量、统计方法等",
    "主要结论对比": "请对比各篇文献的主要结论，找出共识和分歧",
    "局限性总结": "请总结各篇文献提到的研究局限性和不足",
    "推荐引用句子": "请从各篇文献中提取最值得引用的关键句子"
}

# 表格列定义
TABLE_COLUMNS = [
    "文献名", "研究目的", "实验方法", "实验结果和结论", 
    "方法优缺点", "关键引用", "提取时间"
]
