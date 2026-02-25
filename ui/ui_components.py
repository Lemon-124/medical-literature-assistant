"""Streamlit UI组件"""

import os
import sys
import streamlit as st
from typing import List, Dict, Any

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import MODEL_OPTIONS, ANALYSIS_TYPES

def render_sidebar():
    """渲染侧边栏配置"""
    with st.sidebar:
        st.header("⚙️ 配置")
        
        # API Key输入
        api_key = st.text_input("请输入DashScope API Key", type="password")
        
        st.markdown("---")

        st.markdown("""
        <small>
        从阿里云百炼平台获取<strong>API Key</strong>，用于调用DashScope API<br>
        网址：<a href="https://bailian.console.aliyun.com/" target="_blank">https://bailian.console.aliyun.com/</a><br>
        需要先登录，再点击<b>API参考</b>，下方有获取<strong>API Key</strong>教程<br>
        获取<strong>API Key</strong>后，请妥善保管，不要泄露给他人<br>
        <strong>API Key</strong>仅用于调用DashScope API，不会用于其他用途，也不会泄露给他人
        </small>
        """, unsafe_allow_html=True)
        
        return api_key

def render_file_upload():
    """渲染文件上传区域"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "上传中英文医学文献（PDF格式）",
            type=['pdf'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        handle_files = st.button("开始处理文献", type="primary")
    
    with col2:
        st.markdown("### 📋 使用说明")
        st.info("""
        1. 在左侧输入DashScope API Key
        2. 上传中英文医学文献（PDF格式）
        3. 点击"开始处理文献"按钮提取信息
        4. 查看结构化结果表格
        5. 可向助手提问获取多篇文献的综合分析结果
        """)
    
    return uploaded_files, handle_files

def render_results_table(assistant):
    """渲染结果表格"""
    st.markdown("---")
    st.header("📑 提取结果汇总")
    
    # 显示汇总表格
    df = assistant.generate_summary_table()
    st.dataframe(df, use_container_width=True)
    
    # 下载按钮
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "📥 下载汇总表格(CSV)",
        csv,
        "文献综述汇总.csv",
        "text/csv"
    )
    
    return df

def render_qa_interface(assistant):
    """渲染问答界面"""
    st.markdown("---")
    st.header("💬 向助手提问")
    
    # 问题输入
    question = st.text_input("请输入你的问题")
    
    if question:
        with st.spinner("正在思考中..."):
            answer = assistant.ask_question(question)
        st.markdown("### 回答")
        st.write(answer)

def render_analysis_interface(assistant):
    """渲染综合分析界面"""
    st.markdown("---")
    st.header("📊 多篇文献综合分析")
    
    analysis_type = st.selectbox(
        "选择分析类型",
        list(ANALYSIS_TYPES.keys())
    )
    
    if st.button("生成分析"):
        with st.spinner("正在生成分析..."):
            question = ANALYSIS_TYPES[analysis_type]
            answer = assistant.ask_question(question)
            st.markdown(answer)