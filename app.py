import os
import streamlit as st
from models.literature_assistant import LiteratureReviewAssistant
from ui.ui_components import (
    render_sidebar, render_file_upload, 
    render_results_table, render_qa_interface, render_analysis_interface
)
from utils.config import APP_CONFIG

# 设置页面配置
st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon=APP_CONFIG["icon"],
    layout=APP_CONFIG["layout"]
)

def initialize_session_state():
    """初始化会话状态"""
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False

def main():
    st.title(f"{APP_CONFIG['icon']} {APP_CONFIG['title']}")
    st.markdown("---")
    
    # 初始化session state
    initialize_session_state()
    
    # 渲染侧边栏
    api_key = render_sidebar()
    
    # 渲染文件上传区域
    uploaded_files, handle_files = render_file_upload()
    
    # 处理文件
    if uploaded_files and api_key and handle_files:
        try:
            with st.spinner("正在初始化助手..."):
                st.session_state.assistant = LiteratureReviewAssistant(api_key)
            
            with st.spinner("正在加载文档..."):
                documents = st.session_state.assistant.load_documents(uploaded_files)
            
            with st.spinner("正在分割文档..."):
                splits = st.session_state.assistant.split_documents(documents)
            
            with st.spinner("正在创建向量索引..."):
                st.session_state.assistant.create_vector_store(splits)
            
            with st.spinner("正在批量提取信息..."):
                # 准备批量处理数据
                papers_to_process = []
                for i, doc in enumerate(documents):
                    papers_to_process.append((f"文献_{i+1}_{doc.metadata['source']}", doc.page_content))
                
                results = st.session_state.assistant.batch_process_papers(papers_to_process)
            
            st.session_state.processed = True
            st.success("✨ 所有文献处理完成！")
            
        except Exception as e:
            st.error(f"处理过程中出现错误: {str(e)}")
    
    # 显示结果
    if st.session_state.processed and st.session_state.assistant is not None:
        # 渲染结果表格
        df = render_results_table(st.session_state.assistant)
        
        # 渲染问答界面
        render_qa_interface(st.session_state.assistant)
        
        # 渲染综合分析界面
        render_analysis_interface(st.session_state.assistant)

if __name__ == "__main__":
    main()