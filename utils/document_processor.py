"""文档处理工具"""

import streamlit as st
from typing import List, Any
import time

def display_progress(total_papers: int):
    """显示处理进度"""
    progress_container = st.container()
    with progress_container:
        col1, col2, col3 = st.columns(3)
        with col1:
            progress_text = st.empty()
        with col2:
            speed_text = st.empty()
        with col3:
            eta_text = st.empty()

        overall_progress = st.progress(0)
        current_processing = st.empty()
    
    return progress_container, progress_text, speed_text, eta_text, overall_progress, current_processing

def update_progress(processed_count: int, total_papers: int, start_time: float,
                   progress_text, speed_text, eta_text, overall_progress):
    """更新进度显示"""
    progress = processed_count / total_papers
    overall_progress.progress(progress)
    
    # 计算处理速度和预计剩余时间
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        speed = processed_count / elapsed_time * 60  # 篇/分钟
        remaining = total_papers - processed_count
        eta = remaining / (processed_count / elapsed_time) if processed_count > 0 else 0
        
        progress_text.text(f"进度: {processed_count}/{total_papers} ({progress:.1%})")
        speed_text.text(f"速度: {speed:.1f} 篇/分钟")
        eta_text.text(f"预计剩余: {eta/60:.1f} 分钟")

def cleanup_progress_display(progress_container):
    """清理进度显示"""
    progress_container.empty()