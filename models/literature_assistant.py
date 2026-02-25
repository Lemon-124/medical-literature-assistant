"""医学文献处理核心模型"""

import os
import tempfile
import hashlib
import json
import re
import time
import threading
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from utils.config import PROCESSING_CONFIG, TABLE_COLUMNS



class LiteratureReviewAssistant:
    """医学文献综述助手核心类"""
    
    def __init__(self, api_key: str, model_name: str = "qwen3-max"):
        """
        初始化助手
        """
        # 初始化LLM
        self.llm = ChatTongyi(
            temperature=0.1,
            model_name=model_name,
            dashscope_api_key=api_key
        )
        
        # 初始化embedding模型
        self.embeddings = DashScopeEmbeddings(
            dashscope_api_key=api_key
        )
        
        # 初始化向量存储
        self.vector_store = None
        
        # 创建用于存储提取结果的DataFrame
        self.results_df = pd.DataFrame(columns=TABLE_COLUMNS)

    def _get_file_hash(self, file):
        """计算文件哈希值，用于去重"""
        return hashlib.md5(file.getvalue()).hexdigest()

    def load_documents(self, uploaded_files):
        """
        加载上传的文档文件
        支持：PDF
        """
        documents = []
        file_contents = []

        for uploaded_file in uploaded_files:
            # 保存上传的文件到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # 根据文件类型选择加载器
            try:
                if uploaded_file.name.endswith('.pdf'):
                    loader = PDFPlumberLoader(tmp_path)
                else:
                    continue
                
                docs = loader.load()

                # 合并同一文件的所有页面内容
                file_content = ""
                file_metadata = {}
                
                for i, doc in enumerate(docs):
                    file_content += doc.page_content + "\n\n"
                    if i == 0:  # 只取第一页的元数据
                        file_metadata = doc.metadata.copy()
                
                # 创建合并后的文档对象
                merged_doc = Document(
                    page_content=file_content.strip(),
                    metadata={
                        "source": uploaded_file.name,
                        "file_hash": self._get_file_hash(uploaded_file),
                        "total_pages": len(docs)
                    }
                )                

                documents.append(merged_doc)
                
            except Exception as e:
                raise Exception(f"加载文件 {uploaded_file.name} 时出错: {str(e)}")
            finally:
                # 清理临时文件
                os.unlink(tmp_path)
        
        return documents
    
    def smart_text_preprocessing(self, doc_content: str) -> str:
        """
        智能文本预处理
        功能：
        1. 移除多余空白字符
        2. 智能提取关键章节（针对长文献）
        3. 保持文献结构完整性
        4. 控制文本长度适配AI模型
        """
        # 1. 基础清理
        content = re.sub(r'\s+', ' ', doc_content.strip())
    
        # 2. 初始化关键部分列表
        key_sections = []

        # 3. 如果文本过长，启动智能处理
        if len(content) > 10000:  
        
            # 4. 按优先级提取关键部分
            section_priority = [
                r'(abstract|摘要).*?(?=\n\n(?:introduction|引言|method|方法|background|背景))',
                r'(introduction|引言).*?(?=\n\n(?:method|方法|materials|材料|study design|研究设计))',
                r'(method|方法|materials|材料).*?(?=\n\n(?:result|结果|findings|发现|analysis|分析))',
                r'(result|结果|findings|发现).*?(?=\n\n(?:discussion|讨论|conclusion|结论))',
                r'(discussion|讨论|conclusion|结论).*?(?=\n\n(?:reference|参考文献|acknowledgment|致谢))'
            ]
        
        # 5. 提取各关键部分
        for pattern in section_priority:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                section_content = match.group().strip()
                # 过滤过短的内容（少于100字符）
                if len(section_content) > 100:
                    key_sections.append(section_content)
        
        # 6. 智能重组策略
        if key_sections:
            # 按重要性排序并组合
            processed_content = '\n\n'.join(key_sections)
            
            # 7. 长度控制
            max_length = PROCESSING_CONFIG.get("max_content_length", 15000)
            if len(processed_content) > max_length:
                # 保留前几个最重要部分
                truncated_sections = []
                current_length = 0
                
                for section in key_sections:
                    if current_length + len(section) + 2 <= max_length:
                        truncated_sections.append(section)
                        current_length += len(section) + 2  # +2 for \n\n
                    else:
                        # 截断最后一个部分
                        remaining_space = max_length - current_length
                        if remaining_space > 200:  # 至少保留200字符
                            truncated_section = section[:remaining_space-50] + "...[内容截断]"
                            truncated_sections.append(truncated_section)
                        break
                
                processed_content = '\n\n'.join(truncated_sections)
        else:
            # 未找到明确结构，智能截断
            processed_content = content[:PROCESSING_CONFIG.get("max_content_length", 15000)]
            
            # 8. 最终清理
            processed_content = re.sub(r'\s+', ' ', processed_content).strip()
    
        return processed_content

    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "；", "，", "、", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)
    
    def create_vector_store(self, splits, persist_directory="./chroma_db"):
        """
        创建向量存储
        """
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        return self.vector_store
    
    def extract_information_from_paper(self, doc_content: str, doc_name: str) -> Dict:
        """
        从单篇文献中提取结构化信息
        """
        # 定义提取信息的Prompt模板
        prompt_template = ChatPromptTemplate([
            ("system", """你是一个专业的医学文献分析专家。你的任务是从给定的医学文献内容中，提取关键信息，并以结构化的JSON格式返回。
            请仔细阅读文本，提取以下信息：
            1. 研究目的：该文献主要想解决什么问题
            2. 实验方法：使用了什么研究方法、实验设计、样本量等
            3. 实验结果和结论：最重要的研究发现和结论
            4. 方法优缺点：研究方法的优势和局限性
            5. 关键引用：3-5个可以直接用在综述里的关键句子
            
            注意：
            - 如果某部分信息在文本中不存在，请填写"未提及"
            - 保持专业、客观、准确
            - 用中文输出"""),
            ("human", "文献标题/名称：{doc_name}\n\n文献内容：{doc_content}")
        ])
        
        # 创建处理链
        chain = prompt_template | self.llm | StrOutputParser()
        
        # 添加JSON格式要求的额外提示
        format_instruction = """
        请以JSON格式输出，格式如下：
        {
            "研究目的": "这里填写研究目的",
            "实验方法": "这里填写实验方法",
            "实验结果和结论": "这里填写主要结论",
            "方法优缺点": "这里填写方法优缺点",
            "关键引用": ["引用1", "引用2", "引用3"]
        }
        """
        
        try:
            response = chain.invoke({
                "doc_name": doc_name,
                "doc_content": doc_content[:PROCESSING_CONFIG["max_content_length"]] + format_instruction
            })

            json_match = re.search(r'\{.*\}', response, re.DOTALL)

            # 尝试从响应中提取JSON
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"raw_response": response}

            # 添加元数据
            result["提取时间"] = pd.Timestamp.now()
            result["文献名"] = doc_name
            
            return result

        except Exception as e:  
            return {
                "文献名": doc_name,
                "错误信息": str(e),
                "提取时间": pd.Timestamp.now()
            }
    
    def batch_process_papers(self, papers_content: List[tuple], max_workers: int = 4):
        """
        批量处理多篇文献
        """
        results = []
        total_papers = len(papers_content)
        
        # 线程安全的结果收集
        results_lock = threading.Lock()
        processed_count = 0
        start_time = time.time()

        def process_single_paper(paper_data):
            doc_name, doc_content = paper_data
            try:
                result = self.extract_information_from_paper(doc_content, doc_name)
                
                with results_lock:
                    results.append(result)
                    nonlocal processed_count
                    processed_count += 1
                
                return result
            except Exception as e:
                error_result = {
                    "文献名": doc_name,
                    "错误信息": str(e),
                    "提取时间": pd.Timestamp.now()
                }
                with results_lock:
                    results.append(error_result)
                    processed_count += 1
                return error_result
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paper = {
                executor.submit(process_single_paper, paper_data): paper_data 
                for paper_data in papers_content
            }
            
            for future in as_completed(future_to_paper):
                try:
                    future.result(timeout=PROCESSING_CONFIG["timeout"])
                except Exception as e:
                    paper_data = future_to_paper[future]
                    raise Exception(f"任务执行失败 {paper_data[0]}: {str(e)}")
        
        # 转换为DataFrame
        if results:
            new_rows = []
            for r in results:
                row = {
                    "文献名": r.get("文献名", ""),
                    "研究目的": r.get("研究目的", ""),
                    "实验方法": r.get("实验方法", ""),
                    "实验结果和结论": r.get("实验结果和结论", ""),
                    "方法优缺点": r.get("方法优缺点", ""),
                    "关键引用": "\n".join(r.get("关键引用", [])),
                    "提取时间": r.get("提取时间", "")
                }
                new_rows.append(row)
                
            self.results_df = pd.concat([
                self.results_df,
                pd.DataFrame(new_rows)
            ], ignore_index=True)
        
        return results
    
    def generate_summary_table(self):
        """
        生成汇总表格
        """
        return self.results_df
    
    def ask_question(self, question: str, k: int = 5):
        """
        基于已处理的文献回答问题
        """
        if not self.vector_store:
            return "请先上传并处理文献"
        
        # 检索相关文档
        docs = self.vector_store.similarity_search(question, k=k)
        
        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 创建问答链
        qa_prompt = ChatPromptTemplate([
            ("system", "你是一个专业的医学研究助手。基于以下文献内容，回答问题。如果问题涉及多篇文献，请进行综合比较。"),
            ("human", "文献内容：{context}\n\n问题：{question}")
        ])
        
        chain = qa_prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "context": context,
            "question": question
        })

        return response
