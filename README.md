# LCEngine

## ~~v0.1 - 原生 RAG 系统 MVP~~ 2025-09-19

- 一个功能完整的、基于 Streamlit 的 RAG 应用
- 用户可以上传一份 PDF/TXT 文档  
- 用户可以围绕该文档进行多轮对话问答
- 后端有清晰的日志，显示检索到的上下文片段

- [x] 核心 RAG 管道
  - 实现 RAG 全流程： Load -> Split -> Embed -> Store
  - 使用 numpy 余弦相似度 + pickle 存储替代向量数据库
  - 自实现简单的文本分割策略（固定长度 + 重叠）
  - 直接调用 OpenAI embeddings API 进行向量化

- [x] 多轮对话管理
  - 使用 Streamlit session_state 管理对话历史
  - 手动构建上下文：将历史对话与当前问题结合
  - 简单的独立查询生成逻辑
- [x] 前端
  - 使用 Streamlit 构建简洁的交互界面
- [x] 评估准备
  - 构建"黄金标准"评估集： 针对测试文档，精心设计 20-30 个 (问题, 理想答案) 对，并存为 JSON 文件

## v0.2 - 引入框架的评估驱动优化

从原生实现升级到框架支持，引入了高级检索策略，并建立了自动化评估流水线。

- [ ] 自动化评估流水线
  - 集成 RAGAs 框架进行量化评估
  - 编写一个评估脚本 (evaluate.py)，加载"黄金标准"评估集，自动计算 Faithfulness, Answer Relevancy, Context Precision 等核心指标
  - 运行脚本，得到 v0.1 版本的基线性能报告

- [ ] 向量存储升级
  - 从 numpy + pickle 升级到 FAISS 向量数据库
  - 提升检索速度和相似度搜索精度
  
- [ ] 高级检索策略实现
  - 实现 Re-ranking： 在召回后，集成一个 Cross-Encoder 模型（如 bge-reranker-base）进行重新排序，提升上下文的精准度
  - 实现查询转换 (Query Transformation)： 在检索前，让 LLM 根据对话历史和当前问题生成多个不同角度的子查询 (Multi-Query)，合并检索结果以提升召回率
  
- [ ] 迭代与验证
  - 将新策略集成到 RAG 管道中
  - 再次运行评估脚本，用数据量化对比 v0.1 和 v0.2 在各项指标上的提升。将对比结果记录在项目的 README.md 中

## Reference

- <https://python.langchain.com/docs/tutorials/rag/>
