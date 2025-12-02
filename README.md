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

从原生实现升级到框架支持，引入了高级检索策略，并建立自动化评估流水线（不再依赖手写问答）。

- [ ] 自动化评估流水线
  - 集成 RAGAs 框架进行量化评估，并使用其 `Synthetic QA Generation`/`Contextual Precision` 组件自动构造评测样本
  - 编写一个评估脚本 (evaluate.py)，针对任意上传文档运行“生成 QA → 检索 → 度量”全流程，输出 Faithfulness、Answer Relevancy、Context Precision 等指标
  - 运行脚本，得到 v0.1 版本的基线性能报告并记录依赖（例如 `ragas`, `langchain-openai`）
  - 清理/弃用 `data/evaluation_dataset.json` 的手工问答依赖，改为在评测脚本里即时生成或缓存合成数据集
- [ ] 向量存储升级
  - 从 numpy + pickle 升级到 FAISS 向量数据库
  - 提升检索速度和相似度搜索精度
- [ ] 高级检索策略实现
  - 实现 Re-ranking： 在召回后，集成一个 Cross-Encoder 模型（如 bge-reranker-base）进行重新排序，提升上下文的精准度
  - 实现查询转换 (Query Transformation)： 在检索前，让 LLM 根据对话历史和当前问题生成多个不同角度的子查询 (Multi-Query)，合并检索结果以提升召回率
- [ ] 迭代与验证
  - 将新策略集成到 RAG 管道中
  - 再次运行评估脚本，用数据量化对比 v0.1 和 v0.2 在各项指标上的提升。将对比结果记录在项目的 README.md 中

## v0.3 - 论文讲解与学习路径助手

基于 v0.1 的原生 RAG 能力和 v0.2 的自动评估框架，聚焦“大模型/深度学习学习助手”场景，把系统升级成能抓取最新资料、生成讲解与学习计划的垂直产品。

- [ ] 多源知识抓取与清洗
  - 构建异步 ingestion（LangChain loaders + httpx/asyncio），接入 arXiv API、arXiv/顶会 RSS、技术博客、GitHub 教程、社区新闻
  - 统一解析 PDF/Markdown/HTML，抽取摘要、作者、会议、主题标签、引用，写入结构化 metadata，并记录 ingestion runs 以便重放
  - 在 `data/` 下按主题（NLP/CV/Agent 等）分区，配合 v0.2 的评测缓存以便回放
- [ ] 结构化 RAG 升级
  - 为每个主题生成多向量索引（语义 embedding + 专业术语向量 + 引用图 embedding），探索 FAISS 多索引或 Weaviate hybrid
  - 引入 GraphRAG/树状总结（LangGraph、llama-index KG）构建主题图谱，利用 `Chain-of-thought summarization`/`Tree-of-Thoughts` 排序上下文
  - 使用长上下文模型（GPT-4.1 128k、Claude 3.5、o1-mini long）或 `Long-context RAG`：粗召回 → 文本裁剪 → 二次聚焦
- [ ] 学习路径与讲解生成
  - `PaperSummarizerAgent`：输出“中文讲稿 + 知识卡片 + QA 要点 + 引用”；支持多模态（GPT-4o mini）生成示意图
  - `LearningPathAgent`：输入主题后自动规划章节、资料链接、练习题/代码 Demo，结合 ReAct 规划与工具调用
  - 所有回答附 `source`、`similarity`、`confidence`，并写审计日志，方便用户追踪与评估
- [ ] 反馈与评测闭环
  - 复用 v0.2 的 evaluate.py，对新抓取语料运行 RAGAs（Faithfulness/Context Precision）+ 自定义“讲解完整度/引用齐全度”指标
  - 在 UI/README 展示各专题指标变化，收集用户 thumbs up/down/讲解清晰度反馈，形成数据闭环

## v0.4 - Agent 工具链与推理增强

把 v0.3 的“资料 + 讲解”进一步升级到“多 Agent 协同的学习实验室”，引入推理型模型、工具自发现、长期记忆与安全自检。

- [ ] 推理模型与 LangGraph 编排
  - 接入 o1/o3、DeepSeek-R1、Qwen2.5-Math 等 reasoning-first 模型，根据任务自动路由
  - 在 LangGraph/LlamaIndex Workflow 中实现 ReAct/Self-Ask/Plan-and-Solve 策略，必要时使用 `chain_of_thought`/`chain_of_verification`
- [ ] 工具自发现与扩展
  - 设计标准 Tool Registry（检索、代码执行、图表生成、翻译、浏览器等），支持 YAML 配置热插拔
  - 研发 `CodeRunnerAgent`（复现论文伪代码/Colab + Matplotlib 图）、`QuizMasterAgent`（出题/批改 + 解析）、`NewsRadarAgent`（热点推送 + 要点卡片）
  - 探索 AutoTool/ToolGen 让模型根据任务描述组合脚本并在沙盒执行，记录成功率
- [ ] 长期记忆与学习体验
  - 引入 MemGPT/LTM-RAG，将对话、错题、兴趣压缩成长期记忆块，按需加载；展示知识星图、学习曲线
  - UI 增强：错题本、复习提醒、章节进度、推荐下一步任务，形成完整学习闭环
- [ ] 自我验证、合规与可观测性
  - 实现 Chain-of-Verification / Self-RAG，对答案做交叉验证并输出 `confidence_report`
  - 引入 Model Spec/Constitutional 规则，记录 Agent 工具链执行轨迹并在调试面板可视化
  - 评估脚本新增“工具调用成功率 / 反思次数 / 平均延迟”等指标，使多 Agent 提升可量化

## 开发与质量

- 使用 `uv sync` 安装依赖（包含 pre-commit/pyright）
- 安装 git 钩子：`uv run pre-commit install`
- 本地快速检查：`uv run pre-commit run --all-files --show-diff-on-failure`
- 运行测试：`uv run pytest -m "not slow"`

## Reference

- <https://python.langchain.com/docs/tutorials/rag/>
