# LCEngine

Learning copilot for LLM/深度学习：聊天式问答、可插拔检索与外网搜索、长期记忆、轻量 UI，既可自用也可用于面试展示。

## 快速开始

- 依赖：Python 3.10+，`uv`。
- 安装：`uv sync`
- 运行 UI：`python main.py`（默认启动 Streamlit，端口 8501）
  - 调试 UI：`streamlit run app.py --server.port 8501`
- 运行测试：`pytest`（或 `pytest -m "not slow"`）
- 代码质量：`ruff check app tests` / `ruff format app tests`，类型检查 `pyright app tests`

## 核心能力

- 多源知识：本地上传 PDF/TXT/MD，URL/GitHub ingest，外网搜索 Agent 默认开启（可关），站点白名单控制抓取范围；搜索结果会抓取 → 摘要 → 入库并标注来源/时间。
- 记忆体系：对话短期记忆 + 长期学习档案（高价值 Q/A、笔记、错题、已 ingest 资料），带主题/时间/来源/模型版本等 metadata，检索重排优先最近/常学方向。
- 检索与生成：FAISS 主索引、Multi-Query 扩展召回、Cross-Encoder 重排；生成起步用在线 API，后续可用自有数据微调开源模型（检索/重排/生成）。
- 模式：学习模式（讲解+学习计划+资料清单+练习）、解决问题模式（错误日志/调参建议）、快速问答模式（直接检索+回答）；UI 仅保留必要控件（模式切换、添加来源、保存笔记/错题、外网开关）。
- 可观测与安全：回答附来源/相似度；日志记录检索片段和工具调用轨迹；`.env` 管理密钥，外网搜索默认 ON 但可关闭，白名单限制域名。

## 使用方式

1. 启动后在聊天输入框提问，可先上传文档或提供 URL/GitHub 链接作为知识来源。
2. 需要额外上下文时可保持外网搜索开启，或在设置中关闭。
3. 对有价值的回答点“保存为笔记/错题”（待实现的轻量控件），写入长期记忆并带 metadata。
4. 在侧边栏（规划中）查看已 ingest 资料、笔记和开关状态。

## 配置与数据

- `.env`：`OPENAI_API_KEY`（必需），可扩展 `CHAT_MODEL`、`VECTOR_STORE_DIR`、外网白名单等配置。
- 数据目录：`data/` 作为向量库与缓存默认路径，请勿将大型生成物入 Git。
- 日志：默认通过 `config.get_logger`，包含检索片段与工具调用轨迹。

## 版本迭代规划

- **v0.1 基础 RAG MVP（已完成）**：Streamlit 聊天，上传 PDF/TXT，固定切分 + OpenAI Embedding，numpy+pickle 向量存储，多轮对话上下文拼接，日志展示检索片段。
- **v0.2 记忆 + FAISS 基础版**：升级 FAISS，存 metadata（主题/来源/时间/用户）；支持“保存为笔记/错题”写入长期记忆；UI 增加外网搜索开关（默认 ON）；产出首版评估集与 baseline（RAGAS/Hit@k/P95 延迟/成本），`evaluate.py` 最小跑法，结果落盘 `data/eval_runs/`。
- **v0.3 Ingest & 搜索路由**：添加 URL/GitHub ingest，站点白名单；本地命中不足时触发搜索 Agent → 抓取 → 摘要 → 入库并标注来源/时间；侧边栏展示已 ingest 资料。
- **v0.4 检索质量增强**：Multi-Query 查询生成、Cross-Encoder 重排（bge-reranker-\* 等在线模型）；支持按主题/时间/来源过滤与加权重排。
- **v0.5 学习模式 & 评估闭环**：学习模式模板（讲解+计划+练习），解决问题模式模板（错误日志/调参建议）；`evaluate.py` 跑 RAGAS 基线与策略对比，输出 JSON/CSV；在 README 记录关键指标提升。
- **v0.6 Agent 工具链 & 微调展示**：工具注册（搜索/ingest/代码运行/图表生成可选），白名单/开关可视化；收集对话与反馈，筹备或演示基于自有数据的微调/蒸馏；可选 LangGraph/LlamaIndex Workflow 自验证（Self-RAG/CoV）。

评估节奏：v0.2 完成后建立固定评测集和基准，后续每个版本（v0.3/v0.4）迭代后复跑并记录差异，在 `docs/DEVLOG.md` 和 `docs/experiments.md` 记录配置、指标、成本/延迟，方便面试展示多版本对比。

## 开发记录与指标追踪

- 变更与决策：`docs/DEVLOG.md` 记录日期/版本、改动、动机、决策取舍、遇到的坑。
- 实验与指标：`docs/experiments.md` 记录每次评估配置（模型/切分/检索/重排参数）、数据/文档来源、RAGAS 指标、消融/对比结论、成本/延迟。`evaluate.py` 运行结果落盘至 `data/eval_runs/<timestamp>.json|csv`，在文档中引用文件名。
- 演示素材：`docs/demo_script.md`（3-5 分钟剧情）、架构图（PNG/SVG）、指标表格/截图，便于面试演示。
- 安全：外网搜索默认 ON 可关，站点白名单控制抓取范围；向量库/长期记忆持久化在本地 `data/`，按用户/主题分区。

## 开发与质量

- 使用 `uv sync` 安装依赖（包含 pre-commit/pyright）
- 安装 git 钩子：`uv run pre-commit install`
- 本地快速检查：`uv run pre-commit run --all-files --show-diff-on-failure`
- 运行测试：`uv run pytest -m "not slow"`

## 参考

- <https://python.langchain.com/docs/tutorials/rag/>
