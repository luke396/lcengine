# LCEngine Development Roadmap

> **项目定位**：面试展示级的工业 RAG+Agent 系统
> **核心策略**：70% Native 实现（展示深度）+ 30% 框架集成（提高效率）
> **时间规划**：3-4 周完成 v0.2-v0.4 核心功能

---

## 🎯 总体目标

1. **技术深度**：展示对 RAG/Agent 底层原理的深入理解
2. **工程能力**：完整的测试、评估、监控体系
3. **业务价值**：清晰的学习助手场景，解决真实问题
4. **差异化**：创新的长期记忆机制 + 系统化的质量评估

---

## 📋 版本规划总览

| 版本 | 核心目标                    | 技术选型       | 工作量 | 面试权重   | 状态      |
| ---- | --------------------------- | -------------- | ------ | ---------- | --------- |
| v0.1 | 基础 RAG MVP                | 100% Native    | -      | ⭐⭐⭐     | ✅ 已完成 |
| v0.2 | 记忆 + FAISS + 评估         | 100% Native    | 5-7 天 | ⭐⭐⭐⭐⭐ | 🚧 进行中 |
| v0.3 | 数据接入 + 搜索 Agent       | 混合策略       | 4-5 天 | ⭐⭐⭐⭐   | 📅 计划中 |
| v0.4 | 高级检索（Hybrid + Rerank） | 95% Native     | 5-6 天 | ⭐⭐⭐⭐⭐ | 📅 计划中 |
| v0.5 | 学习模式 + 复杂编排         | 引入 LangGraph | 4-5 天 | ⭐⭐⭐     | 📅 可选   |
| v0.6 | 监控 + 数据闭环             | 集成工具       | 3-4 天 | ⭐⭐       | 📅 可选   |

**推荐最小可面试版本**：v0.2 + v0.4（核心 RAG 质量）
**理想面试版本**：v0.2 + v0.3 + v0.4（完整能力展示）

---

## 📦 v0.1 基础 RAG MVP ✅

**状态**：已完成
**核心价值**：打好基础，证明可行性

### 已实现功能

- ✅ Streamlit 聊天界面
- ✅ PDF/TXT 文档上传和处理
- ✅ 基于 NumPy 的向量存储
- ✅ OpenAI Embedding + Chat API 集成
- ✅ 多轮对话上下文管理
- ✅ 完整的测试覆盖（85%+）
- ✅ Pre-commit hooks + CI/CD

### 技术栈

- **文档处理**：PyPDF2 + 自定义 chunking
- **向量存储**：SQLite + NumPy/pickle
- **Embedding**：OpenAI text-embedding-3-small
- **LLM**：OpenAI gpt-4.1-nano-2025-04-14

### 面试话术准备

> "我从最基础的实现开始，自己写了向量存储和检索逻辑。这让我深入理解了 embedding、cosine similarity、chunking 策略等核心概念。虽然 v0.2 会升级到 FAISS，但这个基础让我能清晰讲解向量检索的数学原理..."

---

## 🔥 v0.2 记忆 + FAISS + 评估框架 ⭐⭐⭐⭐⭐

**状态**：🚧 进行中
**优先级**：P0（最高）
**工作量**：5-7 天
**面试权重**：⭐⭐⭐⭐⭐（核心展示版本）

### 核心目标

1. **向量存储升级**：从 NumPy 迁移到 FAISS（工业级性能）
2. **长期记忆系统**：创新的笔记/错题机制（差异化亮点）
3. **评估框架建立**：RAGAS + 自定义指标（工程能力体现）

### 技术选型：100% Native

#### 实现清单

**PR1: FAISS 集成与数据层** (2-3 天)

- [ ] 新建 `app/vector_store/faiss_store.py`
  - 封装 FAISS IndexFlatIP 索引
  - 实现与 SQLiteVectorStore 统一的接口
  - metadata 持久化到 SQLite
- [ ] 工厂模式：`app/vector_store/__init__.py`
  ```python
  def get_vector_store(backend: Literal["faiss", "sqlite"]) -> VectorStore:
      if backend == "faiss":
          return FaissVectorStore(...)
      return SQLiteVectorStore(...)
  ```
- [ ] 配置迁移
  - 新增 `VECTOR_BACKEND` 环境变量（默认"faiss"）
  - 保留 `VECTOR_STORE_DB_PATH` 向后兼容
  - 新增 `FAISS_INDEX_PATH = data/faiss/index.faiss`
- [ ] 数据库 Schema 扩展

  ```sql
  -- 扩展chunks表
  ALTER TABLE chunks ADD COLUMN doc_type TEXT CHECK(doc_type IN ('document','note','mistake'));
  ALTER TABLE chunks ADD COLUMN topic TEXT;
  ALTER TABLE chunks ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP;
  ALTER TABLE chunks ADD COLUMN user TEXT;

  -- 新建labels表
  CREATE TABLE labels(
      chunk_id INTEGER,
      label TEXT,
      FOREIGN KEY(chunk_id) REFERENCES chunks(id)
  );
  CREATE INDEX idx_labels_label ON labels(label);
  ```

- [ ] 单元测试
  - FAISS store CRUD 操作
  - metadata 过滤查询
  - 持久化和加载
  - 性能 benchmark（10k chunks）

**PR2: 长期记忆与 UI 集成** (2-3 天)

- [ ] 扩展 `DocumentChunk` 模型
  ```python
  @dataclass
  class DocumentChunk:
      # 原有字段...
      doc_type: str = "document"  # document | note | mistake
      topic: Optional[str] = None
      labels: list[str] = field(default_factory=list)
      created_at: datetime = field(default_factory=datetime.now)
  ```
- [ ] 记忆保存功能
  - UI 按钮："保存为笔记" / "标记为错题"
  - 可选 LLM 精炼（生成 refined_question/answer/explanation）
  - 写入 vector store + 标记 metadata
- [ ] 检索加权机制（核心创新）
  ```python
  def weighted_retrieval(chunks, query, bias=0.05):
      """笔记/错题在检索时加权+0.05"""
      for chunk in chunks:
          if chunk.doc_type in ['note', 'mistake']:
              chunk.score += bias
      return sorted(chunks, key=lambda x: x.score, reverse=True)
  ```
- [ ] UI 改进
  - Sidebar 显示外网搜索开关（占位，v0.3 实现）
  - 显示向量库统计（文档数/笔记数/错题数）
  - Debug 区展示 doc_type/topic/created_at
- [ ] 集成测试
  - 端到端流程：上传文档 → 提问 → 保存笔记 → 再次检索命中

**PR3: 评估工具链** (1-2 天)

- [ ] 创建 `evaluate.py` 脚本
  ```bash
  python evaluate.py \
      --dataset tests/data/evaluation_dataset.json \
      --k 5 \
      --vector-backend faiss \
      --output data/eval_runs/$(date +%Y%m%d_%H%M%S).json
  ```
- [ ] 实现指标
  - **Hit@k**：检索是否命中正确文档
  - **RAGAS**：
    - Faithfulness（防幻觉）
    - Context Precision/Recall（检索质量）
    - Answer Relevancy（回答相关性）
  - **P95 Latency**：性能指标
  - **Cost Tracking**：API 调用成本
- [ ] 输出格式
  ```json
  {
    "timestamp": "2025-01-15T10:30:00",
    "config": {"model": "gpt-4.1-nano", "k": 5, "backend": "faiss"},
    "metrics": {
      "hit_at_5": 0.82,
      "ragas_faithfulness": 0.88,
      "p95_latency_ms": 1200,
      "avg_cost_per_query": 0.003
    },
    "details": [...]
  }
  ```
- [ ] Baseline 建立
  - 运行 v0.1 配置作为 baseline
  - 运行 v0.2 配置对比提升
  - 记录到 `docs/experiments.md`

### 成功指标

- ✅ FAISS 检索延迟 < NumPy（10k chunks 场景下 <200ms）
- ✅ 长期记忆命中率 > 普通文档（+0.05 bias 效果验证）
- ✅ RAGAS Faithfulness > 0.80
- ✅ 测试覆盖率保持 85%+

### 面试展示重点

1. **FAISS 选型**："我对比了 ChromaDB、Weaviate 和 FAISS。对于我的场景（单机、10k-100k 文档），FAISS 的 IndexFlatIP 足够且无额外依赖。我做了 benchmark..."
2. **长期记忆设计**："传统 RAG 只检索文档，我创新地将用户的笔记和错题也向量化。通过 doc_type 加权，系统会优先回忆起我之前犯过的错误，避免重复踩坑..."
3. **评估体系**："我建立了完整的评估流水线，每次改动都会跑 RAGAS 指标对比。从 v0.1 到 v0.2，Faithfulness 提升了 12%，这归功于长期记忆减少了幻觉..."

### 风险与应对

- **风险**：FAISS 迁移可能导致数据格式不兼容
  - **应对**：提供迁移脚本 + 清晰的迁移文档
- **风险**：LLM 精炼记忆的成本
  - **应对**：设为可选功能，失败回退到原始内容

---

## 🌐 v0.3 数据接入 + 搜索 Agent ⭐⭐⭐⭐

**优先级**：P1
**工作量**：4-5 天
**面试权重**：⭐⭐⭐⭐（Agent 能力展示）

### 核心目标

1. 多源数据接入（URL/GitHub）
2. 实现基础搜索 Agent
3. 动态知识库更新

### 技术选型：混合策略

#### 为什么这里可以用框架？

- **数据接入**：LlamaIndex Readers 处理 HTTP/API 细节（非核心价值）
- **搜索逻辑**：自己实现 Agent 决策和工具调用（核心价值）
- **比例**：30% 框架（Readers） + 70% Native（Agent 逻辑）

#### 实现清单

**数据接入** (1-2 天)

- [ ] 集成 LlamaIndex Readers（框架部分）

  ```python
  # 仅用于数据获取
  from llama_index.readers import SimpleWebPageReader, GithubRepositoryReader

  def ingest_url(url: str):
      reader = SimpleWebPageReader()
      docs = reader.load_data([url])
      # 后续处理自己实现
      return custom_process_and_chunk(docs)
  ```

- [ ] 自定义后处理（Native 部分）
  ```python
  def custom_process_and_chunk(docs):
      """
      针对不同内容类型的智能分块
      - 代码：按函数/类切分
      - 文档：按语义切分
      - 表格：结构化提取
      """
      chunks = []
      for doc in docs:
          if is_code(doc):
              chunks.extend(chunk_by_ast(doc))  # AST解析
          elif is_markdown(doc):
              chunks.extend(chunk_by_header(doc))
          else:
              chunks.extend(semantic_chunking(doc))
      return chunks
  ```
- [ ] 站点白名单
  ```python
  ALLOWED_DOMAINS = [
      "pytorch.org",
      "huggingface.co",
      "github.com",
      "arxiv.org"
  ]
  ```

**搜索 Agent 实现** (2-3 天) - 100% Native

- [ ] Agent 基础框架

  ```python
  class SearchAgent:
      """
      决策逻辑：
      1. 本地检索置信度 > 0.7 → 直接返回
      2. 置信度 < 0.7 → 触发搜索
      3. 搜索结果 → 摘要 → 入库
      """
      def __init__(self):
          self.tools = {
              'search': self._search_web,
              'fetch': self._fetch_url,
              'summarize': self._summarize
          }

      def decide_and_act(self, query, local_confidence):
          if local_confidence > 0.7:
              return "use_local"

          # 搜索决策
          search_results = self.tools['search'](query)
          filtered = self._filter_by_whitelist(search_results)

          for url in filtered[:3]:
              content = self.tools['fetch'](url)
              summary = self.tools['summarize'](content)
              self._add_to_vector_store(summary, metadata={
                  'source': url,
                  'ingested_at': datetime.now(),
                  'doc_type': 'web_search'
              })
  ```

- [ ] 工具实现
  - `_search_web`: 集成 DuckDuckGo API 或 SerpAPI
  - `_fetch_url`: requests + BeautifulSoup 清洗
  - `_summarize`: 调用 LLM 生成摘要

**UI 集成** (1 天)

- [ ] 外网搜索开关（v0.2 已占位）
- [ ] 侧边栏展示已 ingest 资料
  ```
  📚 知识库 (125条)
  ├─ 📄 本地文档 (45)
  ├─ 🌐 Web页面 (58)
  ├─ 💾 GitHub仓库 (12)
  ├─ 📝 笔记 (8)
  └─ ❌ 错题 (2)
  ```

### 成功指标

- ✅ 成功接入 3 种数据源（PDF/URL/GitHub）
- ✅ 搜索 Agent 触发准确率 > 90%
- ✅ Web 内容摘要质量（人工评估 5 分制 > 4 分）
- ✅ 白名单阻止率 100%（安全性）

### 面试展示重点

1. **框架使用判断**："数据接入我用了 LlamaIndex 的 Reader，因为处理 HTTP 和 HTML 清洗是通用问题。但**chunking 策略我自己实现**，因为代码和文档的切分逻辑完全不同——代码要保持 AST 完整性，文档要按语义分块..."
2. **Agent 设计**："我设计了一个简单但有效的决策机制：当本地检索置信度低于 0.7 时触发搜索。搜索结果会自动摘要并标注来源时间，形成动态更新的知识库..."
3. **安全控制**："外网搜索虽然默认开启，但我实现了双重保护：域名白名单 + 用户可随时关闭。这在企业场景很重要..."

---

## 🚀 v0.4 高级检索质量优化 ⭐⭐⭐⭐⭐

**优先级**：P0（与 v0.2 并列最高）
**工作量**：5-6 天
**面试权重**：⭐⭐⭐⭐⭐（技术深度核心）

### 核心目标

这是展示 RAG 技术深度的**核心版本**，全部自己实现高级检索算法。

### 技术选型：95% Native

#### 为什么几乎全 Native？

这部分是面试的**技术高光时刻**，必须能深入讲解数学原理和实现细节。

#### 实现清单

**1. BM25 检索** (1 天)

```python
# 使用第三方BM25库（不算框架）
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """
    稀疏检索，擅长关键词匹配
    公式：score = IDF(q) * (f(q,D) * (k1+1)) / (f(q,D) + k1*(1-b+b*|D|/avgDL))
    """
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.bm25 = BM25Okapi(corpus, k1=k1, b=b)
        self.k1 = k1
        self.b = b

    def retrieve(self, query, top_k=20):
        scores = self.bm25.get_scores(tokenize(query))
        return get_top_k(scores, top_k)
```

**2. Hybrid 检索融合** (2 天) - 核心算法

```python
class HybridRetriever:
    """
    混合BM25（稀疏）和向量检索（密集）
    解决各自的短板：
    - BM25擅长关键词，但无语义理解
    - 向量擅长语义，但关键词匹配弱
    """
    def __init__(self, bm25_retriever, vector_store, alpha=0.7):
        self.bm25 = bm25_retriever
        self.vector = vector_store
        self.alpha = alpha  # 向量权重（通过评估集调优）

    def retrieve(self, query, top_k=5):
        # 1. 两路并行检索
        bm25_results = self.bm25.retrieve(query, top_k=20)
        vector_results = self.vector.search(query, top_k=20)

        # 2. 分数归一化（关键！）
        bm25_scores = self._normalize(bm25_results)
        vector_scores = self._normalize(vector_results)

        # 3. 加权融合（RRF或线性加权）
        combined = self._reciprocal_rank_fusion(
            bm25_scores, vector_scores
        )
        # 或线性加权：
        # combined = alpha * vector_scores + (1-alpha) * bm25_scores

        return combined[:top_k]

    def _reciprocal_rank_fusion(self, results_a, results_b, k=60):
        """
        RRF: score = 1/(k + rank_a) + 1/(k + rank_b)
        无需分数归一化，直接用排名
        """
        scores = defaultdict(float)
        for rank, (doc_id, _) in enumerate(results_a):
            scores[doc_id] += 1 / (k + rank + 1)
        for rank, (doc_id, _) in enumerate(results_b):
            scores[doc_id] += 1 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**3. Cross-Encoder 重排序** (1-2 天)

```python
from sentence_transformers import CrossEncoder

class Reranker:
    """
    二阶段检索：
    1. Hybrid检索召回top-20（快速）
    2. Cross-Encoder精排top-5（慢但准）
    """
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3'):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k=5):
        """
        计算query和每个候选的相关性分数
        时间复杂度：O(n) - 每对都要过BERT
        """
        pairs = [(query, doc.text) for doc in candidates]
        scores = self.model.predict(pairs)

        # 结合原始检索分数
        for doc, rerank_score in zip(candidates, scores):
            doc.rerank_score = rerank_score

        return sorted(candidates, key=lambda x: x.rerank_score, reverse=True)[:top_k]
```

**4. 元数据过滤与加权** (1 天)

```python
class MetadataAwareRetriever:
    """
    根据元数据动态调整检索策略
    """
    def retrieve(self, query, filters=None, weights=None):
        # 1. 基础检索
        candidates = self.hybrid_retriever.retrieve(query, top_k=50)

        # 2. 元数据过滤
        if filters:
            candidates = [c for c in candidates if self._match_filters(c, filters)]

        # 3. 时间衰减加权
        for c in candidates:
            age_days = (datetime.now() - c.created_at).days
            time_decay = math.exp(-age_days / 30)  # 30天半衰期
            c.score *= time_decay

        # 4. 类型加权（v0.2的长期记忆机制）
        if weights:
            for c in candidates:
                if c.doc_type in weights:
                    c.score *= weights[c.doc_type]

        # 5. 重排序
        return self.reranker.rerank(query, candidates, top_k=5)
```

**5. 完整 Pipeline 集成** (1 天)

```python
class AdvancedRAGPipeline:
    def __init__(self):
        self.bm25 = BM25Retriever(...)
        self.vector_store = FaissVectorStore(...)
        self.hybrid = HybridRetriever(self.bm25, self.vector_store, alpha=0.72)
        self.reranker = Reranker()
        self.metadata_retriever = MetadataAwareRetriever(self.hybrid, self.reranker)

    def query(self, question, filters=None):
        # 1. 检索
        chunks = self.metadata_retriever.retrieve(
            question,
            filters=filters,
            weights={'note': 1.2, 'mistake': 1.5, 'document': 1.0}
        )

        # 2. 构建上下文
        context = self._build_context(chunks)

        # 3. 生成答案
        answer = self.llm.generate(question, context)

        return {
            'answer': answer,
            'sources': [c.source for c in chunks],
            'debug': {
                'retrieval_scores': [c.score for c in chunks],
                'rerank_scores': [c.rerank_score for c in chunks]
            }
        }
```

### 评估计划

在 `docs/experiments.md` 记录对比实验：

| 策略                   | Hit@5    | RAGAS<br>Context<br>Recall | RAGAS<br>Faithfulness | P95<br>Latency | 备注               |
| ---------------------- | -------- | -------------------------- | --------------------- | -------------- | ------------------ |
| Baseline (v0.2 纯向量) | 0.68     | 0.72                       | 0.80                  | 450ms          | -                  |
| +BM25 混合             | 0.75     | 0.78                       | 0.82                  | 520ms          | alpha=0.7          |
| +RRF 融合              | 0.77     | 0.80                       | 0.83                  | 530ms          | k=60               |
| +Cross-Encoder         | **0.85** | **0.88**                   | **0.89**              | 1100ms         | bge-reranker-v2-m3 |
| +元数据加权            | **0.87** | **0.90**                   | **0.90**              | 1150ms         | 时间衰减+类型权重  |

### 成功指标

- ✅ Hit@5 > 0.85（相比 v0.2 提升 >15%）
- ✅ RAGAS Context Recall > 0.88
- ✅ P95 Latency < 1500ms（可接受的延迟增加）
- ✅ 能清晰讲解每个算法的数学原理

### 面试展示重点（核心！）

**1. Hybrid Search 深度讲解**

> "纯向量检索在关键词匹配上有弱点，比如查'PyTorch 2.0 新特性'，如果文档里是'PyTorch version 2.0 features'，语义相近但向量距离可能不是最优。BM25 能精准匹配'PyTorch'和'2.0'这些关键词。
>
> 我实现了 RRF（Reciprocal Rank Fusion）融合策略，公式是 score = 1/(k+rank)。相比线性加权，RRF 的优势是不需要归一化分数，直接用排名，更鲁棒..."

**2. 参数调优过程**

> "我通过评估集调优了两个关键参数：
>
> - Hybrid 的 alpha 权重：测试了 0.5-0.9，发现 0.72 时 Hit@5 最高
> - RRF 的 k 值：测试了 30/60/100，k=60 平衡了两路检索的贡献度
>
> 这些都记录在`experiments.md`里，有完整的消融实验..."

**3. 延迟 vs 质量权衡**

> "Cross-Encoder 让延迟从 450ms 增加到 1100ms，但准确率提升了 17 个百分点。对于学习场景，用户更在意答案质量而非实时性，所以这个 trade-off 是值得的。
>
> 未来优化方向可以考虑：
>
> - 异步重排序（先返回 Hybrid 结果，后台重排）
> - 缓存热门 query 的重排序结果
> - 使用更快的轻量级 reranker"

---

## 🎓 v0.5 学习模式 + LangGraph 编排 ⭐⭐⭐

**优先级**：P2（可选）
**工作量**：4-5 天
**面试权重**：⭐⭐⭐（展示框架使用判断）

### 核心目标

实现复杂的多步骤工作流，展示"何时用框架"的判断力。

### 技术选型：引入 LangGraph

#### 为什么这里用框架合适？

- **复杂度**：多模式切换、状态管理（>500 LOC 自己实现）
- **框架优势**：LangGraph 专门为此设计，有可视化调试
- **不影响核心**：v0.2-v0.4 的 RAG 逻辑仍是自己的
- **展示判断力**：证明你知道"何时造轮子、何时用轮子"

### 实现清单

**学习模式状态机** (2-3 天)

```python
from langgraph.graph import StateGraph, END

class LearningModeWorkflow:
    """
    学习模式流程：
    理解概念 → 制定计划 → 推荐资料 → 生成练习 → 评估理解
    """
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline  # 使用v0.4的检索pipeline
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(State)

        # 定义节点（每个节点仍用自己的RAG）
        workflow.add_node("understand", self._understand_concept)
        workflow.add_node("plan", self._create_learning_plan)
        workflow.add_node("resources", self._recommend_resources)
        workflow.add_node("practice", self._generate_practice)

        # 定义边
        workflow.add_edge("understand", "plan")
        workflow.add_edge("plan", "resources")
        workflow.add_conditional_edges(
            "resources",
            self._should_practice,
            {
                "yes": "practice",
                "no": END
            }
        )

        workflow.set_entry_point("understand")
        return workflow.compile()

    def _understand_concept(self, state):
        """使用自己的RAG检索相关概念"""
        concept = state['query']
        chunks = self.rag.metadata_retriever.retrieve(
            f"explain {concept}",
            filters={'doc_type': ['document', 'note']}
        )
        explanation = self.rag.llm.generate(
            f"Explain {concept} based on:\n{chunks}"
        )
        state['explanation'] = explanation
        return state
```

**问题解决模式** (1-2 天)

```python
class ProblemSolvingWorkflow:
    """
    问题解决流程：
    分析错误 → 检索相似错误 → 生成解决方案 → 标记为错题
    """
    def _analyze_error(self, state):
        error_log = state['error_log']
        # 检索历史错题
        similar_mistakes = self.rag.metadata_retriever.retrieve(
            error_log,
            filters={'doc_type': 'mistake'},
            weights={'mistake': 2.0}  # 高权重历史错误
        )
        # ...
```

### 面试展示重点

> "前面的核心 RAG 我都是自己实现的，但到了复杂工作流编排，我引入了 LangGraph。原因有三：
>
> 1. 状态管理复杂度高（多模式、条件跳转）
> 2. LangGraph 提供可视化调试，提高开发效率
> 3. 关键是**我的检索逻辑仍然用 v0.4 自己实现的 Hybrid+Rerank**，框架只负责编排
>
> 这体现了我的判断：核心算法自己掌控，外围工具合理使用。"

---

## 📊 v0.6 监控 + 数据闭环 ⭐⭐

**优先级**：P3（时间充裕可做）
**工作量**：3-4 天
**面试权重**：⭐⭐（加分项）

### 实现清单

- [ ] 集成 LangSmith/LangFuse 做 trace 可视化
- [ ] Prometheus metrics 导出
- [ ] 用户反馈收集界面
- [ ] （可选）微调数据收集 pipeline

---

## 📈 评估与质量保证

### 每个版本的评估流程

1. **运行评估脚本**

   ```bash
   python evaluate.py --dataset tests/data/evaluation_dataset.json
   ```

2. **记录到 experiments.md**

   ```markdown
   ## v0.4 Hybrid Search 实验 (2025-01-20)

   ### 配置

   - Model: gpt-4.1-nano
   - Retrieval: BM25(k1=1.5, b=0.75) + FAISS(dim=1536) + RRF(k=60)
   - Reranker: bge-reranker-v2-m3

   ### 结果

   | 指标               | v0.2 Baseline | v0.4 | 提升 |
   | ------------------ | ------------- | ---- | ---- |
   | Hit@5              | 0.68          | 0.85 | +25% |
   | RAGAS Faithfulness | 0.80          | 0.89 | +11% |

   ### 结论

   RRF 融合效果优于线性加权，alpha 参数对结果影响显著...
   ```

3. **更新 DEVLOG.md**
   记录决策过程、遇到的坑、解决方案

---

## 🎯 面试准备清单

### 3 分钟电梯演讲

```
我开发了LCEngine，一个工业级的RAG学习助手。

【问题】：通用LLM在深度学习领域容易幻觉，且无法记住我的学习进度。

【方案】：我实现了三个核心创新：
1. Hybrid检索（BM25+向量+重排序），准确率提升25%
2. 长期记忆机制（笔记/错题向量化），避免重复踩坑
3. 完整评估体系（RAGAS+自定义指标），每次改动都有数据支撑

【技术亮点】：
- 核心算法70%自己实现，展示深度理解
- 30%合理使用框架（数据接入/复杂编排），展示工程判断
- 完整的测试（85%覆盖）、评估、文档体系

【成果】：从v0.1到v0.4，检索准确率从68%提升到87%，
        这些提升都记录在我的experiments.md中...
```

### Demo 脚本准备

创建 `docs/demo_script.md`，包含：

1. 场景演示（5 分钟）
2. 代码 walk-through（5 分钟）
3. 指标对比展示（2 分钟）

### 关键问题准备

1. "为什么不用 LangChain？" → 已准备
2. "如果数据量增长到 1000 万文档怎么办？" → 扩展方案
3. "如何防止幻觉？" → Faithfulness 指标 + 来源标注
4. "成本如何控制？" → Embedding 缓存 + 模型降级策略

---

## 📅 时间线建议

### 最小可面试版本（2 周）

- Week 1: v0.2 完整实现
- Week 2: v0.4 完整实现
- **成果**：核心 RAG 质量优秀，足以面试

### 理想版本（3 周）

- Week 1: v0.2
- Week 2: v0.3 + v0.4 前半部分
- Week 3: v0.4 完成 + 文档/Demo 准备
- **成果**：完整能力展示

### 完整版本（4 周）

- 前 3 周同上
- Week 4: v0.5（可选）+ v0.6 监控 + 完善文档
- **成果**：工业级完整度

---

## 🎓 学习资源

### 推荐阅读

- RAGAS 论文及文档
- FAISS 官方文档
- BGE Reranker 技术报告
- LangGraph 教程（v0.5 使用）

### 对比项目研究

- LlamaIndex 官方示例
- LangChain RAG 教程
- 分析它们的优缺点，准备面试对比讨论

---

**总结**：这个路线图的核心策略是**"在展示深度的地方 Native 实现，在提高效率的地方合理使用框架"**。完成 v0.2+v0.4 后，你将拥有一个能深入讲解技术细节、又有完整工程实践的面试项目。
