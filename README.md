# 法律条文检索问答（RAG 小项目）

> 基于法律条文的检索增强问答：**文档切分 → 向量化(bge-m3) → Milvus 向量检索 → DeepSeek 生成带条文引用的回答**。
> 覆盖 RAG 全流程：加载 → 切分 → 向量化 → 入库 → 检索 → 拼 Prompt → 生成。

## 功能特点

- 按"第X条"优先切分，超长条文再按固定长度切（带重叠），保证语义完整
- 使用 **Milvus Lite**（本地文件向量库，零部署、免 Docker）
- 向量召回 top-k 相关法条，拼进 Prompt 交给大模型
- 回答**强制标注引用【第X条】**；检索不到时明确回复"提供的法条中没有相关规定"（防幻觉）

## 流程图

```
法条文本 law.text
   ↓ ① 加载 + ② 切分（按"第X条"切，超长带重叠再切）
chunks: [{article, text}, ...]
   ↓ ③ 向量化（bge-m3，1024 维）
   ↓ ④ 写入 Milvus（id / vector / text / article / source）
                     ↓
用户问题 ──⑤ 向量化──→ Milvus 相似度检索（COSINE，top-k）
                     ↓
        ⑥ 拼接 Prompt（问题 + 检索到的法条）
                     ↓
        ⑦ DeepSeek 生成 ⑧ 带【第X条】引用的回答
```

## 目录结构

```
.
├── main.py            # 主流程：加载/切分/入库/检索/生成
├── agent.py           # 对话模型（DeepSeek，OpenAI 兼容）
├── 嵌入模型.py          # Embedding 模型（BAAI/bge-m3，OpenAI 兼容接口）
├── law.text           # 法条文本（示例数据）
├── requirements.txt
├── .env.example       # 环境变量模板
├── .gitignore
└── README.md
```

## 技术栈

| 环节 | 使用 |
|---|---|
| 文档切分 | Python 正则（按"第X条"）+ 定长重叠切分 |
| Embedding | BAAI/bge-m3（1024 维，OpenAI 兼容接口） |
| 向量数据库 | Milvus Lite（本地文件，pymilvus） |
| 生成模型 | DeepSeek Chat（LangChain `init_chat_model`） |
| 编排 | 纯 Python 流程（未使用 LangGraph） |

## 运行方法

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置密钥
cp .env.example .env      # 填入 LLM_API_KEY 与 EMBED_API_KEY

# 3. 运行
python main.py
```

## 运行效果（示例）

```
📄 文档切分为 7 个片段
✅ 入库完成：7 个片段

👤 用户： 转租需要经过出租人同意吗？
🔍 检索到的法条（含相似度）：
   [第四条] 分数=0.7141  第四条 承租人未经出租人同意转租的，出租人可以解除合同。
🤖 助手： 需要。
        依据：【第四条】明确规定，承租人未经出租人同意转租的，出租人可以解除合同。
```

## 已知不足与改进方向

1. **纯向量检索存在语义漂移**：例如问"承租人一直不交租金怎么办"时，
   "转租"条得分 0.85 排在"支付租金"条（0.42）之前，靠 top-k=3 才让正确条文进入上下文。
2. 计划改进：
   - **混合检索**：BM25 关键词 + 向量，按 RRF 融合，缓解专有名词/关键词召回失准
   - **Rerank 重排**：召回 top-10 后用 bge-reranker 精排取 top-3
   - **Query / Document 分离编码**：检索查询走 `embed_query`，文档入库走 `embed_documents`
   - **切分与 top-k 对比实验**：记录不同参数下的检索命中率
   - **建立评估集**：20 个"问题 + 期望条文"，统计 Recall@k / MRR
3. 数据说明：当前 `law.text` 为**示例文本**（仅用于演示流程），正式使用需替换为真实法条并标注来源。

## 免责声明

本项目中的法条内容为示例文本，仅用于技术流程演示，不构成任何法律意见。
