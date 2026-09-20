# 覆盖 RAG 全流程：
#   ① 加载文档  ② 切分(chunk)  ③ 向量化(embedding)  ④ 入库(Milvus)
#   ⑤ 检索(retrieve)  ⑥ 拼 Prompt  ⑦ 模型生成  ⑧ 输出带引用的回答


import os
import re
from dotenv import load_dotenv
from pymilvus import MilvusClient
from requests_toolbelt.adapters import source

from embedding import embedding_model

load_dotenv(override=True)

#1.配置
MILVUS_URI = "./milvus_legal.db"
COLLECTION = "legal_articles"
TOP_K = 3
DIM = 1024 #用的嵌入模型转换为向量的维度
CHUNK_SIZE = 200      # 每个片段的最大字符数
CHUNK_OVERLAP = 50    # 相邻片段重叠的字符数

SYSTEM_PROMPT = (
    "你是一个法律条文检索助手。只能依据下面提供的【检索到的法条】回答问题，"
    "回答时逐条标注引用来源，格式如【第一条】；"
    "如果检索到的法条不足以回答，就明确说'提供的法条中没有相关规定'，不要编造。"
    "回答简洁，先给结论再列依据。"
)
def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedding_model.embed_documents(texts)#直接调用嵌入模型文件的函数

#加载文档
SAMPLE_LAW = """
第一条 本示例文本仅用于演示检索流程，不构成任何真实法律意见。
第二条 承租人应当按照合同约定的期限和金额支付租金。承租人无正当理由未支付租金的，出租人可以要求其在合理期限内支付。
第三条 出租人应当保证租赁物符合约定的用途，并在租赁期间保持租赁物符合约定用途。
第四条 承租人未经出租人同意转租的，出租人可以解除合同。
第五条 租赁期限不得超过二十年。超过二十年的，超过部分无效。
第六条 因不可抗力致使合同无法履行的，双方可以协商解除合同，并互不承担违约责任。
第七条 合同解除后，双方应当及时结算租金、押金以及其他费用。
"""

def load_documents() -> str:
    path = "law.text"
    if os.path.exists(path):#exists是存在的意思，这句话是在判断，这个路径对应的东西存不存在
        with open(path, encoding="utf-8") as f:
            return f.read()
    return SAMPLE_LAW#不存在就返回上面定义的SAMPLE_LAW


#切分
def split_articles(text:str)->list[dict]:
    parts = re.split(r"(?=第[一二三四五六七八九十百零〇\d]+条)", text.strip())
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"(第[一二三四五六七八九十百零〇\d]+条)", part)
        article = m.group(1) if m else "未知条文"

        if len(part) <= CHUNK_SIZE:
            chunks.append({"article": article, "text": part})
        else:
            start = 0
            while start < len(part):
                piece = part[start:start + CHUNK_SIZE]
                chunks.append({"article": article, "text": piece})
                start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def build_index(chunks: list[dict]) -> MilvusClient:
    """建集合 → 向量化 → 写入 Milvus（重复运行会先删库重建，幂等）。"""
    client = MilvusClient(uri=MILVUS_URI)

    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)          # 重新建立索引：先删旧的

    client.create_collection(
        collection_name=COLLECTION,
        dimension=DIM,
        metric_type="COSINE",                       # 余弦相似度（文本检索常用）
    )

    # 第 1 步：从每个片段里只抽出"正文"，组成一个纯文本列表
    texts=[]
    for c in chunks:
        texts.append(c["text"])
    # 第 2 步：把这 7 句话一起向量化 → 得到 7 串数字
    vectors = embed_texts(texts)
    # 第 3 步：把"片段"和"它的向量"配对，打包成 Milvus 要的记录
    data=[]
    for i in range(len(chunks)):#i=0,1,2,3,4,5,6
        row={
            "id":i,
            "vector": vectors[i],#第i行对应向量
            "text": chunks[i]["text"],#第i行对应文本
            "article": chunks[i]["article"],#第i行对应条文
            "source": "示例法条",
        }
        data.append(row)

    # vectors = embed_texts([c["text"] for c in chunks])   # ③ 向量化
    # data = [
    #     {
    #         "id": i,
    #         "vector": vectors[i],
    #         "text": chunks[i]["text"],
    #         "article": chunks[i]["article"],
    #         "source": "示例法条",
    #     }
    #     for i in range(len(chunks))
    # ]
    client.insert(collection_name=COLLECTION, data=data)  # ④ 入库
    client.flush(COLLECTION)                              # 确保可立即检索
    print(f"✅ 入库完成：{len(data)} 个片段")
    return client
# ======================= ⑤ 检索 =======================
def retrieve(client: MilvusClient, question: str, top_k: int = TOP_K) -> list[dict]:
    """把问题向量化，去 Milvus 找最相似的 top_k 条法条。"""
    q_vector = embed_texts([question])[0]
    res = client.search(
        collection_name=COLLECTION,
        data=[q_vector],
        limit=top_k,
        output_fields=["text", "article", "source"],
    )
    hits = []
    for hit in res[0]:
        entity = hit.get("entity", {})
        hits.append({
            "article": entity.get("article", "未知条文"),
            "text": entity.get("text", ""),
            "score": hit.get("distance", 0.0),       # 相似度分数（越大越像）
        })
    return hits

# ======================= ⑦ 模型生成 =======================
from agent import chat_model



def answer(question: str, hits: list[dict]) -> str:
    """⑥ 拼 Prompt（问题 + 检索到的法条）→ ⑦ 让模型基于法条回答。"""
    context = "\n\n".join(f"[{h['article']}] {h['text']}" for h in hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【检索到的法条】\n{context}\n\n【用户问题】\n{question}"},
    ]
    return chat_model.invoke(messages).content          # ⑧ 带引用的回答

# ======================= main =======================
def main():
    # ①②切分
    text = load_documents()
    chunks = split_articles(text)
    print(f"📄 文档切分为 {len(chunks)} 个片段")

    # ③④ 建索引入库
    client = build_index(chunks)


    questions = [
        "承租人一直不交租金，出租人可以怎么做？",
        "转租需要经过出租人同意吗？",
        "租赁合同最多能签多少年？",
    ]

    for q in questions:
        print("\n" + "=" * 60)
        print("👤 用户：", q)

        hits = retrieve(client, q)               # ⑤ 检索
        print("🔍 检索到的法条（含相似度）：")
        for h in hits:
            print(f"   [{h['article']}] 分数={h['score']:.4f}  {h['text'][:40]}...")

        reply = answer(q, hits)             # ⑥⑦⑧ 生成
        print("🤖 助手：", reply)


if __name__ == "__main__":
    main()



