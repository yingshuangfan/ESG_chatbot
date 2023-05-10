import rocketqa
import time
from utils.summarize import inference_long, inference

query = "reduce emissions"  # topic query

with open("dataset/HSBC_HK_ESG_2022.txt", encoding="utf8") as fp:
    para_list = fp.readlines()
query_list = [query for _ in range(len(para_list))]   # query_list.len == para_list.len

# init dual encoder
MODEL = "pair_marco_de"
dual_encoder = rocketqa.load_model(model=MODEL, use_cuda=False, device_id=0, batch_size=16)

# encode query & para
# start = time.time()
q_embs = dual_encoder.encode_query(query=query_list)
p_embs = dual_encoder.encode_para(para=para_list)
# print(f"encode time: {time.time() - start}")

# compute dot product of query representation and para representation
start = time.time()
dot_products = dual_encoder.matching(query=query_list, para=para_list)
# print(f"match time: {time.time() - start}")

# retrive the TOP-N related sentences
top_n = 3
start = time.time()
scores = [(i, s) for i, s in enumerate(dot_products)]
print(f"score-loader time: {time.time() - start}")

start = time.time()
assert len(para_list) == len(scores)
scores.sort(key=lambda item: item[1], reverse=True)
top_scores = scores[:top_n]
top_para = []
for i, s in top_scores:
    print(f"score: {s}, paragraph: {para_list[i]}")
    top_para.append(para_list[i])
print(f"sort time: {time.time() - start}")

start = time.time()
# summarization
summary = inference(" ".join(top_para))
print(summary)
print(f"summarize time: {time.time() - start}")
