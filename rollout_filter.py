import json
import os
import random
import re
from PIL import Image as PILImage
from tqdm import tqdm
from multiprocessing import Process
from vllm import LLM, SamplingParams
from vllm.utils import random_uuid
from datasets import Dataset, DatasetDict, Features, Value, Image
from mathruler.grader import grade_answer
import sys
import torch

# ====== 配置 ======
NUM_PROCESSES = 8
BATCH_SIZE = 24
SAMPLES_PER_ROUND = 8
MAX_NUM_REPLIES = 1
MAX_ROUNDS = 1

# 模型与路径设置
model_path = "path/to/your/model" 

json_path = "path/to/your/data" 
image_root = "path/to/your/images" 

final_output_dir_pass = "path/to/your/output"  # 替换为实际输出路径
temp_output_dir = "path/to/temp/output"  # 替换为实际临时输出路径

os.makedirs(temp_output_dir, exist_ok=True)

pattern = re.compile(r"<think>.*?</think>\s*\\boxed\{(.*?)\}", re.DOTALL)
placeholder = "<|image_pad|>"


with open(json_path, "r") as f:
    data = json.load(f)

chunk_size = len(data) // NUM_PROCESSES
chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(NUM_PROCESSES)]
if len(data) % NUM_PROCESSES:
    chunks[-1].extend(data[NUM_PROCESSES * chunk_size:])

# 预加载图片
all_image_paths = {os.path.join(image_root, img) for item in data for img in item.get("images", [])}
image_cache = {}
for path in tqdm(all_image_paths, desc="预加载图片"):
    try:
        image_cache[path] = PILImage.open(path).convert("RGB")
    except:
        image_cache[path] = None

# ====== 子进程函数 ======
def worker_fn(idx, data_chunk, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # make sure to set enough max_model_len if needed !!
    # llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.96, max_model_len=8192)
    llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.85)
    results = []
    skipped_items = []

    def build_batch(data_slice):
        batch, meta = [], []
        for item in data_slice:
            messages = item.get("messages", [])
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
            if not user_msg:
                meta.append(None)
                continue
            gt = messages[-1]["content"].strip()
            answer_msg = f"The answer of this question is {gt}"
            user_msg = user_msg + '\n' + answer_msg + " You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
            limit_msg=f"1. Give out coresponding reasoning process and final answer in the format of <think> </think> and \\boxed{{}}. \n2. Your reasoning should be as natural, detailed, and logical as normal problem solving. Do not rely on or refer to the standard answer, which is only used as a reference for you to check the correctness of your reasoning.\n3. The final answer must be accurate and consistent with the standard answer, but do not make the reader feel that you know the answer in advance."
            user_msg = user_msg + "\n\n" + limit_msg

            image_paths = item.get("images", [])
            if not image_paths:
                meta.append(None)
                continue

            image_path = os.path.join(image_root, image_paths[0])
            image = image_cache.get(image_path, None)
            if image is None or image.height < 28 or image.width < 28:
                meta.append(None)
                continue

            label = next((m["content"] for m in messages if m["role"] == "assistant"), None)
            user_content = user_msg.replace("<image>", f"<|vision_start|>{placeholder}<|vision_end|>")
            prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            batch.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image},
                "request_id": random_uuid(),
            })
            meta.append({
                "user_msg": user_msg,
                "label": label,
                "image_paths": image_paths,
                "item": item
            })
        return batch, meta

    #for start in range(0, len(data_chunk), BATCH_SIZE):
    if idx == 0:
        loop_iter = tqdm(
            range(0, len(data_chunk), BATCH_SIZE),
            desc=f"[GPU {gpu_id}] Worker {idx}",
            dynamic_ncols=True
        )
    else:
        loop_iter = range(0, len(data_chunk), BATCH_SIZE)

    for start in loop_iter:
        data_slice = data_chunk[start:start + BATCH_SIZE]
        collected_list = [[] for _ in data_slice]
        meta_list = [None for _ in data_slice]

        for round_idx in range(MAX_ROUNDS):
            indices = [i for i, r in enumerate(collected_list) if len(r) < MAX_NUM_REPLIES]
            if not indices:
                break
            sub_slice = [data_slice[i] for i in indices]
            batch, meta = build_batch(sub_slice)
            if not batch:
                continue
            params = SamplingParams(temperature=1.0, n=SAMPLES_PER_ROUND * (2 ** round_idx), max_tokens=1024)
            outputs = llm.generate(batch, params)

            for j, out_group in enumerate(outputs):
                i = indices[j]
                info = meta[j]
                if info is None:
                    continue
                for out in out_group.outputs:
                    if len(collected_list[i]) >= MAX_NUM_REPLIES:
                        break
                    reply = out.text.strip()
                    match = pattern.search(reply)
                    if not match:
                        continue
                    answer = match.group(1).strip()
                    # 检查答案是否正确
                    is_correct = False
                    if info["label"]:
                        is_correct = grade_answer(answer, info["label"].strip())
                    
                    # 只有答案正确时才添加到结果中
                    if is_correct:
                        collected_list[i].append({
                            "messages": [
                                {"role": "user", "content": info["user_msg"]},
                                {"role": "assistant", "content": reply}
                            ],
                            "images": info["image_paths"]
                        })
                meta_list[i] = info

        # 只保存有正确答案的数据项
        for i, replies in enumerate(collected_list):
            if replies:  # 只有当这个数据项有至少一个正确回答时才保存
                results.extend(replies)
            else:
                skipped_items.append(data_slice[i])

    with open(os.path.join(temp_output_dir, f"results_{idx}.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(temp_output_dir, f"skipped_{idx}.json"), "w") as f:
        json.dump(skipped_items, f, indent=2, ensure_ascii=False)
    
    del llm
    torch.cuda.empty_cache()

# ====== 主控制函数 ======
def process_for_parquet(vqa_data):
    images, problems, answers = [], [], []
    for example in vqa_data:
        problems.append(example["messages"][0]["content"])
        answers.append(example["messages"][1]["content"])
        if "images" in example:
            img_path = example["images"][0]
            if img_path in image_cache:
                images.append(image_cache.get(img_path, None))
            else:
                img_path = os.path.join(image_root, example["images"][0])
                images.append(image_cache.get(img_path, None))
        else:
            images.append(None)
    return {"images": images, "problem": problems, "answer": answers}

def main():
    processes = []
    for i in range(NUM_PROCESSES):
        p = Process(target=worker_fn, args=(i, chunks[i], i))  # 每个进程绑到第 i 张 GPU
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        print(f"Process {p.pid} exit code: {p.exitcode}")
        
    print("✅ 所有子进程已 join，开始合并结果...", flush=True)
    all_results, all_skipped = [], []
    for i in range(NUM_PROCESSES):
        with open(os.path.join(temp_output_dir, f"results_{i}.json")) as f:
            all_results += json.load(f)
        with open(os.path.join(temp_output_dir, f"skipped_{i}.json")) as f:
            all_skipped += json.load(f)
    
    print(f"✅ 合并完成，共 {len(all_results)} 条通过（能做对的数据），{len(all_skipped)} 条跳过（一次都做不对的数据）")
    print("✅ 所有结果合并完毕，开始写入过滤后的数据集", flush=True)
    with open(final_output_dir_pass, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✅ 过滤后的数据集已保存到: {final_output_dir_pass}", flush=True)
    print(f"✅ 跳过的数据项保存在: {temp_output_dir}/skipped_*.json", flush=True)

if __name__ == "__main__":
    main()

