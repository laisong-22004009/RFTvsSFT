import json
import os
import random
import re
from PIL import Image as PILImage
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.utils import random_uuid
from datasets import Dataset, load_dataset
from mathruler.grader import grade_answer
from huggingface_hub import login
import torch

# HuggingFace登录
login(token="your_token")  # 替换为你的token

# 配置
BATCH_SIZE = 24
SAMPLES_PER_ITEM = 8

model_path = "model_path"
final_output_dir = "final_data_dir"

os.makedirs(final_output_dir, exist_ok=True)

pattern = re.compile(r"<think>.*?</think>\s*\\boxed\{(.*?)\}", re.DOTALL)
placeholder = "<|image_pad|>"

# 加载数据集
print("Loading dataset from HuggingFace...")
dataset_dict = load_dataset("zhhxte/mllm_cl_pathvqa")
train_data = list(dataset_dict["train"])
test_data = list(dataset_dict["test"])

print(f"\nOriginal dataset: {len(train_data)} train, {len(test_data)} test")

# 为训练集添加索引
for idx, item in enumerate(train_data):
    item['global_index'] = idx

def build_batch_item(item):
    problem = item.get("problem", "")
    if not problem:
        return None, None
    
    answer = item.get("answer", "").strip()
    images = item.get("images", None)
    
    if images is None or len(images) == 0:
        return None, None
    
    image = images[0]
    
    if not isinstance(image, PILImage.Image):
        return None, None
    
    if image.height < 28 or image.width < 28:
        return None, None
    
    user_msg = f"{problem}\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{{}}."
    
    user_content = user_msg.replace("<image>", f"<|vision_start|>{placeholder}<|vision_end|>")
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
        "request_id": random_uuid(),
    }, {
        "problem": problem,
        "answer": answer,
        "global_index": item.get('global_index')
    }

def main():
    # 创建单个模型实例
    print("Loading model...")
    llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.7)
    
    results = []
    
    # 批处理所有训练数据
    print("Processing training data...")
    for start in tqdm(range(0, len(train_data), BATCH_SIZE), desc="Processing batches"):
        data_slice = train_data[start:start + BATCH_SIZE]
        batch, meta_list = [], []
        
        for item in data_slice:
            batch_item, meta = build_batch_item(item)
            if batch_item is None:
                results.append({
                    "global_index": item.get('global_index'),
                    "problem": item.get("problem", ""),
                    "answer": item.get("answer", ""),
                    "correct_count": 0,
                    "total_count": 0,
                    "responses": []
                })
            else:
                batch.append(batch_item)
                meta_list.append(meta)
        
        if not batch:
            continue
        
        # 推理
        params = SamplingParams(temperature=1.0, n=SAMPLES_PER_ITEM, max_tokens=1024)
        outputs = llm.generate(batch, params)
        
        # 处理结果
        for j, out_group in enumerate(outputs):
            meta = meta_list[j]
            correct_count = 0
            responses = []
            
            for out in out_group.outputs:
                reply = out.text.strip()
                match = pattern.search(reply)
                
                if match:
                    answer = match.group(1).strip()
                    is_correct = grade_answer(answer, meta["answer"])
                    if is_correct:
                        correct_count += 1
                    
                    responses.append({
                        "response": reply,
                        "extracted_answer": answer,
                        "is_correct": is_correct
                    })
                else:
                    responses.append({
                        "response": reply,
                        "extracted_answer": None,
                        "is_correct": False
                    })
            
            results.append({
                "global_index": meta["global_index"],
                "problem": meta["problem"],
                "answer": meta["answer"],
                "correct_count": correct_count,
                "total_count": SAMPLES_PER_ITEM,
                "responses": responses
            })
    
    # 清理模型
    del llm
    torch.cuda.empty_cache()
    
    print(f"\nTotal processed items: {len(results)}")
    
    # 分组：solvable (1-8/8) vs unsolvable (0/8)
    solvable = []
    unsolvable = []
    
    for item in results:
        correct_count = item["correct_count"]
        total_count = item["total_count"]
        
        if total_count == 0 or correct_count == 0:
            unsolvable.append(item)
        else:
            solvable.append(item)
    
    print(f"\nSolvable (1-8/8 correct): {len(solvable)}")
    print(f"Unsolvable (0/8 correct): {len(unsolvable)}")
    
    # 计算max_sample
    max_sample = min(len(solvable), len(unsolvable), 7500)
    print(f"\nMax sample size: {max_sample}")
    
    # 随机采样
    random.seed(42)
    solvable_sampled = random.sample(solvable, max_sample)
    unsolvable_sampled = random.sample(unsolvable, max_sample)
    
    print(f"Sampled solvable: {len(solvable_sampled)}")
    print(f"Sampled unsolvable: {len(unsolvable_sampled)}")
    
    # 转换为Dataset
    def convert_to_dataset(items, original_data):
        if not items:
            return None
        
        data_dict = {
            "images": [],
            "problem": [],
            "answer": [],
            "correct_count": [],
            "total_count": [],
            "responses": []
        }
        
        for item in items:
            global_idx = item.get("global_index")
            if global_idx is not None and global_idx < len(original_data):
                images = original_data[global_idx].get("images")
                image = images[0] if images and len(images) > 0 else None
            else:
                image = None
            
            data_dict["images"].append([image] if image is not None else [])
            data_dict["problem"].append(item.get("problem", ""))
            data_dict["answer"].append(item.get("answer", ""))
            data_dict["correct_count"].append(item["correct_count"])
            data_dict["total_count"].append(item["total_count"])
            data_dict["responses"].append(json.dumps(item["responses"], ensure_ascii=False))
        
        return Dataset.from_dict(data_dict)
    
    # 准备test集
    def prepare_test_dataset(test_data):
        data_dict = {
            "images": [],
            "problem": [],
            "answer": [],
            "correct_count": [],
            "total_count": [],
            "responses": []
        }
        
        for item in test_data:
            images = item.get("images")
            image = images[0] if images and len(images) > 0 else None
            
            data_dict["images"].append([image] if image is not None else [])
            data_dict["problem"].append(item.get("problem", ""))
            data_dict["answer"].append(item.get("answer", ""))
            data_dict["correct_count"].append(-1)  # 标记为未推理
            data_dict["total_count"].append(-1)
            data_dict["responses"].append("[]")
        
        return Dataset.from_dict(data_dict)
    
    # 准备random数据集
    def prepare_random_dataset(train_data, max_sample):
        random.seed(42)
        sampled_data = random.sample(train_data, min(max_sample, len(train_data)))
        
        data_dict = {
            "images": [],
            "problem": [],
            "answer": [],
            "correct_count": [],
            "total_count": [],
            "responses": []
        }
        
        for item in sampled_data:
            images = item.get("images")
            image = images[0] if images and len(images) > 0 else None
            
            data_dict["images"].append([image] if image is not None else [])
            data_dict["problem"].append(item.get("problem", ""))
            data_dict["answer"].append(item.get("answer", ""))
            data_dict["correct_count"].append(-1)  # 标记为未推理
            data_dict["total_count"].append(-1)
            data_dict["responses"].append("[]")
        
        return Dataset.from_dict(data_dict)
    
    test_dataset = prepare_test_dataset(test_data)
    print(f"\nTest set: {len(test_dataset)} samples")
    
    # 准备random训练集
    random_train_dataset = prepare_random_dataset(train_data, max_sample)
    print(f"Random train set: {len(random_train_dataset)} samples")
    
    # 保存三个数据集到本地
    datasets_to_save = [
        ("mllm_cl_pathvqa_solvable", solvable_sampled),
        ("mllm_cl_pathvqa_unsolvable", unsolvable_sampled),
        ("mllm_cl_pathvqa_random", None)
    ]
    
    for dataset_name, train_items in datasets_to_save:
        save_path = os.path.join(final_output_dir, dataset_name)
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\nSaving to {save_path}...")
        
        if dataset_name == "mllm_cl_pathvqa_random":
            print(f"  Train: {len(random_train_dataset)} samples")
            print(f"  Test: {len(test_dataset)} samples")
            
            random_train_dataset.to_parquet(os.path.join(save_path, "train.parquet"))
            test_dataset.to_parquet(os.path.join(save_path, "test.parquet"))
        else:
            if train_items:
                train_ds = convert_to_dataset(train_items, train_data)
                if train_ds:
                    print(f"  Train: {len(train_ds)} samples")
                    print(f"  Test: {len(test_dataset)} samples")
                    
                    train_ds.to_parquet(os.path.join(save_path, "train.parquet"))
                    test_dataset.to_parquet(os.path.join(save_path, "test.parquet"))
        
        print(f"{dataset_name} saved successfully!")
    
    print("\nAll datasets saved to local directory!")
    print(f"Location: {final_output_dir}")
    print(f"\nSummary:")
    print(f"  mllm_cl_pathvqa_solvable: {max_sample} train samples")
    print(f"  mllm_cl_pathvqa_unsolvable: {max_sample} train samples")
    print(f"  mllm_cl_pathvqa_random: {min(max_sample, len(train_data))} train samples")
    print(f"  All datasets share the same test set: {len(test_dataset)} samples")

if __name__ == "__main__":
    main()
