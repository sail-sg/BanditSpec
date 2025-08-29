from eagle_samd import Eagle
from llama import LlamaForCausalLM
from qwen import Qwen2ForCausalLM 
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from generate_utils import double_buffer_spec_generate, ucb_spec_generate_with_batch_mask, ucb_length_spec_generate
from generate_utils import vanilla_generate
import json
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from eagle_qwen import QwenEagle
import time

def format_llama(prompt_list):
    return [(
        "<|begin_of_text|>"
        f"<|start_header_id|>USER<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    ) for prompt in prompt_list]

target_path = "PATH_TO_LLaMA_TARGET"
eagle_path = "PATH_TO_LLaMA_EAGLE"

tokenizer = AutoTokenizer.from_pretrained(target_path)
tokenizer.pad_token = tokenizer.eos_token
with open(f"{eagle_path}/config.json", "r") as f:
    config = json.load(f)
config["sliding_window"] = False
draft_model = Eagle(LlamaConfig.from_dict(config))
draft_model.load_weight(eagle_path)
draft_model = draft_model.cuda().to(torch.bfloat16)
draft_model = torch.compile(draft_model)

target_model = LlamaForCausalLM.from_pretrained(target_path, torch_dtype=torch.bfloat16, device_map="auto")

for dataset in ['long_alpaca.jsonl', 'long_code.jsonl']:
    with open(dataset, "r") as f:
        prompts = [json.loads(line)["turns"][0] for line in f]

    import random
    random.seed(42)
    random.shuffle(prompts)

    cuda_prompts = []
    prompts_length = []

    bsz = 256
    if dataset == "long_code.jsonl":
        bsz_list = [1, 2, 4, 10, 20, 30, 40]
    else:
        bsz_list = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 250, 256]

    for index in range(0, len(prompts), bsz):
        format_prompt = format_llama(prompts[index:index+bsz])
        a = tokenizer(format_prompt, return_tensors="pt", padding=True, max_length=512, truncation=True, padding_side="right")
        a_cuda = a['input_ids'].cuda()
        prompts_length.append(a['attention_mask'].sum(dim=-1).cuda())
        cuda_prompts.append(a_cuda)

    print("we are finish loading.")
    print("bsz\tspec_quota\tgamma\tthroughput")

    with torch.inference_mode():
        for bsz in bsz_list:
            for gamma in [4]:
                for all_quota in [156, 256, 356, 512]:
                    if (gamma > -1) and (all_quota > 200):
                        break
                    counts = 0
                    wall_time = .0
                    nums = 0
                    draft_wall_time = .0
                    num_run = 0
                    for prompt, prompt_length in zip(cuda_prompts, prompts_length):
                        prompt = prompt[:bsz]
                        prompt_length = prompt_length[:bsz]
                        output_ids, count, num, elapsed_time, draft_time = \
                            ucb_length_spec_generate(draft_model, target_model, prompt, input_len=prompt_length, max_gen_len=256, eos_id=128009, pad_token=128001, gamma=gamma)
                        print(f"{(count + num) / elapsed_time}tokens/s")
                        for i in range(5):
                            _, tempc, tempnum, temptime, _ = double_buffer_spec_generate(
                                draft_model, target_model, prompt, input_len=prompt_length,
                                max_gen_len=256, eos_id=128009, pad_token=128001, gamma=i)
                            print(f"{i}\t{(tempc + tempnum) / temptime}tokens/s")
                        if num_run > 8:
                            break
                        else:
                            num_run += 1
                        if num_run > 1:
                            wall_time += elapsed_time
                            draft_wall_time += draft_time
                            nums += num
                            counts += count
                    try:
                        value = (nums + counts).item()
                    except:
                        value = (nums + counts)
                    print([bsz, all_quota, gamma, value / wall_time])

# 匿名路径替换
eagle_path = "/path/to/EAGLE-MODEL"
target_path = "/path/to/TARGET-MODEL"

tokenizer = AutoTokenizer.from_pretrained(target_path)
with open(f"{eagle_path}/config.json", "r") as f:
    config = json.load(f)
config["sliding_window"] = False
draft_model = QwenEagle(Qwen2Config.from_dict(config))
draft_model.load_weight(eagle_path)
draft_model = draft_model.cuda().to(torch.bfloat16)
draft_model = torch.compile(draft_model)
target_model = Qwen2ForCausalLM.from_pretrained(target_path, torch_dtype=torch.bfloat16, device_map="auto")
torch.set_printoptions(threshold=200)

def format_qwen(prompt_list):
    return [
        (
            f"<|im_start|>system\n"
            f"You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{prompt} Let's think step by step and give long output.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        for prompt in prompt_list
    ]

for dataset in ['code.jsonl', 'debug.jsonl', 'gsm8k.jsonl']:
    with open(dataset, "r") as f:
        prompts = [json.loads(line)["turns"][0] for line in f]

    random.seed(42)
    random.shuffle(prompts)

    cuda_prompts = []
    prompts_length = []

    bsz = 256
    for index in range(0, len(prompts), bsz):
        format_prompt = format_qwen(prompts[index:index+bsz])
        a = tokenizer(format_prompt, return_tensors="pt", padding=True, max_length=256, truncation=True, padding_side="right")
        a_cuda = a['input_ids'].cuda()
        prompts_length.append(a['attention_mask'].sum(dim=-1).cuda())
        cuda_prompts.append(a_cuda)

    print("we are finish loading.")
    print("bsz\tspec_quota\tgamma\tthroughput")

    with torch.inference_mode():
        for bsz in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 250, 256]:
            for gamma in [-1, 0, 1, 2, 3, 4]:
                for all_quota in [156, 256, 356, 512]:
                    if (gamma > -1) and (all_quota > 200):
                        break
                    counts = 0
                    wall_time = .0
                    nums = 0
                    draft_wall_time = .0
                    num_run = 0
                    for prompt, prompt_length in zip(cuda_prompts, prompts_length):
                        prompt = prompt[:bsz]
                        prompt_length = prompt_length[:bsz]
                        output_ids, count, num, elapsed_time, draft_time = \
                            ucb_spec_generate_with_batch_mask(draft_model, target_model, prompt, input_len=prompt_length,
                                                              max_gen_len=256, eos_id=151645, pad_token=151643,
                                                              gamma=gamma, all_quota=all_quota)
                        if num_run > 8:
                            break
                        else:
                            num_run += 1
                        if num_run > 1:
                            wall_time += elapsed_time
                            draft_wall_time += draft_time
                            nums += num
                            counts += count
                    try:
                        value = (nums + counts).item()
                    except:
                        value = (nums + counts)
                    result = [bsz, all_quota, gamma, value / wall_time]
                    print(result)
                    with open(f"results_{dataset.replace('.jsonl', '')}.txt", "a") as f:
                        f.write(f"{str(result)}\n")
                    time.sleep(30)
