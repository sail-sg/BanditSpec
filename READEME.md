````markdown
# BanditSpec: Bandit-Based Speculative Decoding for Efficient Autoregressive Generation

This repository implements **BanditSpec**, a speculative decoding framework that adaptively balances exploration and exploitation using bandit algorithms to accelerate autoregressive generation in large language models (LLMs). The framework is compatible with both LLaMA and Qwen2 architectures.

## 🧠 Key Components

- `eagle_samd.py`: Defines the Eagle (Li Y. et al. 2024 ) draft model based on LLaMA.
- `eagle_qwen.py`: Defines the Eagle (Li Y. et al. 2024 ) draft model based on Qwen2.
- `llama.py`, `qwen.py`: Customized versions of LLaMA and Qwen2 architectures.
- `generate_utils.py`: Implements core decoding strategies including BanditSpec.
- `inference_length.py`: Main script to run throughput benchmarking across different batch sizes and strategies.
- `llama_long.png`: Visualization of throughput improvement comparisons.

## 🔧 Setup

### Install Dependencies

```bash
pip install torch transformers fairscale flash-attn tqdm
````

> ⚠️ Make sure `flash-attn` is compiled for your CUDA and PyTorch version.

> Download EAGLE models from their repo (https://github.com/SafeAILab/EAGLE)

### Folder Structure

```
project/
├── inference_length.py
├── eagle_samd.py
├── eagle_qwen.py
├── llama.py
├── qwen.py
├── generate_utils.py
├── llama_long.png
├── llama_model/       # contains config.json and pytorch_model.bin for LLaMA
└── eagle_model/       # contains config.json and pytorch_model.bin for Eagle
```

Modify `inference_length.py` to set:

```python
target_path = "llama_model"
eagle_path = "eagle_model"
```

## 🚀 Running BanditSpec

```bash
python inference_length.py
```

This will run decoding experiments across:

* Different batch sizes
* Various `gamma` values
* Baselines like `Best Arm`, `Worst Arm`, and fixed `gamma`

## 📊 Output Format

```text
bsz	spec_quota	gamma	throughput
10	256	        BanditSpec	1.43
20	256	        gamma=1	    1.61
...
```

## Reference
Li Y, Wei F, Zhang C, et al. Eagle: Speculative sampling requires rethinking feature uncertainty[J]. arXiv preprint arXiv:2401.15077, 2024.


