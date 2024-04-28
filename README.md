# Infini-Attention

[![GitHub Stars](https://img.shields.io/github/stars/jlamprou/Infini-Attention?style=social)](https://github.com/jlamprou/Infini-Attention/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/jlamprou/Infini-Attention)](https://github.com/jlamprou/Infini-Attention/issues)
[![License](https://img.shields.io/github/license/jlamprou/Infini-Attention)](https://github.com/jlamprou/Infini-Attention/blob/main/LICENSE)

![image](https://github.com/jlamprou/Infini-Attention/assets/41962910/f66fd556-e1d2-4ccc-89e9-f4812244b8a2)

**Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention Pytorch Implementation** ([Paper](https://arxiv.org/abs/2404.07143))

This repository provides a PyTorch implementation of the Infi-Attention mechanism, following the paper step-by-step. The code is written using the HuggingFace template for seamless integration with their ecosystem.

## Why this is NOT the golden solution 
- [Yannic's Kilcher comprehensive explanation](https://youtu.be/r_UBBfTPcF0?si=4Y6dyRuk0ZcpvGW_)
- This work introduces some neat tricks to replace the global softmax with kernels, but if you check the related work this is not a new concept at all, lot's of research has been done around linear attention which seemed promising but failed in real-world scenarios.
- This is definitely not production-ready research or code.
- The compressive memory scheme is an idea based on the human brain. The context of the compressive memory is "blurry" like the human brain. The problem is that this memory is not learnable. That means that the model doesn't know which information should be kept with higher quality.
- Further research would be interesting with a learnable compressive memory scheme, here are some ideas:
  - Differentiable Neural Computer (DNC):
The DNC, introduced by DeepMind, is a memory-augmented neural network that combines a controller network with an external memory matrix. The controller interacts with the memory using differentiable read and write operations. The DNC has shown promising results in tasks requiring long-term memory and complex reasoning.
  - Turing Machines and Neural Turing Machines (NTMs):
Turing Machines are abstract computational models that can simulate any algorithm. NTMs, introduced by Google DeepMind, incorporate Turing Machine-like properties into neural networks. NTMs have an external memory matrix and a controller that learns to read from and write to the memory using attention mechanisms.

## Latest Updates:
- Updated InfiniAttention module:
  - Changed memory M_z to a buffer for improved efficiency
  - Implemented reset_memory() function to zero out M and z tensors after processing all segments
  - Eliminated the need for detach() tricks and argument tricks, simplifying the codebase
- Introduced new SegmentedDataset class:
  - Handles data segmentation within the dataset itself
  - Optimizes time and memory usage during the training process
- Added passkey retrieval finetuning and testing script, with this script we can actually evaluate our implementation with a 1M Keypass Retrieval like the paper. We need at least 1x80GB GPU, once that is available we cant test.


## TODO
- [ ] 1M Passkey Retrieval Finetuning and benchmark, we have to finetune and benchmark the Qwen1.5 model to check the performance of the implementation. (My 2xA100 server is on maintenance for a few days) 
- [ ] Triton/CUDA optimized implementation of the memory ops
- [x] New Dataset class that takes care of the segmentation
- [ ] New Huggingface Trainer Class that trains segment-wise

## Features

- PyTorch implementation of Infi-Attention
- Follows the paper's methodology closely
- Utilizes the HuggingFace template for easy integration

## Why Qwen1.5MoE?
- MoE : The model has 14.3B parameters in total and 2.7B activated parameters during runtime.
- Simple and modifiable architecture.
- Great Benchmark Perfomance: The model achieves comparable perfomance with bigger LLMs with only 2.2B paramaters on activation. 

   **Using this model we can test the benchmark perfomance compared to other big LLMs (llama, Mixtral etc.) without the need of huge resources. The pre-trained model is a great candidate to test in long-contexts.**

## Current Limitations and Areas for Improvement

1. **Segment-wise Attention**: The paper does not provide clear guidance on when the segmentation should occur during the model training process. Two potential theories are being explored:
  - Theory 1: Segment the input within the attention class and perform in-class operations using a loop for each segment. However, this approach does not result in significant memory savings compared to classic SDPA.
  - Theory 2: Segment the input during the training loop and feed each segment to the model with gradient accumulation steps equal to Sequence Length / Segment Length.

#### ***We Implemented theory 2 and it seems to work!!! I tested the accuracy at every batch using the concat of logits and labels to check if the accuarcy on the total sequence length is improving during training and once the learnable beta got some data we got the same accuracy rate with normal SDPA attention.***




## Run Qwen CLM pre-training:
I don't know for sure that this is the right way to segment!!!

I have modified the [HuggingFace run_clm_no_trainer.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py) to segment each batch and perform N gradient accumulation steps for N segments. 

Training Script Example with [Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B) pretraining on [PG-19](https://huggingface.co/datasets/pg19) at 32K context and 2k segments:

```bash
python run_clm.py 
--model_name_or_path Qwen/Qwen1.5-MoE-A2.7B
--block_size 32768 
--per_device_train_batch_size 1
--tokenizer_name Qwen/Qwen1.5-MoE-A2.7B 
--num_train_epochs 10 
--dataset_name pg19 
--learning_rate 0.01 
--segment_length 2048
```

## Run Qwen CLM keypass retrieval
This script creates random keypass retrieval tasks and finetunes the model using LoRa. 

To run with 32k context:
```bash
python passkey_retrieval.py
--num_tokens 32000
```

## Contributing

Contributions to this project are welcome! If you have any ideas, suggestions, or would like to discuss the implementation, please feel free to open an issue on the [GitHub repository](https://github.com/jlamprou/Infi-Attention). Your feedback and contributions can help improve the Infi-Attention implementation.

## License

This project is licensed under the MIT License.

---

🌟 Give this project a star on [GitHub](https://github.com/jlamprou/Infini-Attention) to show your support!
