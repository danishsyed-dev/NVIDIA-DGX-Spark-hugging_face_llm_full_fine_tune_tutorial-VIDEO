# 🍕 Fully Fine-Tune a Small Language Model (Gemma 3 270M)

A tutorial on how to fully fine-tune Google's Gemma 3 270M model using Hugging Face libraries to extract food and drink items from text.

## 📖 Overview

This project demonstrates **Supervised Fine-Tuning (SFT)** of a Small Language Model (SLM) for a specific task: extracting food and drink items from text. The fine-tuned model can process text inputs and return structured data about food/drink content.

### Why Fine-tune a Small Language Model?

- ✅ **Own the model** - Run anywhere without API costs
- ✅ **Simple tasks work well** with smaller models
- ✅ **No API calls needed** - Run offline
- ✅ **Batch processing** - Much faster than API calls
- ✅ **Task-specific optimization** - Better performance on your use case

## 🎯 What We're Building

A model that extracts food and drink items from text, returning structured output:

**Input:**
```
A plate of rice cakes, salmon, cottage cheese and small cherry tomatoes with a cup of tea.
```

**Output:**
```
food_or_drink: 1
tags: fi
foods: rice cakes, salmon, cottage cheese, cherry tomatoes
drinks: cup of tea
```

## 🛠️ Technologies Used

- **Model**: [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m-it)
- **Dataset**: [FoodExtract-1k](https://huggingface.co/datasets/mrdbourke/FoodExtract-1k)
- **Libraries**: 
  - `transformers` - Model loading and inference
  - `trl` - Transformers Reinforcement Learning (SFT)
  - `datasets` - Data loading
  - `accelerate` - Training acceleration
  - `gradio` - Interactive demo

## 📋 Requirements

- Python 3.8+
- GPU with at least 16GB VRAM (Google Colab T4 works!)
- Hugging Face account (for uploading model)

## 🚀 Quick Start

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrdbourke/learn-huggingface/blob/main/notebooks/hugging_face_llm_full_fine_tune_tutorial.ipynb)

### 2. Enable GPU Runtime
- Go to `Runtime` → `Change runtime type` → Select `GPU`

### 3. Run All Cells
The notebook will:
1. Install dependencies
2. Load the base model
3. Prepare the dataset
4. Fine-tune (3 epochs, ~18 minutes)
5. Evaluate and create demo

## 📊 Training Results

After 3 epochs of training:

| Epoch | Training Loss | Validation Loss | Token Accuracy |
|-------|--------------|-----------------|----------------|
| 1 | 2.17 | 2.24 | 58.8% |
| 2 | 1.25 | 2.28 | 58.9% |
| 3 | 1.07 | 2.46 | 58.6% |

## 🔑 Key Concepts

### Full Fine-Tuning vs LORA
- **Full Fine-Tuning**: All model weights updated (used here)
- **LORA**: Only adapter weights trained (less resources needed)

### SLM (Small Language Model)
- Models under 1B parameters
- Great for specific tasks
- Can be tailored for your use case

### Tokens In, Tokens Out
- Think of any problem as: *What tokens do I want in, and what tokens do I want out?*

## 📁 Project Structure

```
├── NVIDIA-DGX-Spark-hugging_face_llm_full_fine_tune_tutorial-VIDEO.ipynb
├── README.md
├── .gitignore
└── Fully Fine-Tune a SLM-Gemma 3 270M.pdf (reference)
```

## 🎥 Video Tutorial

[Watch the full video walkthrough](https://youtu.be/2hoNAr-id-E)

## 📚 Resources

- [Original Notebook Source](https://github.com/mrdbourke/learn-huggingface/blob/main/notebooks/hugging_face_llm_full_fine_tune_tutorial.ipynb)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [TRL Documentation](https://huggingface.co/docs/trl/en/index)
- [Gemma 3 270M Model](https://huggingface.co/google/gemma-3-270m-it)
- [FoodExtract-1k Dataset](https://huggingface.co/datasets/mrdbourke/FoodExtract-1k)

## 🏷️ Tags Dictionary

The model assigns these tags to text:

| Tag | Meaning |
|-----|---------|
| np | Nutrition Panel |
| il | Ingredient List |
| me | Menu |
| re | Recipe |
| fi | Food Items |
| di | Drink Items |
| fa | Food Advertisement |
| fp | Food Packaging |

## 📝 License

This project is for educational purposes. Please refer to the original sources for licensing information.

## 👤 Author

Created following the tutorial by [Daniel Bourke](https://github.com/mrdbourke)

---

**Note**: The model weights are not included in this repository. You can either:
1. Run the notebook to create your own fine-tuned model
2. Download from Hugging Face Hub if available
