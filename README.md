# Cross-lingual Language Detoxification

This project explores **cross-lingual detoxification** using parameter-efficient fine-tuning of multilingual language models. The system is designed to rewrite toxic input sentences into neutral and polite expressions, even in **unseen target languages** such as Chinese and Spanish. We adopt a fine-tune-then-infer approach using the `mT0-large` model and design a multi-stage reranking pipeline to enhance generation quality.

---

## 🔧 Project Structure

- `fine-tuning-mt0large.ipynb`: Fine-tuning notebook for `mT0-large` using detox parallel datasets (English, Russian, Ukrainian) with **LoRA adapters**.
- `inference.ipynb`: Inference notebook using beam search + reranking on zero-shot test data (e.g., Chinese, Spanish).
- `baseline-delete.ipynb`: Implements a rule-based baseline that deletes toxic words from input text.
- `baseline-backtranslation.ipynb`: Implements a backtranslation-based detoxification baseline using machine translation to paraphrase input sentences.
- `mt0l-lora-adapter-largelearning_low/`: Contains LoRA adapter weights for the fine-tuned `mT0-large` model. Full checkpoints (e.g., optimizer states) are excluded due to GitHub file size limits.

---

## 🚀 Key Features

- **Multilingual fine-tuning** with LoRA on mT0/mT5 models.
- **Zero-shot generalization** to languages not seen during training.
- **Reranking pipeline** combining:
  - Toxic word filtering (toxic lexicon list)
  - Toxicity classifier scores (`textdetox/xlmr-large-toxicity-classifier`)
  - Sentence embedding similarity (semantic preservation,using `LaBSE` embedding)

---

## 📊 Results Summary

|      model      | Language |   STA  |  SIM  | XCOMET | J (Joint Score) |
|-----------------|----------|--------|-------|--------|-----------------|
| mt0-large       |  Spanish |  0.844 | 0.875 | 0.884  |      0.660      | 
| baseline-delete |  Spanish |  0.737 | 0.889 | 0.877  |      0.578      |
| mt0-large       |  Chinese |  0.730 | 0.835 | 0.821  |      0.500      | 
| baseline-delete |  Chinese |  0.836 | 0.823 | 0.747  |      0.523      |

- Spanish: model outperformed the baseline in Joint score and achieved significantly higher scores in style transfer accuracy 
- Chinese: Lower Joint Score than the baseline, better performance in content preservation and fluency. The model struggle with typologically different language with different forms of toxicity 


## 📁 Data & Training

- Fine-tuning used 38197 parallel detox pairs (EN, RU, UK). (5% as development set)
- Target format:  
  ```
  {"source": "<prefix> <toxic sentence>", "target": "<neutral rewrite>"}
  ```
  prefix: Rewrite the sentence by replacing toxic or offensive words with neutral and polite expressions. Preserve the original meaning.
- Model: [`bigscience/mt0-large`](https://huggingface.co/bigscience/mt0-large) + LoRA  
  - LoRA config: `r=32, alpha=64, dropout=0.05`
- Optimizer: AdamW, 3 epochs, cosine scheduler

---

## 🔍 Inference Process

1. Load fine-tuned model with adapter weights.
2. Generate candidates via **beam search** (`num_beams=10`).
3. Apply reranking:
   - Remove candidates with toxic words.
   - Rank by toxicity classifier.
   - Rerank by semantic similarity to input.

---

## 📦 Requirements

```bash
transformers
peft
datasets
sentence-transformers==2.6.1
scipy==1.14.1
torch
pandas
tqdm
```

---

## 📌 Notes

- GitHub large file limits prevent uploading full checkpoints (>100MB `.pt` files).
- Evaluation metrics include SIM, COMET, J (joint), and STA (style transfer accuracy).

---

## 🧠 Acknowledgements

- Detox datasets adapted from [TextDetox](https://huggingface.co/datasets/textdetox/multilingual_paradetox , https://huggingface.co/datasets/s-nlp/ru_paradetox , https://huggingface.co/datasets/s-nlp/paradetox , https://huggingface.co/datasets/textdetox/uk_paradetox)
- Toxicity classifier from [textdetox/xlmr-large-toxicity-classifier](https://huggingface.co/textdetox/xlmr-large-toxicity-classifier)
- LaBSE embedding(https://huggingface.co/sentence-transformers/LaBSE)

---

## 📝 License

MIT License.