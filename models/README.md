
# FinBERT Sentiment Analyzer (Fine-Tuned)

## Model Description
This is a fine-tuned version of `ProsusAI/finbert` designed specifically for classifying the sentiment of financial news headlines into three distinct categories: **Positive, Negative, and Neutral**. 

This model serves as the core inference engine for the FinBERT Sentiment Analyzer FastAPI backend.

## Dataset & Class Imbalance Strategy
The model was trained on a heavily cleaned and preprocessed version of the Financial PhraseBank dataset. During exploratory data analysis, a severe class imbalance was identified, with the **Neutral** class representing roughly 61% of the data. 

To prevent the model from collapsing into a majority-class predictor, we implemented a custom MLOps training strategy:
1. **Dynamic Class Weights:** Penalty weights were calculated using the balanced heuristic ($N / (C \times n_i)$).
2. **Custom Loss Function:** A custom Hugging Face `Trainer` subclass was built to inject these weights directly into a PyTorch `CrossEntropyLoss` function during gradient descent, heavily penalizing misclassifications of the minority (Positive/Negative) classes.

## Evaluation Results
The model was evaluated on a strictly segregated test set (1,000 samples) pulled directly from the Hugging Face Hub to ensure zero data leakage.

* **Macro F1-Score:** `0.9394`
* **Accuracy:** `0.9600`
* **Validation Loss:** `0.1891`

*(Note: Macro F1-Score was prioritized over standard accuracy to validate true performance across the minority classes).*

## Intended Use
This model is intended to be loaded into a FastAPI application for real-time financial sentiment inference. The heavy weight files (`.safetensors`) are hosted on the Hugging Face Hub under the repository name `finbert-finetuned`, while the tokenizer configurations and application logic reside in the associated GitHub repository.

## Developer
**Muyiwa J. Obadara**  
Data Scientist & AI Engineer
