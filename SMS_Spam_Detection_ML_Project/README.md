# SMS Spam Detection — CodSoft ML Task

Machine-learning text classifier trained on the provided `spam.csv` dataset.

## Dataset
- Original ZIP provided by the user; extracted as `data/spam.csv`.
- Original rows: 5,572.
- Cleaning: retained label/message columns, removed missing rows, invalid labels, empty messages and duplicate records.
- Cleaned rows: 5,158 (4,516 ham; 642 spam).

## Models evaluated
- TF-IDF (English stop words, unigrams + bigrams) + Multinomial Naive Bayes
- TF-IDF (English stop words, unigrams + bigrams) + Logistic Regression

## Actual test results
80/20 stratified split, random seed 42 (1,032 test messages).

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1 |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 95.54% | 100.00% | 64.06% | 78.10% |
| Logistic Regression | 95.25% | 98.77% | 62.50% | 76.56% |

Best model selected by spam F1: **Multinomial Naive Bayes**. Its confusion matrix (actual rows ham, spam; predicted columns ham, spam): `[[904, 0], [46, 82]]`.

Note: Accuracy is high partly because ham is the majority class. Spam recall is lower: the test set had 128 spam messages and the model detected 82. Review the confusion matrix and spam recall, not accuracy alone.

## Run
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
python train_model.py
python predict_sms.py
```

The training script saves trained models in `models/` and metrics in `results/`. The interactive prediction script loads the saved best model.

## Example inputs
- `Congratulations! You have won a free prize. Call now to claim.`
- `Hey, are we meeting at 6 today?`

The probability is a model score and should not be interpreted as a guarantee.
