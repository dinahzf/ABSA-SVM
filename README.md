# Aspect-Based Sentiment Analysis on E-commerce Reviews

## Overview
This project applies Aspect-Based Sentiment Analysis (ABSA) to 5,000 food product reviews collected from an online grocery marketplace. Rather than classifying an entire review as simply positive or negative, the analysis identifies which specific product aspects (taste, price, service, delivery, size) customers discuss and the sentiment attached to each one, then trains a Support Vector Machine (SVM) classifier per aspect.

## Objectives
- Identify the aspects most frequently praised or criticized within customer reviews.
- Build and evaluate a per-aspect sentiment classifier that businesses can use to automatically triage incoming reviews.
- Translate aspect-level sentiment distribution into practical recommendations for marketing and service improvement.

## Data
- "Food Reviews: Text Mining and Sentiment Analysis" dataset (Kaggle), 5,000 grocery product reviews.
- Columns: ProductID, Helpfulness_Score, Rating (1-5), Review_Title, and Text.

## Methodology
1. Text preprocessing: case folding, contraction expansion, removal of non-relevant characters (URLs, mentions, numbers, punctuation), duplicate removal, and tokenization.
2. Aspect extraction and mapping: noun extraction via spaCy, grouped into five predefined aspects (taste, price, service, delivery, size).
3. Sentiment scoring per aspect using VADER, classifying reviews as positive (compound score > 0.05) or negative (compound score < -0.05); neutral reviews were excluded as non-substantive.
4. Feature representation using TF-IDF, followed by per-aspect SVM training and evaluation with a stratified 60:40 train-test split, chosen to keep the minority (negative) class adequately represented in both sets.
5. Evaluation using macro F1-score, selected to weight majority and minority classes equally given the class imbalance in the data.

## Key Findings
- All five aspects (delivery, price, service, size, taste) showed a strong majority of positive reviews (over 90% positive in every aspect), with taste and price the most frequently discussed.
- SVM classifiers reliably detect positive sentiment across aspects (F1-scores of 0.96-0.98) but are considerably weaker at detecting negative sentiment (F1-scores ranging from 0.00 for delivery to 0.50 for price and service), reflecting the scarcity of negative examples available for training.
- Macro F1-scores range from 0.48 (delivery) to 0.74 (price and service), highlighting that overall accuracy figures can mask poor minority-class performance.

## Business Relevance
The consistently high share of positive reviews across aspects supports marketing messaging built around taste and price competitiveness, and highlights fast delivery and responsive service as reinforcing themes. At the same time, the model's weak negative-class detection is itself an actionable finding: businesses relying on automated review triage should not assume low negative-flag counts mean an absence of complaints, and should supplement automated triage with periodic manual review, especially for aspects with few negative examples such as delivery.

## Tools & Libraries
Python, spaCy, VADER (SentimentIntensityAnalyzer), scikit-learn (TF-IDF, SVM), contractions, NLTK.

## Repository Contents
- `FoodReviews.csv` — cleaned review dataset with aspect and sentiment labels.
- `ABSA_SVM.ipynb` — full analysis notebook (preprocessing, aspect extraction, VADER scoring, SVM training and evaluation).
- `README.md` — project documentation in English and Indonesian.

## Limitations
The 60:40 train-test split, chosen to preserve minority-class examples, still leaves very few negative samples for some aspects (e.g., only 4 in the delivery test set), limiting how reliably classifier performance can be assessed for that class.
