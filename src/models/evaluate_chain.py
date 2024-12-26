import numpy as np
import re
import pandas as pd
import sys
import evaluate
import spacy
from sentence_transformers import SentenceTransformer, util
import pickle
import json


def tokenize_words(sent):
  '''Токенизация предложений.'''
  sent = re.sub(r'[^\w\s]','',sent)
  return sent.lower().split(' ')


def evaluate_generation():
  '''Оценка генерации с помощью метрик машинного перевода.
      Параметры: 
          df_test (pd.DataFrame): датафрейм, содержащий question, ground_truth, contexts, answer 
      Возвращаемое знаение:
          result_metrics (dict): Словарь с метриками.'''

  # загружаем пайплайн обработки текста из библиотеки spacy для русского языка
  nlp = spacy.load("ru_core_news_sm")

  ground_truths = []

  with open('data/processed/rag_chain.pkl', 'wb') as file:
        rag_chain = pickle.load(file)

  df_test = pd.read_csv(sys.argv[1])
  questions = df_test['question'].values.tolist()

  for q in questions:
     ground_truths.append(rag_chain.invoke(q))

  # далаем предобработку текстов
  ground_truths = list(map(lambda x: ' '.join(
        token.lemma_.lower() for token in nlp(x) if
        not token.is_stop
        and not token.is_punct
        and not token.is_digit
        and not token.like_email
        and not token.like_num
        and not token.is_space
    ), ground_truths)
  )

  df_test['cleaned_answer'] = df_test['answer'].apply(
    lambda x: ' '.join(
        token.lemma_.lower() for token in nlp(x) if
        not token.is_stop
        and not token.is_punct
        and not token.is_digit
        and not token.like_email
        and not token.like_num
        and not token.is_space
    )
  )

  # получаем предобработанные ответы из документации и из пайплайна RAG
  ground_truths = pd.Series(ground_truths)
  answers = df_test["answer"]

  # расчет метрики rouge
  rouge = evaluate.load("rouge")
  rouge_scores = rouge.compute(
      predictions=answers, references=ground_truths, tokenizer = tokenize_words
  )
  
  # расчет метрики bleu
  bleu = evaluate.load("bleu")
  bleu_scores = bleu.compute(predictions=answers, references=ground_truths, tokenizer=tokenize_words)
  
  # расчет метрики bleu
  model = SentenceTransformer('DeepPavlov/rubert-base-cased')
  embeddings_true= [model.encode(i, convert_to_tensor=True) for i in ground_truths]
  embedding_pred = [model.encode(i, convert_to_tensor=True) for i in answers]
  similarity = [util.pytorch_cos_sim(i[0], i[1]) for i in zip(embeddings_true, embedding_pred)]
  similarity_score = np.mean([i[0][0] for i in similarity])

  # расчет метрики meteor
  meteor = evaluate.load("meteor")
  meteor_scores = meteor.compute(predictions=answers, references=ground_truths)

  # расчет метрики ter
  ter = evaluate.load("ter")
  ter_scores = ter.compute(predictions=ground_truths, references=answers)

  # расчет метрики chrf
  chrf = evaluate.load("chrf")
  chrf_scores = chrf.compute(predictions=ground_truths, references=answers)

  # собираем итоговый словарь с метриками
  result_metrics = {
    'rouge1': rouge_scores['rouge1'], 
    'rouge2': rouge_scores['rouge2'],
    'rougeL': rouge_scores['rougeL'],
    'rougeLsum': rouge_scores['rougeLsum'],
    'bleu': bleu_scores['precisions'][0],
    'similarity_score': similarity_score,
    'meteor': meteor_scores['meteor'],
    'ter': ter_scores['score'],
    'chrf': chrf_scores['score']
  }

  with open('data/processed/metrics.json', 'w') as json_file:
    json.dump(result_metrics, json_file)

  return result_metrics


if __name__ == "__main__":
  evaluate_generation() 