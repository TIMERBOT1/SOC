import numpy as np
import re
import pandas as pd
import sys
import evaluate
import spacy
from sentence_transformers import SentenceTransformer, util
from clearml import Task
from rag_chain import built_rag_chain
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

  ground_truth = [] 

  rag_chain = built_rag_chain()

  df_test = pd.read_csv(sys.argv[1])
  questions = df_test['question'].values.tolist()

  for q in questions:
     ground_truth.append(rag_chain.invoke(q))
  df_test['ground_truth'] = ground_truth

  # далаем предобработку текстов
  df_test['cleaned_ground_truth'] = df_test['ground_truth'].apply(
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
  ground_truths = df_test['cleaned_ground_truth']
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

  result_metrics = {key: float(np.float32(value)) for key, value in result_metrics.items()}

  with open('data/processed/metrics.json', 'w') as json_file:
    json.dump(result_metrics, json_file)

  task = Task.init(project_name="RAG Evaluation", task_name="Evaluation Metrics Logging")
  logger = task.get_logger()  

  # Логирование метрик
  for metric_name, metric_value in result_metrics.items():
    logger.report_scalar("Evaluation Metrics", metric_name, metric_value, iteration=0) 

  # Логирование текстов
  logger.report_text("Ground Truths:\n" + "\n".join(ground_truths))
  logger.report_text("Model Predictions:\n" + "\n".join(df_test["answer"].tolist()))  

  task.close()


if __name__ == "__main__":
  evaluate_generation() 