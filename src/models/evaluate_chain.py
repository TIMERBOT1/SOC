from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rouge import Rouge


def evaluate_answers(generated, expected):
    # Преобразуем текст для оценки
    rouge = Rouge()
    scores = rouge.get_scores(generated, expected, avg=True)

    # Для различных метрик
    print("ROUGE-1:", scores['rouge-1']['f'])
    print("ROUGE-2:", scores['rouge-2']['f'])
    print("ROUGE-L:", scores['rouge-l']['f'])

    # Для числовых метрик
    # Заметьте, что для числовой оценки может понадобиться предобработка текста
    accuracy = accuracy_score(expected, generated)
    precision = precision_score(expected, generated, average='weighted', zero_division=0)
    recall = recall_score(expected, generated, average='weighted', zero_division=0)
    f1 = f1_score(expected, generated, average='weighted', zero_division=0)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)