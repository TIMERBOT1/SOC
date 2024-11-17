import pickle

def save_retriever(retriever, filename):
    try:
        with open(filename, 'wb') as file:
            pickle.dump(retriever, file)
        print(f"Сохранено: {filename}")
    except Exception as e:
        print(f"Ошибка сохранения {filename}: {e}")

def load_retriever(filename):
    try:
        with open(filename, 'rb') as file:
            retriever = pickle.load(file)
        print(f"Загружено: {filename}")
        return retriever
    except Exception as e:
        print(f"Ошибка загрузки {filename}: {e}")
        return None