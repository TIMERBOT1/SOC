class Retriever:
    
    def __init__(self, persist_directory, embeddings, dataset):
        self.persist_directory = persist_directory
        self.embeddings = embeddings
        self.dataset = dataset

    def create_retriever(self):
        pass