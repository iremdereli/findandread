import re

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


class InvertedIndexWord2Vec:
    def __init__(self, corpus, vector_size=100, min_count=1):
        self.corpus = corpus
        self.model = self.train_word2vec_model(vector_size, min_count)
        self.index = self.build_index()

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def tokenize_text(self, text):
        return text.split()

    def train_word2vec_model(self, vector_size, min_count):
        sentences = [self.tokenize_text(self.preprocess_text(doc)) for doc in self.corpus]
        model = Word2Vec(sentences=sentences, vector_size=vector_size, min_count=min_count)
        return model

    def build_index(self):
        index = {}
        for i, doc in enumerate(self.corpus):
            tokens = self.tokenize_text(self.preprocess_text(doc))
            for token in tokens:
                if token in self.model.wv:
                    if token not in index:
                        index[token] = set()
                    index[token].add(i)
        return index

    def calculate_scores(self, query):
        query_tokens = set(self.tokenize_text(self.preprocess_text(query)))
        relevant_docs = set()
        for token in query_tokens:
            if token in self.index:
                relevant_docs.update(self.index[token])

        query_vector = np.mean([self.model.wv[word] for word in query_tokens if word in self.model.wv], axis=0)
        doc_vectors = np.array([np.mean([self.model.wv[word] for word in self.tokenize_text(self.preprocess_text(doc))
                                         if word in self.model.wv], axis=0) for doc in self.corpus])

        scores = cosine_similarity([query_vector], doc_vectors)[0]

        document_scores = {i: score for i, score in enumerate(scores) if i in relevant_docs}
        sorted_scores = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores

    def rocchio_relevance_feedback(self, query, relevant_docs, irrelevant_docs, alpha, beta, gamma):
        # Calculate the initial document scores using Word2Vec
        initial_scores = self.calculate_scores(query)

        # Convert query and document IDs to indices for matrix operations
        doc_ids = list(initial_scores)
        doc_indices = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

        num_docs = len(doc_ids)
        num_terms = len(self.model.wv.key_to_index)

        # Create a matrix to store the term frequencies for each document
        tf_matrix = np.zeros((num_docs, num_terms))

        query_vectors = []
        query_terms = self.tokenize_text(self.preprocess_text(query))
        for term in query_terms:
            if term in self.model.wv.key_to_index:
                query_vectors.append(self.model.wv.get_vector(term))

        if query_vectors:
            query_vector = np.sum(query_vectors, axis=0)
        else:
            query_vector = np.zeros(num_terms)

        for term in query_terms:
            if term in self.model.wv.key_to_index:
                for doc_id in self.index[term]:
                    if doc_id in doc_indices:
                        doc_idx = doc_indices[doc_id]
                        tf_matrix[doc_idx] += np.multiply(query_vector, self.index[term][doc_id])

        # Convert the relevant and irrelevant document IDs to indices
        relevant_indices = [doc_indices[doc_id] for doc_id in relevant_docs if doc_id in doc_indices]
        irrelevant_indices = [doc_indices[doc_id] for doc_id in irrelevant_docs if doc_id in doc_indices]

        # Apply Rocchio's algorithm to update the query vector
        updated_query_vector = alpha * query_vector
        if relevant_indices:
            updated_query_vector += beta * np.mean(tf_matrix[relevant_indices], axis=0)
        if irrelevant_indices:
            updated_query_vector -= gamma * np.mean(tf_matrix[irrelevant_indices], axis=0)

        # Find the most similar terms to the updated query vector
        top_terms = self.model.wv.similar_by_vector(updated_query_vector, topn=10)
        top_terms = [term for term, _ in top_terms]

        # Construct the expanded query
        expanded_query = ' '.join(top_terms)

        return expanded_query
