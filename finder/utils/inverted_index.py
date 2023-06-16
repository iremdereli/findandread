import math
from collections import defaultdict
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)
        self.document_lengths = defaultdict(int)
        self.num_documents = 0
        self.average_document_length = 0
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def preprocess_text(self, text):
        # Convert text to lowercase
        text = text.lower()

        # Remove non-alphanumeric characters and extra whitespaces
        text = re.sub(r'[^a-z0-9\s*]', '', text)
        text = re.sub(r'\s+', ' ', text)

        # Remove stopwords
        tokens = text.split()
        tokens = [token for token in tokens if token not in self.stopwords]

        # Stemming
        tokens = [self.stemmer.stem(token) for token in tokens]

        # Join tokens back into a string
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text

    def tokenize_text(self, text):
        # Tokenize the preprocessed text into terms
        # You can use a simple whitespace tokenizer or more advanced tokenization techniques
        return text.split()

    def add_document(self, doc_id, text):
        self.num_documents += 1
        preprocessed_text = self.preprocess_text(text)
        tokens = self.tokenize_text(preprocessed_text)
        #print(self.document_lengths, doc_id, tokens, self.document_lengths[doc_id])
        self.document_lengths[doc_id] = len(tokens)
        self.average_document_length += len(tokens)

        # Create the index by mapping each term to the document ID
        term_frequencies = defaultdict(int)
        for token in tokens:
            term_frequencies[token] += 1

        for term, frequency in term_frequencies.items():
            self.index[term].append((doc_id, frequency))

    def calculate_scores(self, query):
        scores = defaultdict(float)
        query_terms = self.tokenize_text(self.preprocess_text(query))

        for term in query_terms:
            if term not in self.index:
                # Perform a wildcard query if the term is a wildcard
                if '*' in term:
                    matching_terms = self.wildcard_query(term)
                    for matching_term in matching_terms:
                        idf = math.log(self.num_documents / len(self.index[matching_term]))
                        for doc_id, term_frequency in self.index[matching_term]:
                            tf_component = term_frequency * (1.0 / (1.0 + 0.5 + 1.5 * (self.document_lengths[doc_id] / self.average_document_length)))
                            scores[doc_id] += idf * tf_component
                continue

            idf = math.log(self.num_documents / len(self.index[term]))

            for doc_id, term_frequency in self.index[term]:
                tf_component = term_frequency * (1.0 / (1.0 + 0.5 + 1.5 * (self.document_lengths[doc_id] / self.average_document_length)))
                scores[doc_id] += idf * tf_component

        return scores

    def rocchio_relevance_feedback(self, query, relevant_docs, irrelevant_docs, alpha, beta, gamma):
        # Calculate the initial document scores using BM25
        initial_scores = self.calculate_scores(query)

        # Convert query and document IDs to indices for matrix operations
        doc_ids = list(initial_scores.keys())
        doc_indices = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

        num_docs = len(doc_ids)
        num_terms = len(self.index)

        # Create a matrix to store the term frequencies for each document
        tf_matrix = np.zeros((num_docs, num_terms))
        term_indices = {term: idx for idx, term in enumerate(self.index.keys())}

        for term in self.index.keys():
            term_idx = term_indices[term]
            for doc_id, term_frequency in self.index[term]:
                if doc_id in doc_indices:
                    doc_idx = doc_indices[doc_id]
                    tf_matrix[doc_idx, term_idx] = term_frequency

        # Convert the relevant and irrelevant document IDs to indices
        relevant_indices = [doc_indices[doc_id] for doc_id in relevant_docs if doc_id in doc_indices]
        irrelevant_indices = [doc_indices[doc_id] for doc_id in irrelevant_docs if doc_id in doc_indices]

        # Convert the query to a vector
        query_vec = np.zeros(num_terms)
        query_terms = self.tokenize_text(self.preprocess_text(query))
        for term in query_terms:
            if term in self.index:
                term_idx = term_indices[term]
                query_vec[term_idx] += 1

        # Apply Rocchio's algorithm to update the query vector
        updated_query_vec = alpha * query_vec
        if relevant_indices:
            updated_query_vec += beta * np.mean(tf_matrix[relevant_indices], axis=0)
        if irrelevant_indices:
            updated_query_vec -= gamma * np.mean(tf_matrix[irrelevant_indices], axis=0)

        # Sort the updated query vector in descending order and retrieve top terms
        top_terms_indices = np.argsort(updated_query_vec)[::-1][:10]
        top_terms = [term for term, idx in term_indices.items() if idx in top_terms_indices]

        # Construct the expanded query
        expanded_query = ' '.join(top_terms)

        return expanded_query

    def wildcard_query(self, term):
        # Convert the wildcard term to a regular expression
        pattern = term.replace('*', '.*')

        matching_terms = []
        for index_term in self.index.keys():
            if re.match(pattern, index_term):
                matching_terms.append(index_term)

        return matching_terms