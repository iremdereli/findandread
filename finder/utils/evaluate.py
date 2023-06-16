class Evaluation:
    def __init__(self, inverted_index):
        self.inverted_index = inverted_index

    def evaluate_query(self, query, relevant_docs):
        # Retrieve the initial document scores using BM25
        initial_scores = self.inverted_index.calculate_scores(query)

        # Get the top-k retrieved documents based on the initial scores
        k = min(10, len(initial_scores))  # Adjust k based on your requirements
        retrieved_docs = sorted(initial_scores, key=initial_scores.get, reverse=True)[:k]

        # Compute precision, recall, and F1-score
        precision = self.calculate_precision(retrieved_docs, relevant_docs)
        recall = self.calculate_recall(retrieved_docs, relevant_docs)
        f1_score = self.calculate_f1_score(precision, recall)

        # Print the evaluation metrics
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1_score:.4f}")

    def calculate_precision(self, retrieved_docs, relevant_docs):
        num_retrieved = len(retrieved_docs)
        num_relevant = len(relevant_docs)

        if num_retrieved == 0:
            return 0.0

        num_correct = len(set(retrieved_docs) & set(relevant_docs))
        precision = num_correct / num_retrieved

        return precision

    def calculate_recall(self, retrieved_docs, relevant_docs):
        num_retrieved = len(retrieved_docs)
        num_relevant = len(relevant_docs)

        if num_relevant == 0:
            return 0.0

        num_correct = len(set(retrieved_docs) & set(relevant_docs))
        recall = num_correct / num_relevant

        return recall

    def calculate_f1_score(self, precision, recall):
        if precision + recall == 0:
            return 0.0

        f1_score = (2 * precision * recall) / (precision + recall)

        return f1_score