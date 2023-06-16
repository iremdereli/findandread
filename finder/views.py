import pickle
from random import shuffle

from django.shortcuts import render
from django.views import View

from finder.models import Book


class BookView(View):
    url_name = 'Book View'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = 1.0
        self.beta = 0.75
        self.gamma = 0.25

    def get(self, request):
        random_books = list(Book.objects.filter(img_url__isnull=False)[:18])
        shuffle(random_books)

        context = {
            'books': Book.objects.all(),
            'random_books': random_books,
            'query': False
        }

        return render(request, 'finder/book.html', context=context)

    def post(self, request):
        random_books = list(Book.objects.filter(img_url__isnull=False)[:18])
        shuffle(random_books)

        relevant_doc_ids = []
        doc_ids = []
        query = request.POST['query']

        if 'docs' in request.POST:
            for r in request.POST.getlist('docs'):
                relevant_doc_ids.append(int(r))

        with open('finder/inverted_index/idx', 'rb') as f:
            index = pickle.load(f)
            scores = index.calculate_scores(query)
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for doc_id, score in sorted_scores[:10]:
                doc_ids.append(doc_id)

        if relevant_doc_ids:
            expanded_query = index.rocchio_relevance_feedback(query, relevant_doc_ids,
                                                              set(doc_ids).difference(set(relevant_doc_ids)),
                                                              self.alpha, self.beta, self.gamma)
            updated_scores = index.calculate_scores(expanded_query)
            sorted_updated_scores = sorted(updated_scores.items(), key=lambda x: x[1], reverse=True)
            doc_ids = []
            for doc_id, score in sorted_updated_scores[:10]:
                doc_ids.append(doc_id)

        context = {
            'books': Book.objects.filter(id__in=doc_ids),
            'random_books': random_books,
            'query': request.POST['query']
        }

        return render(request, 'finder/book.html', context=context)

