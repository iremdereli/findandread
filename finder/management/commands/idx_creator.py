import pickle

from django.core.management import BaseCommand

from finder.models import Book
from finder.utils.inverted_index import InvertedIndex
from finder.utils.inverted_index_word2vec import InvertedIndexWord2Vec


class Command(BaseCommand):

    def handle(self, *args, **options):
        idx = InvertedIndex()
        for book in Book.objects.all():
            str = ""
            for i in range(100):
                str += f'{book.book_title} '
                str += f'{book.author} '
            str += book.summary
            idx.add_document(book.id, str)

        with open('finder/inverted_index/idx', 'wb') as f:
            pickle.dump(idx, f)

        corpuss = []
        for book in Book.objects.all():
            corpuss.append(book.summary)

        indexW2V = InvertedIndexWord2Vec(corpuss)
        with open('finder/inverted_index/idx_word2vec',
                  'wb') as f:
            pickle.dump(indexW2V, f)
