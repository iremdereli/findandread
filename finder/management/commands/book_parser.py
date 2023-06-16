import csv
import json
import pandas as pd

from django.core.management import BaseCommand
from django.db import connection

from finder.models import Book
from finder.utils.utils import parse_genre_entry


class Command(BaseCommand):
    help = 'Closes the specified poll for voting'

    def handle(self, *args, **options):
        Book.objects.all().delete()

        file = 'dataset/booksummaries.txt'
        data = []
        with open(file, 'r') as f:
            reader = csv.reader(f, dialect='excel-tab')
            for row in reader:
                data.append(row)

        books = pd.DataFrame.from_records(data, columns=['book_id', 'freebase_id', 'book_title', 'author',
                                                         'publication_date', 'genre', 'summary'])
        books['genre'] = books['genre'].apply(parse_genre_entry)

        df_records = books.to_dict('records')
        book_bulk_data = [Book(
            book_id=record['book_id'],
            freebase_id=record['freebase_id'],
            book_title=record['book_title'],
            author=record['author'],
            publication_date=record['publication_date'],
            genre=record['genre'],
            summary=record['summary']
        ) for record in df_records]

        cursor = connection.cursor()
        cursor.execute("TRUNCATE TABLE `finder_book`")
        Book.objects.bulk_create(book_bulk_data)
