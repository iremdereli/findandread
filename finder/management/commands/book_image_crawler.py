import requests
from bs4 import BeautifulSoup

from django.core.management import BaseCommand

from finder.models import Book


class Command(BaseCommand):
    def __init__(self):
        super().__init__()

    def handle(self, *args, **options):

        for book in Book.objects.filter(img_url__istartswith='https://www.goodreads'):
            if url := book.img_url:
                req = requests.get(url)
                content = req.text
                soup = BeautifulSoup(content, 'html.parser')
                image = soup.find("img", {"class": "ResponsiveImage"})
                print(book.book_title, image)
                if not image:
                    continue
                book.img_url = image['src']
                book.save()

        for book in Book.objects.filter(img_url__isnull=True):
            url = f"https://www.goodreads.com/search?q={'+'.join(book.book_title.replace('.', '').replace(':', '').split(' '))}"
            req = requests.get(url)
            content = req.text
            soup = BeautifulSoup(content, 'html.parser')
            link = soup.find('a', {'class': 'bookTitle'})
            if link:
                new_url = f'https://www.goodreads.com/{link["href"]}'
                req = requests.get(new_url)
                content = req.text
                soup = BeautifulSoup(content, 'html.parser')
                image = soup.find("img", {"class": "ResponsiveImage"})
                print(book.book_title, image)
                if not image:
                    continue
                book.img_url = image['src']
                book.save()
