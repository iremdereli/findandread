from django.db import models


class Book(models.Model):
    book_id = models.IntegerField()
    freebase_id = models.CharField(max_length=64, null=True)
    book_title = models.CharField(max_length=255, null=True)
    author = models.CharField(max_length=255, null=True)
    publication_date = models.CharField(max_length=64, null=True)
    genre = models.CharField(max_length=255, null=True)
    summary = models.TextField()
    img_url = models.CharField(max_length=1028, null=True)
