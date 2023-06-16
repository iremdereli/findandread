# Generated by Django 4.2.2 on 2023-06-13 17:51

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Book',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('book_id', models.IntegerField()),
                ('freebase_id', models.CharField(max_length=64, null=True)),
                ('book_title', models.CharField(max_length=255, null=True)),
                ('author', models.CharField(max_length=255, null=True)),
                ('publication_date', models.DateTimeField(null=True)),
                ('genre', models.CharField(max_length=255, null=True)),
                ('summary', models.TextField()),
            ],
        ),
    ]