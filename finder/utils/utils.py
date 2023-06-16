import json


def parse_genre_entry(genre_info):
    if genre_info == '':
        return None
    genre_dict = json.loads(genre_info)
    genres = list(genre_dict.values())
    return ", ".join(genres)
