import random


def weighted_choice(choices):
    return random.choices(population=list(choices.keys()), weights=choices.values(), k=1)[0]
