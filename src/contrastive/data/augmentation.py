import random

def delete_random_tokens(string_value):
    """Deletes 10% of the tokens"""
    tokens = string_value.split()
    # Calculate number of to be removed tokens
    num_remove = int(len(tokens)/10) + 1
    for _ in range(num_remove):
        tokens.pop(random.randint(0, len(tokens)-1))

    return ' '.join(tokens)
