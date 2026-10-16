#!/usr/bin/env python
# This script demonstrates how to find the most similar item in a list based on embeddings.
# For installation and usage instructions, please refer to the README.md file.

import os
import sys
from openai import OpenAI

# Initialize the API key from a hidden file in the user's home directory.
my_api_key = ""
path_to_key = os.path.join(os.path.expanduser("~/"), "keys", ".scads_llm_key")
if os.path.exists(path_to_key):
    with open(path_to_key) as keyfile:
        my_api_key = keyfile.readline().strip()
if not my_api_key:
    print("Error: The key file '.tudai-api-key' did not contain any key. Please ensure the file exists and contains only your API key.")
    sys.exit(1)

# Initialize the OpenAI client with the provided API key and base URL.
client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=my_api_key)

def distance_squaredl2norm(vec1, vec2):
    """
    Computes the squared L2 distance between two vectors.
    """
    return sum((a - b)**2 for a, b in zip(vec1, vec2))

def findMostSimilarItem(some_list, index_of_item_to_find_match_for):
    """
    Finds the index of the item with the best similarity in some_list compared to the item at index_of_item_to_find_match_for. Also returns its similarity score.
    """
    print("Creating embeddings for: " + str(some_list))
    response = client.embeddings.create(
        model = "Qwen/Qwen3-Embedding-4B",
        input = some_list,
        # dimensions=512
        # we do not support matryoshka representation =(
    )

    # Uncomment the following line to see how the response looks like.
    print(response)

    embeddings = [item.embedding for item in response.data]

    current_best_match_score = sys.maxsize
    current_best_match_index = -1

    for index, some_embedding in enumerate(embeddings):
        # don't compare with itself
        if index != index_of_item_to_find_match_for:
            score = distance_squaredl2norm(some_embedding, embeddings[index_of_item_to_find_match_for])
            if score < current_best_match_score:
                current_best_match_score = score
                current_best_match_index = index

    return current_best_match_index, current_best_match_score

# Example list of text snippets.
text_snippets = ["Guten Tag!", "spaghetti code", "Good afternoon!", "clean code", "Schnitzel"]

# Find and print the most similar item for each item in the list.
for i, snippet in enumerate(text_snippets):
    bestMatch, similarity_score = findMostSimilarItem(text_snippets, i)
    print("The most similar item compared to " + snippet + " is: " + text_snippets[bestMatch] + " (similarity score: " + str(similarity_score) + ")")
