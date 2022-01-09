import pathlib
import random

import pandas as pd
import time
import Graph as graph
import copy
import os
from Diagram import Diagram as class_diagram
from nltk.corpus import stopwords
import nltk

nltk.download('wordnet_ic')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('en_core_web_lg')
from similarity_maxtrix import matrix

import spacy

nlp = spacy.load("en_core_web_sm")
matrix_obj = matrix()
Similarity_matrix = matrix_obj.edge_matrix()
Similarity_matrix_class = matrix_obj.class_matrix()

def greedy_algorithm_recursive(target_graph, query_graph, th, k, hop):
    open_list = []
    output_list = []
    close_list_query = [1]
    is_visited = []
    similarity = similarity_estimation('0', target_graph, query_graph)

    is_visited.append(similarity[0][1])
    open_list.append([(similarity[0][1], 1, similarity[0][0],
                       [similarity[0][1]])])  # (target_vertex, query_vertex, similarity, is_visted)

    while len(open_list) != 0:

        state = open_list.pop()

        query_id = state[len(state) - 1][1]
        neighbors = query_graph.neighbors(query_id)
        # print(len(open_list))
        # print(state)
        # print(query_graph)
        if not neighbors or query_graph.is_last(query_id):
            output_list.append(state)
            continue

        for vertex_query in neighbors:

            close_list_query.append(vertex_query)
            is_above, similar_nodes, is_visited = next_similar_node(target_graph, query_graph, vertex_query,
                                                                    state[len(state) - 1][1], state[len(state) - 1][0],
                                                                    th,
                                                                    state[len(state) - 1][3], k, hop, is_visited)
            # next_similar_node(target_graph, query_graph, query_id, query_id_prev, target_id, th, is_visited_vertex)

            for vertex in similar_nodes:
                cloned_state = copy.deepcopy(state)
                if is_above:
                    cloned_state.append(vertex)
                    open_list.append(cloned_state)
                elif len(vertex[3]) < 4:
                    # else:
                    cloned_state.append((vertex[0], state[len(state) - 1][1], vertex[2], vertex[3]))
                    open_list.append(cloned_state)

                # else:
                #     print()

    return output_list


def get_type_sim(type1, type2):
    return Similarity_matrix_class[type1][type2]


def get_sim_between_2_edges(edge1, edge2):
    if edge2 in Similarity_matrix[edge1].keys():
        return Similarity_matrix[edge1][edge2]
    else:
        return Similarity_matrix[edge2][edge1]

def get_sim_between_2_nodes(target_graph, key, query_graph, query_id2):
    doc1 = target_graph.vertex_vectors[key]
    doc2 = query_graph.vertex_vectors[int(query_id2)]
    try:
        sim = doc1.similarity(doc2)
    except:
        print()
    return sim


def similarity_estimation(vertex_2, target_graph, query_graph):
    # print("Start finding the most similar vertex in the code graph to the first vertex of a query:")
    max_similarity = 0
    # node_id = vertex_2
    # query_text = query_graph.vertex_info[int(node_id)].split()
    # print(str(vertex_2))
    for key in target_graph.vertex_info:
        # node_text = target_graph.vertex_info[key].split()
        type_1 = query_graph.get_type(0)
        type_2 = target_graph.get_type(key)
        # print("[" + str(key) + "]: " + str(vertex_2))

        sim_semantic = get_sim_between_2_nodes(target_graph, key, query_graph, vertex_2)

        sim_type = get_type_sim(type_1, type_2)

        sim = (sim_semantic + sim_type) / 2
        # print("similarity: " + str(sim))
        if sim > max_similarity:
            max_similarity = sim
            node_id = key

    return [(max_similarity, node_id)]


def read_results(results):
    tot_output = []
    for item in results:
        output = []
        similarity = 0
        for step in item:
            output.append(step[0])
            similarity += step[2]

        # print(output, similarity / len(item))
        tot_output.append((output, similarity / len(item)))
    return tot_output

post = class_diagram("check/" + "m1.json", nlp)
query = class_diagram("check/" + "q1.json", nlp)
results = greedy_algorithm_recursive(post, query, 0.45, 2, 1)
outputs = read_results(results)