import nltk
import json
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
import Metrics
import pandas as pd

nlp = spacy.load("en_core_web_sm")
comb_vertex = [
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
    [0.7, 0.8, 0.6, 0.8, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1],
    [0.5, 0.7, 0.9, 0.5, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3],
    [0.8, 0.7, 0.6, 0.8, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6],
    [0.9, 0.9, 0.9, 0.9, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4],
    [0.6, 0.8, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4, 0.4, 0.5, 0.6, 0.4, 0.6]
]
comb_edge = [
    [0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
    [0.7, 0.8, 0.6, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1],
    [0.5, 0.7, 0.9, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3],
    [0.8, 0.7, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6],
    [0.8, 0.7, 0.6, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4],
    [0.9, 0.9, 0.9, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4],
    [0.6, 0.8, 0.6, 0.1, 0.1, 0.1, 0.5, 0.5, 0.6, 0.8, 0.6, 0.1, 0.1, 0.1, 0.5, 0.5]

]
matrix_obj = matrix(comb_vertex[5], comb_edge[6])
Similarity_matrix = matrix_obj.edge_matrix()
Similarity_matrix_class = matrix_obj.class_matrix()


def search_algorithm(target_graph, query_graph, th, k, hop):
    open_list = []
    output_list = []
    close_list_query = [0]
    is_visited = []
    similarity = similarity_estimation('0', target_graph, query_graph)

    is_visited.append(similarity[0][1])
    open_list.append([(similarity[0][1], 0, similarity[0][0],
                       [similarity[0][1]])])  # (target_vertex, query_vertex, similarity, is_visted)

    while len(open_list) != 0:
        state = open_list.pop()
        query_id = state[len(state) - 1][1]
        neighbors = query_graph.neighbors(query_id)
        if not neighbors:
            output_list.append(state)
            continue
        # if not neighbors or query_graph.is_last(query_id):
        #     output_list.append(state)
        #     continue

        for vertex_query in neighbors:

            close_list_query.append(vertex_query)
            is_above, similar_nodes, is_visited = next_similar_node(target_graph, query_graph, vertex_query,
                                                                    state[len(state) - 1][1], state[len(state) - 1][0],
                                                                    th, state[len(state) - 1][3], k, hop, is_visited)
            # next_similar_node(target_graph, query_graph, query_id, query_id_prev, target_id, th, is_visited_vertex)

            for vertex in similar_nodes:
                cloned_state = copy.deepcopy(state)
                if is_above:
                    cloned_state.append(vertex)
                    open_list.append(cloned_state)
                elif len(vertex[3]) < 4:
                    cloned_state.append((vertex[0], state[len(state) - 1][1], vertex[2], vertex[3]))
                    open_list.append(cloned_state)

    return output_list


def evaluate(actual, predict):
    avrege_MRR_domain = 0
    average_MRR_exact = 0
    average_semantic = 0
    average_semantic_count = 0.0
    recall_5 = 0

    for key in predict:
        temp = predict[key]
        temp.sort(key=lambda tup: tup[0][1])
        temp.reverse()
        actual_result = actual.loc[actual['query'] == int(key)]
        # print(f'actual: {actual_result}')
        counter = 0
        is_entered_name = False
        is_entered_path = False
        for index, result in enumerate(temp):
            if index < 10:
                print(result)
            predicted_path = ''.join(str(e) for e in result[0][0])
            predicted_name = result[1]
            try:
                expected_path = actual_result['result1'].values[0].replace(',', '')
            except:
                print('result 1 not working: index 0 is out of bounds for axis 0 with size 0')
                print(actual_result['result1'])
            try:
                expected_path_2 = actual_result['result2'].values[0].replace(',', '')
            except:
                print('result 2 not working: index 0 is out of bounds for axis 0 with size 0')
                print(actual_result['result2'])
            try:
                expected_name = actual_result['domain'].values[0]
            except:
                print('domain not works')
                print(actual_result['domain'])
            if predicted_name == expected_name:
                if not is_entered_name:
                    if counter < 5:
                        recall_5 = recall_5 + 1

                    avrege_MRR_domain = avrege_MRR_domain + (1 / (counter + 1))
                    is_entered_name = True
                    average_semantic += result[0][1]
                    average_semantic_count = average_semantic_count + 1

                if predicted_path == expected_path or predicted_path == expected_path_2:
                    if not is_entered_path:
                        is_entered_path = True
                        average_MRR_exact = average_MRR_exact + (1 / (counter + 1))
            counter += 1
    if average_semantic_count == 0:
        return average_MRR_exact / len(predict), avrege_MRR_domain / len(
            predict), 0, recall_5 / len(predict)
    else:
        return average_MRR_exact / len(predict), avrege_MRR_domain / len(
            predict), average_semantic / average_semantic_count, recall_5 / len(predict)


def next_similar_node(target_graph, query_graph, query_id, query_id_prev, target_id, th, is_visited_vertex, k, hop,
                      is_visited):
    max_similarity = []
    query_arc = query_graph.arc_type(query_id_prev, query_id)
    query_type = query_graph.get_type(query_id)
    for key in target_graph.neighbors(target_id):
        if key in is_visited:
            continue
        target_arc = target_graph.arc_type(target_id, key)
        target_type = target_graph.get_type(key)

        label_similarity = get_sim_between_2_nodes(target_graph, key, query_graph, query_id, target_id, query_id_prev)
        edge_type_similarity = get_sim_between_2_edges(target_arc, query_arc)
        vertex_type_similarity = get_type_sim(target_type, query_type)
        similarity = (label_similarity + edge_type_similarity + vertex_type_similarity) / 3

        cloned_is_visited_vertex = copy.deepcopy(is_visited_vertex)
        cloned_is_visited_vertex.append(key)
        is_visited.append(key)
        max_similarity.append((key, query_id, similarity, cloned_is_visited_vertex))
    max_similarity.sort(key=lambda tup: tup[2])
    max_similarity.reverse()
    if not max_similarity:
        return False, [], is_visited
    if max_similarity[0][2] > th:
        output = []
        counter = 0
        for item in max_similarity:
            if counter < k:
                output.append(item)
                counter += 1
        return True, output, is_visited
    else:
        new_max_similarity = []
        for item in max_similarity:
            new_max_similarity.append((item[0], item[1], hop, item[3]))
        return False, new_max_similarity, is_visited


def Average(lst):
    return sum(lst) / len(lst)


def get_type_sim(type1, type2):
    # print("1"+type1)
    # print("2"+type2)
    return Similarity_matrix_class[type1][type2]


def get_sim_between_2_edges(edge1, edge2):
    if edge2 in Similarity_matrix[edge1].keys():
        return Similarity_matrix[edge1][edge2]
    else:
        return Similarity_matrix[edge2][edge1]


def get_sim_between_2_nodes(target_graph, key, query_graph, query_id2, code_from, query_from):
    doc1 = target_graph.edge_info[(code_from, key)][1]
    doc2 = query_graph.edge_info[(query_from, int(query_id2))][1]
    try:
        sim = doc1.similarity(doc2)
    except:
        print()
    return sim


#
def get_sim_between_2_nodes_a(target_graph, key, query_graph, query_id2):
    doc1 = target_graph.vertex_vectors[key]
    doc2 = query_graph.vertex_vectors[int(query_id2)]
    try:
        sim = doc1.similarity(doc2)
    except:
        print()
    return sim


def similarity_estimation(vertex_2, target_graph, query_graph):
    # Start finding the most similar vertex in the code graph to the first vertex of a query
    max_similarity = 0
    for key in target_graph.vertex_info:
        type_1 = query_graph.get_type(0)
        type_2 = target_graph.get_type(key)
        sim_semantic = get_sim_between_2_nodes_a(target_graph, key, query_graph, vertex_2)
        sim_type = get_type_sim(type_1, type_2)
        sim = (sim_semantic + sim_type) / 2
        if sim > max_similarity:
            max_similarity = sim
            node_id = key

    return [(max_similarity, node_id)]


# ----------------------------dani----------------------------
# def similarity_estimation(vertex_2, target_graph, query_graph):
#     # Start finding the most similar vertex in the code graph to the first vertex of a query
#     max_similarity = 0
#     counter = 0
#     from_query_list = [x[0] for x in query_graph.edge_info if x[1] == int(vertex_2)]
#     print(query_graph.edge_info )
#     for from_, to_ in target_graph.edge_info:
#         type_1 = query_graph.get_type(0)
#         type_2 = target_graph.get_type(key)
#         try:
#             sim_semantic = get_sim_between_2_nodes(target_graph, key, query_graph, vertex_2, from_, from_query_list[counter])
#         except:
#             pass
#         sim_type = get_type_sim(type_1, type_2)
#         sim = (sim_semantic + sim_type) / 2
#         if sim > max_similarity:
#             max_similarity = sim
#             node_id = key
#         counter += 1
#     return [(max_similarity, node_id)]


def read_results(results):
    tot_output = []
    for item in results:
        output = []
        similarity = 0
        for step in item:
            output.append(step[0])
            similarity += step[2]
        tot_output.append((output, similarity / len(item)))
    return tot_output


#
# post = class_diagram("check/" + "m1.json", nlp)
# query = class_diagram("check/" + "q1.json", nlp)
# results = greedy_algorithm_recursive(post, query, 0.45, 2, 1)
# outputs = read_results(results)
path = pathlib.Path(__file__).parent.absolute()
maps = {}
# evaluation = pd.read_csv(str(path) + '\\evaluation.csv')
# benchmark = pd.read_csv(str(path) + '\\check\\Benchmark.csv')
ths = [0.45, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
ks = [1, 2, 3, 4, 5]
hops = [1, 0.9, 0.8, 0.7, 0.6]
df_results = {'hops': [], 'th': [], 'k': [], 'ExactMRR': [], 'DomainMrr': [], 'AvgSemantic': [], 'Recall5': []}

# load the codes graphs into maps
for filename in os.listdir(str(path) + "\\check/codes"):
    map_name = filename.split('.')[0]
    post = class_diagram("check/codes/" + filename, nlp)
    maps[map_name] = post

th = 0.51
k = 5
hop = 1
# for th in ths:
#     for k in ks:
#         for hop in hops:
upperbound = []
data_result_search = {}
# --------------------------------------------
output = []
queries = []
for filename in os.listdir(str(path) + "\\check/queries"):
    query_key = filename[:len(filename) - 5]
    queries.append(query_key)
    query = class_diagram("check/queries/" + f'{filename}', nlp)

    data_result_search[query_key] = []
    for key_map in maps.keys():
        map_name = key_map
        post = maps[key_map]
        # print(f'threshhold: {th}, k: {k}, hop:{hop} \n')
        results = search_algorithm(post, query, th, k, hop)
        outputs = read_results(results)
        if outputs:
            for output in outputs:
                data_result_search[query_key].append((output, map_name))
for k in data_result_search:
    data_result_search[k] = sorted(data_result_search[k], key=lambda x: x[0][1], reverse=True)
for k, v in data_result_search.items():
    print("query number:" + k)
    print("Relevant paths:")
    for i in range(len(v)):
        print("Project:" + v[len(v) - 1 - i][1])
        print("Score:" + str(v[len(v) - 1 - i][0][1]))
        print(v[len(v) - 1 - i][0][0])
        print("#--------------------------------------------")

print("Presenting evaluation of the result:")
bm = pd.read_csv('Benchmark.csv')

for q in range(5,20):
    evaluation_df = pd.DataFrame(columns=['Query', 'Precision', 'Recall', 'F1'])
    for index, row in bm.iterrows():
        query_id = row[0]
        query_text = row[1]
        project = row[2]
        path = [p.strip()[1:len(p.strip())-1] for p in row[3].split('~')]
        relevant_path = [p.split(',')for p in path]
        temp_list = []
        for i, pr in enumerate(relevant_path):
            for j, x in enumerate(pr):
                relevant_path[i][j] = int(relevant_path[i][j])
                # temp_list.append(int(pr))

        # relevant_path = [[int(x[0]),int(x[1])] for x in relevant_path ]
        retrieved_result = [x[0][0] for x in data_result_search[str(index + 1)]]

        precision = Metrics.precision_at_k(relevant_path,retrieved_result,q)
        recall = Metrics.recall_at_k(relevant_path,retrieved_result, q)
        f1 = Metrics.f1(precision, recall)
        evaluation_df.loc[len(evaluation_df.index)] = [query_id,precision,recall,f1]
    evaluation_df.to_csv(f'evaluation{q}.csv')
# --------------------------------------------


#
# for i in range(len(comb_vertex)):
#     for j in range(len(comb_edge)):
#         matrix_obj = matrix(comb_vertex[i], comb_edge[j])
#         for queryname in os.listdir("check/queries/"):
#             output = []
#             query = class_diagram("check/queries/" + queryname, nlp)
#             query_key = queryname.split('.')[0]
#             data_result_search[query_key] = []
#             for key_map in maps.keys():
#                 map_name = key_map
#                 post = maps[key_map]
#                 print(f'threshhold: {th}, k: {k}, hop:{hop} \n')
#                 results = search_algorithm(post, query, th, k, hop)
#                 outputs = read_results(results)
#                 if outputs:
#                     for output in outputs:
#                         data_result_search[query_key].append((output, map_name))
#             print(
#                 '------------------------------------------------------------------------------------------------------')
#
#         avg_exact, avg_domain, avg_semantic, recall = evaluate(evaluation, data_result_search)
#
#
#         df_results['hops'].append(hop)
#         df_results['th'].append(th)
#         df_results['k'].append(k)
#         df_results['ExactMRR'].append(avg_exact)
#         df_results['DomainMrr'].append(avg_domain)
#         df_results['AvgSemantic'].append(avg_semantic)
#         df_results['Recall5'].append(recall)
# -------------------------------------------------------------

# import pathlib
# import random
#
# import pandas as pd
# import time
# import Graph as graph
# import copy
# import os
# from Diagram import Diagram as class_diagram
# from nltk.corpus import stopwords
# import nltk
#
# nltk.download('wordnet_ic')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('en_core_web_lg')
# import sematch.sematch.semantic.similarity as smch
# from similarity_maxtrix import matrix
# # from similarity_maxtrix_cd import matrix
# from similarity_maxtrix import matrix
#
# import spacy
#
# nlp = spacy.load("en_core_web_sm")
#
# # wns = smch.WordNetSimilarity()
# matrix_obj = matrix()
# Similarity_matrix = matrix_obj.edge_matrix()
# Similarity_matrix_class = matrix_obj.class_matrix()
# path = pathlib.Path(__file__).parent.absolute()
# evaluation = pd.read_csv(str(path) + '\\evaluation.csv')
# print(evaluation)
#
#
# def greedy_algorithm_recursive(target_graph, query_graph, th, k, hop):
#     open_list = []
#     output_list = []
#     close_list_query = [-1]
#     is_visited = []
#     similarity = similarity_estimation('-1', target_graph, query_graph)
#
#     is_visited.append(similarity[0][1])
#     open_list.append([(similarity[0][1], -1, similarity[0][0],
#                        [similarity[0][1]])])  # (target_vertex, query_vertex, similarity, is_visted)
#
#     while len(open_list) != 0:
#
#         state = open_list.pop()
#
#         query_id = state[len(state) - 1][1]
#         neighbors = query_graph.neighbors(query_id)
#         # print(len(open_list))
#         # print(state)
#         # print(query_graph)
#         if not neighbors or query_graph.is_last(query_id):
#             output_list.append(state)
#             continue
#
#         for vertex_query in neighbors:
#
#             close_list_query.append(vertex_query)
#             is_above, similar_nodes, is_visited = next_similar_node(target_graph, query_graph, vertex_query,
#                                                                     state[len(state) - 1][1], state[len(state) - 1][0],
#                                                                     th,
#                                                                     state[len(state) - 1][3], k, hop, is_visited)
#             # next_similar_node(target_graph, query_graph, query_id, query_id_prev, target_id, th, is_visited_vertex)
#
#             for vertex in similar_nodes:
#                 cloned_state = copy.deepcopy(state)
#                 if is_above:
#                     cloned_state.append(vertex)
#                     open_list.append(cloned_state)
#                 elif len(vertex[3]) < 4:
#                     # else:
#                     cloned_state.append((vertex[0], state[len(state) - 1][1], vertex[2], vertex[3]))
#                     open_list.append(cloned_state)
#
#                 # else:
#                 #     print()
#
#     return output_list
#
#
# def next_similar_node(target_graph, query_graph, query_id, query_id_prev, target_id, th, is_visited_vertex, k, hop,
#                       is_visited):
#     max_similarity = []
#     temp_similarity_calculation = []
#     # print("next_similar_node", str(query_id))
#     # query_text = query_graph.vertex_info[query_id].split()
#     query_arc = query_graph.arc_type(query_id_prev, query_id)
#     query_type = query_graph.get_type(query_id)
#     for key in target_graph.neighbors(target_id):
#         if key in is_visited:
#             continue
#         # target_text = target_graph.vertex_info[key].split()
#         target_arc = target_graph.arc_type(target_id, key)
#         target_type = target_graph.get_type(key)
#
#         label_similarity = get_sim_between_2_nodes(target_graph, key, query_graph, query_id)
#         edge_type_similarity = get_sim_between_2_edges(target_arc, query_arc)
#         vertex_type_similarity = get_type_sim(target_type, query_type)
#
#         similarity = (label_similarity + edge_type_similarity + vertex_type_similarity) / 3
#
#         cloned_is_visited_vertex = copy.deepcopy(is_visited_vertex)
#         cloned_is_visited_vertex.append(key)
#         is_visited.append(key)
#         max_similarity.append((key, query_id, similarity, cloned_is_visited_vertex))
#         # print("[" + str(key) + "]: " + str(query_id), str(similarity))
#     max_similarity.sort(key=lambda tup: tup[2])
#     max_similarity.reverse()
#     # print(query_id)
#     if not max_similarity:
#         # print([])
#         return False, [], is_visited
#     if max_similarity[0][2] > th:
#         output = []
#         counter = 0
#         for item in max_similarity:
#             if counter < k:
#                 output.append(item)
#                 counter += 1
#         # print(output)
#         return True, output, is_visited
#     else:
#         new_max_similarity = []
#         for item in max_similarity:
#             new_max_similarity.append((item[0], item[1], hop, item[3]))
#         # print(new_max_similarity)
#         return False, new_max_similarity, is_visited
#
#
# def get_sim_between_2_nodes(target_graph, key, query_graph, query_id2):
#     doc1 = target_graph.vertex_vectors[key]
#     doc2 = query_graph.vertex_vectors[int(query_id2)]
#     try:
#         sim = doc1.similarity(doc2)
#     except:
#         print()
#     return sim
#
#
# # def get_sim_between_2_nodes(target_graph, key, query_graph, query_id2):
# #     # This function receives two nodes an returns the calculated semantic similarity between them.
# #     node1 = target_graph.vertex_info[key].split()
# #     node2 = query_graph.vertex_info[int(query_id2)].split()
# #     node1.reverse()
# #     if node1 == node2:
# #         return 1
# #     stopwords_set = stopwords.words('english')
# #     node1 = list(set(node1) - set(stopwords_set))
# #     node2 = list(set(node2) - set(stopwords_set))
# #     sim = 0
# #     for word1 in node1:
# #         max = 0
# #         for word2 in node2:
# #             if word2 == word1:
# #                 add = 1
# #             else:
# #                 # if word1 not in self.wns.vocab or word2 not in self.wns.vocab:
# #                 #     add = 0.1
# #                 # else:
# #                 add = wns.word_similarity(word1, word2)  # model similarity # wns word_similarity
# #             if add > max:
# #                 max = add
# #             # add = add / len(node1) if len(node1) >= len(node2) else add / len(node2)
# #         sim = sim + max
# #     sim = sim / len(node1) if len(node1) >= len(node2) else sim / len(node2)
# #
# #     return sim
#
#
# def get_type_sim(type1, type2):
#     return Similarity_matrix_class[type1][type2]
#
#
# def get_sim_between_2_edges(edge1, edge2):
#     if edge2 in Similarity_matrix[edge1].keys():
#         return Similarity_matrix[edge1][edge2]
#     else:
#         return Similarity_matrix[edge2][edge1]
#
#
# def similarity_estimation(vertex_2, target_graph, query_graph):
#     # print("Start finding the most similar vertex in the code graph to the first vertex of a query:")
#     max_similarity = 0
#     # node_id = vertex_2
#     # query_text = query_graph.vertex_info[int(node_id)].split()
#     # print(str(vertex_2))
#     for key in target_graph.vertex_info:
#         # node_text = target_graph.vertex_info[key].split()
#         type_1 = query_graph.get_type(-1)
#         type_2 = target_graph.get_type(key)
#         # print("[" + str(key) + "]: " + str(vertex_2))
#
#         sim_semantic = get_sim_between_2_nodes(target_graph, key, query_graph, vertex_2)
#
#         sim_type = get_type_sim(type_1, type_2)
#
#         sim = (sim_semantic + sim_type) / 2
#         # print("similarity: " + str(sim))
#         if sim > max_similarity:
#             max_similarity = sim
#             node_id = key
#
#     return [(max_similarity, node_id)]
#
#
# def read_results(results):
#     tot_output = []
#     for item in results:
#         output = []
#         similarity = 0
#         for step in item:
#             output.append(step[0])
#             similarity += step[2]
#
#         # print(output, similarity / len(item))
#         tot_output.append((output, similarity / len(item)))
#     return tot_output
#
#
# def Average(lst):
#     return sum(lst) / len(lst)
#
#
# def evaluate(actual, predict):
#     avrege_MRR_domain = 0
#     average_MRR_exact = 0
#     average_semantic = 0
#     average_semantic_count = 0.0
#     recall_5 = 0
#
#     for key in predict:
#         # print('++++++++++++++++++++++++++++++++')
#         temp = predict[key]
#         temp.sort(key=lambda tup: tup[0][1])
#         temp.reverse()
#         actual_result = actual.loc[actual['query'] == int(key)]  # int()
#         # actual_result = actual[actual['query'] == int(key)]  # int()
#         print(f'actual: {actual_result}')
#         counter = 0
#         is_entered_name = False
#         is_entered_path = False
#         for index, result in enumerate(temp):
#             if index < 10:
#                 print(result)
#             predicted_path = ''.join(str(e) for e in result[0][0])
#             predicted_name = result[1]
#             try:
#                 # expected_path = actual_result['result1'].values[0].replace(',', '')
#                 expected_path = actual_result['result1'].values[0].replace(',', '')
#             except:
#                 print('result 1 not working: index 0 is out of bounds for axis 0 with size 0')
#                 print(actual_result['result1'])
#             try:
#                 expected_path_2 = actual_result['result2'].values[0].replace(',', '')
#             except:
#                 print('result 2 not working: index 0 is out of bounds for axis 0 with size 0')
#                 print(actual_result['result2'])
#             try:
#                 expected_name = actual_result['domain'].values[0]
#             except:
#                 print('domain not works')
#                 print(actual_result['domain'])
#             if predicted_name == expected_name:
#                 # print(key + ': expected name: ', counter)
#                 if not is_entered_name:
#                     if counter < 5:
#                         recall_5 = recall_5 + 1
#
#                     avrege_MRR_domain = avrege_MRR_domain + (1 / (counter + 1))
#                     is_entered_name = True
#                     average_semantic += result[0][1]  #####
#                     average_semantic_count = average_semantic_count + 1
#
#                 if predicted_path == expected_path or predicted_path == expected_path_2:
#                     # print(key + ':expected path: ', counter)
#                     if not is_entered_path:
#                         is_entered_path = True
#                         average_MRR_exact = average_MRR_exact + (1 / (counter + 1))
#             counter += 1
#     # print(len(predict))
#     return average_MRR_exact / len(predict), avrege_MRR_domain / len(
#         predict), average_semantic / average_semantic_count, recall_5 / len(predict)
#
#
# maps = {}
#
# start = time.time()
#
# for filename in os.listdir(str(path) + "\\ME-MAP/maps"):
#     map_name = filename.split('.')[0]
#     post = class_diagram("ME-MAP/maps/" + filename, nlp)
#     maps[map_name] = post
# end = time.time()
# print(end - start)
#
# ths = [0.45, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
# ks = [1, 2, 3, 4, 5]
# complexity = ['Easy', 'Medium', 'Hard', 'MultiPath']
# hops = [1, 0.9, 0.8, 0.7, 0.6]
#
# # ths = [0.65]
# # ks = [2]
# # complexity = ['Easy', 'Medium', 'Hard', 'MultiPath']
# # hops = [0.9]
#
# df_results = {'complexity': [], 'hops': [], 'th': [], 'k': [], 'ExactMRR': [], 'DomainMrr': [], 'AvgSemantic': [],
#               'Time': [], 'Recall5': []}
# for th in ths:
#     for k in ks:
#         for hop in hops:
#             for level in complexity:
#                 upperbound = []
#                 data_result_search = {}
#                 for queryname in os.listdir("ME-MAP/queries/" + level + "/"):
#                     output = []
#                     query = class_diagram("ME-MAP/queries/" + level + "/" + queryname, nlp)
#                     key = queryname.split('.')[0]
#                     data_result_search[key] = []
#                     times = []
#                     tot_time = 0
#                     for key_map in maps.keys():
#
#                         map_name = key_map
#                         post = maps[key_map]
#                         start = time.time()
#                         results = greedy_algorithm_recursive(post, query, th, k, hop)
#                         end = time.time()
#                         tot_time = tot_time + (end - start)
#                         outputs = read_results(results)
#                         if outputs:
#                             for output in outputs:
#                                 data_result_search[key].append((output, map_name))
#                     print('++++++++++++')
#                     times.append(tot_time)
#
#                 avg_exact, avg_domain, avg_semantic, recall = evaluate(evaluation, data_result_search)
#
#                 df_results['hops'].append(hop)
#                 df_results['complexity'].append(level)
#                 df_results['th'].append(th)
#                 df_results['k'].append(k)
#                 df_results['ExactMRR'].append(avg_exact)
#                 df_results['DomainMrr'].append(avg_domain)
#                 df_results['AvgSemantic'].append(avg_semantic)
#                 df_results['Recall5'].append(recall)
#                 df_results['Time'].append(Average(times))
#
#                 print(avg_exact, avg_domain, avg_semantic)
#
# df = pd.DataFrame(df_results)
# df.to_csv('results.csv')
