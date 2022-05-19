def recall_at_k(true_list, predicted_list, k=1000):
    """
    This function calculate the recall@k metric.

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, recall@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    try:
        tp = [x for x in predicted_list[:k] if x in true_list]
        res = round(len(tp) / len(true_list), 3)
        return res if res < 1 else 1
    except:
        return 0



def precision_at_k(true_list, predicted_list, k=1000):
    """
    This function calculate the precision@k metric.

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, precision@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    try:
        new_predicted = predicted_list[:k]
        relevants = [x for x in new_predicted if x in true_list]
        res = round(len(relevants) / len(new_predicted), 3)
        return res if res < 1 else 1
    except:
        return 0


def f1(precision, recall):
    try:
        return 2 * (precision * recall) / (precision + recall)
    except:
        return 0
