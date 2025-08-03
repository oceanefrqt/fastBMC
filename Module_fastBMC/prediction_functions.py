# prediction_functions.py

## PREDICTION OF A POINT

def predict_uncertainty(p, bpr, bpb, rev, up):
    """
    Predict the label for a given point.

    Parameters:
    - p: Point to predict (tuple of x, y coordinates).
    - bpr: Breakpoints for the red area.
    - bpb: Breakpoints for the blue area.
    - rev: Boolean indicating whether to test increasing or decreasing isotonicity.
    - up: Boolean indicating whether to test for the upper or lower boundary.

    Returns:
    - Label for the point: 1 for red, 0 for blue, or -1 for uncertainty.
    """
    flag = -1

    if not rev and up:  # CASE 1
        for b in bpr:
            if p[0] >= b[0] and p[1] >= b[1]:
                flag = 1

        for b in bpb:
            if p[0] <= b[0] and p[1] <= b[1]:
                flag = 0

    elif rev and up:  # CASE 2
        for b in bpr:
            if p[0] <= b[0] and p[1] >= b[1]:
                flag = 1

        for b in bpb:
            if p[0] >= b[0] and p[1] <= b[1]:
                flag = 0

    elif not rev and not up:  # CASE 3
        for b in bpr:
            if p[0] <= b[0] and p[1] <= b[1]:
                flag = 1

        for b in bpb:
            if p[0] >= b[0] and p[1] >= b[1]:
                flag = 0

    elif rev and not up:  # CASE 4
        for b in bpr:
            if p[0] >= b[0] and p[1] <= b[1]:
                flag = 1

        for b in bpb:
            if p[0] <= b[0] and p[1] >= b[1]:
                flag = 0

    assert flag in [1, 0, -1], "Problem with prediction of the point, label is not 0, 1 or -1"

    return flag


def predict_fav_class1(p, bpr, bpb, rev, up):
    """
    Predict label for a given point, while favoring class 1.

    Parameters:
    - p: Point to predict (tuple of x, y coordinates).
    - bpr: Breakpoints for the class 1 area.
    - bpb: Breakpoints for the class 0 area.
    - rev: Boolean indicating whether to test increasing or decreasing isotonicity.
    - up: Boolean indicating whether to test for the upper or lower boundary.

    Returns:
    - Label for the point: 0 for blue, 1 for red.
    """
    # Points in the grey area are automatically labeled in the blue area.
    # Points in the grey area are automatically labeled in the red area.
    flag = predict_uncertainty(p, bpr, bpb, rev, up)
    if flag == -1:
        flag = 1

    assert flag in [1, 0], "Problem with prediction of the point, label is not 0 or 1"

    return flag


def predict_fav_class0(p, bpr, bpb, rev, up):
    
    """
    Predict label for a given point, while favoring class 0.

    Parameters:
    - p: Point to predict (tuple of x, y coordinates).
    - bpr: Breakpoints for the class 1 area.
    - bpb: Breakpoints for the class 0 area.
    - rev: Boolean indicating whether to test increasing or decreasing isotonicity.
    - up: Boolean indicating whether to test for the upper or lower boundary.

    Returns:
    - Label for the point: 0 for blue, 1 for red.
    """
    # Points in the grey area are automatically labeled in the blue area.
    flag = predict_uncertainty(p, bpr, bpb, rev, up)
    if flag == -1:
        flag = 0

    assert flag in [1, 0], "Problem with prediction of the point, label is not 0 or 1"

    return flag


## MAJORITY VOTE PREDICTION BASED ON A LIST OF PREDICTIONS 

def pred_ensemble_model(preds):
    """
    Predict the majority label based on the predictions from classifiers.

    Parameters:
    - preds: Array with predicted labels from classifiers.

    Returns:
    - Majority vote prediction (0 or 1).
    """
    count_1 = preds.count(1)
    count_0 = preds.count(0)
    count_minus_1 = preds.count(-1)
    
    assert count_1 + count_0 + count_minus_1 == len(preds), "Problem in the error predictions for a patient"
    
    if count_minus_1 == len(preds):
        # In the case where all the predictions are equal to -1, predict the patient as 1
        return 1
    
    # If less 1 than 0, predict as 0
    if count_1 < count_0:
        return 0
    # If more or same number of 1 as number of 0, predict as 1
    else:
        return 1

def pred_ensemble_model_2(proba, thresh):
    """
    Predict the majority label based on a probability threshold.

    Parameters:
    - proba: Probability of being class 1.
    - thresh: Threshold for predicting class 1.

    Returns:
    - Majority vote prediction (0 or 1).
    """
    # If less 1 than 0, predict as 0
    if proba < thresh:
        return 0
    # If more or same probability of 1 as probability of 0, predict as 1
    else:
        return 1

def proba_ensemble_model(preds):
    """
    Calculate the probability of being class 1 based on predictions from classifiers.

    Parameters:
    - preds: Array with predicted labels from classifiers.

    Returns:
    - Probability of being class 1.
    """
    count_1 = preds.count(1)
    count_0 = preds.count(0)
    count_minus_1 = preds.count(-1)
    
    assert count_1 + count_0 + count_minus_1 == len(preds), "Problem in the probabilities for a patient"
    
    if count_1 + count_0 != 0:
        # Probability of getting a 1
        return count_1 / (count_1 + count_0)
    else:
        return -1

