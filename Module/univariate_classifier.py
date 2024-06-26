import numpy as np

def best_clf_single(X, y):
    '''
     Function to find the best split point and direction for a single feature
     
     Parameters:
     - X: Array of feature values
     - y: Array of corresponding labels (0 or 1)
     
     Output:
     - Tuple of best split point, direction and classification error
    
    '''
    # Initialize variables to keep track of the best split
    min_err = len(X)
    best_pt = None  # Best split point
    direction = None  # Direction of the split (True for Class 0 / Class 1, False for Class 1 / Class 0)
    
    # Iterate through each possible split point
    for i in range(len(X)):
        # Calculate errors for splitting at this point with Class 0 / Class 1
        err0 = np.count_nonzero(np.array(y[:i]) == 1)
        err1 = np.count_nonzero(np.array(y[i:]) == 0)
        total_err = err0 + err1

        # Update best split if the total error is lower
        if min_err > total_err:
            min_err = total_err
            best_pt = X[i]
            direction = True  # Class 0 / Class 1

        # Calculate errors for splitting at this point with Class 1 / Class 0
        err0 = np.count_nonzero(np.array(y[:i]) == 0)
        err1 = np.count_nonzero(np.array(y[i:]) == 1)
        total_err = err0 + err1

        # Update best split if the total error is lower
        if min_err > total_err:
            min_err = total_err
            best_pt = X[i]
            direction = False  # Class 1 / Class 0
    
    # Return the best split point, direction and classification error
    return best_pt, direction, min_err

def single_feature_LOOCVE(X, y):
    '''
     Function to perform Leave-One-Out Cross-Validation Error for a single feature
     
    Parameters:
     - X: Array of feature values
     - y: Array of corresponding labels (0 or 1)
     
     Output:
     - Leave-One-Out Cross-Validation Error 
    '''
    # Sort the feature values and corresponding labels based on the feature values
    indices = np.argsort(X)
    X = X[indices]
    y = y[indices]

    # Initialize error counter
    err = 0

    # Iterate through each data point, leaving one out for testing in each iteration
    for i in range(len(X)):
        out = (X[i], y[i])

        # Create training set without the current data point
        X_train = np.concatenate([X[:i], X[i+1:]])
        y_train = np.concatenate([y[:i], y[i+1:]])

        # Find the best split point and direction for the training set
        sep, dirct, ce = best_clf_single(X_train, y_train)

        # Make a prediction based on the split point and direction
        if dirct:
            pred = 0 if out[0] < sep else 1
        else:
            pred = 1 if out[0] <= sep else 0

        # Update error counter based on the difference between true label and prediction
        err += abs(out[1] - pred)

    # Return the average error over all iterations of the model
    return err / len(X)
