from Module import univariate_classifier as uc


import pandas as pd



def single_feature(features, df):
    dico = dict()
    for feature in features:
        X, y = df[feature].array, df['target'].array
        loocve = uc.single_feature_LOOCVE(X, y)
        dico[feature] = loocve
    return dico



def heatmap_classic(df, features, cm):
    # Create an empty DataFrame for the heatmap
    heat = pd.DataFrame(index=features, columns=features)
    
    # Calculate univariate contributions for each feature
    dic_uc = single_feature(features, df)
    
    # Iterate over each pair of features
    for feat1 in features:
        for feat2 in features:
            # If the features are the same, set the value in the heatmap to 1 - univariate contribution
            if feat1 == feat2:
                heat.at[feat1, feat2] = 1 - dic_uc[feat1]
                
            else:
                # Iterate over each subgroup (1, 2, 3, 4) and find the maximum LOOCVE value
                max_loocva = 0
                for subgroup in [1, 2, 3, 4]:
                    try:
                        loocva = 1 - cm.at['LOOCVE', f'{feat1}/{feat2}/{subgroup}'] #take the accuracy and not the error
                    except:
                        try:
                            loocva = 1 - cm.at['LOOCVE', f'{feat2}/{feat1}/{subgroup}']
                        except:
                            loocva = 0
                    # Update max_loocve if the current loocve is greater
                    max_loocva = max(max_loocva, loocva)
                
                # Set the values in the heatmap for both (feat1, feat2) and (feat2, feat1)
                heat.at[feat1, feat2] = float(max_loocva)
                heat.at[feat2, feat1] = float(max_loocva)
    
    heat = heat[heat.columns].astype(float)
    return heat


def heatmap_coeff_pair(df, features, cm):
    # Create an empty DataFrame for the heatmap
    heat = pd.DataFrame(index=features, columns=features)
    
    # Calculate univariate contributions for each feature
    dic_uc = single_feature(features, df)
    
    # Iterate over each pair of features
    for feat1 in features:
        for feat2 in features:
            # If the features are the same, set the value in the heatmap to 1 - univariate contribution
            if feat1 == feat2:
                heat.at[feat1, feat2] = 0
                
            else:
                # Iterate over each subgroup (1, 2, 3, 4) and find the maximum LOOCVE value
                max_loocve = 0
                for subgroup in [1, 2, 3, 4]:
                    try:
                        loocve = 1 - cm.at['LOOCVE', f'{feat1}/{feat2}/{subgroup}']
                    except:
                        try:
                            loocve = 1 - cm.at['LOOCVE', f'{feat2}/{feat1}/{subgroup}']
                        except:
                            loocve = 0
                    # Update max_loocve if the current loocve is greater
                    max_loocve = max(max_loocve, loocve)
                
                # Set the values in the heatmap for both (feat1, feat2) and (feat2, feat1)
                heat.at[feat1, feat2] = float(max_loocve) - max([1 - dic_uc[feat1], 1 - dic_uc[feat2]])
                heat.at[feat2, feat1] = float(max_loocve) - max([1 - dic_uc[feat1], 1 - dic_uc[feat2]])
    
    heat = heat[heat.columns].astype(float)
    
    a = min(heat.values.flatten())
    b = max(heat.values.flatten())

    heat = (heat - a)/(b-a)
    return heat


def heatmap_corrected(df, features, cm):
    return heatmap_coeff_pair(df, features, cm) * heatmap_classic(df, features, cm)
 
    