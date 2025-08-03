import pandas as pd
import random
import numpy as np
import sys


def compute_pairs(max_pairs, max_c, noise):
    # Initialize a dictionary to store pairs of data for four different orientations
    pairs = {1: list(), 2: list(), 3: list(), 4: list()}
    counter = 0

    # Loop to generate the specified number of pairs
    while counter < max_pairs:
        # Randomly choose an orientation (1 through 4)
        tt = random.randint(1, 4)

        # Call the appropriate function based on the chosen orientation
        if tt == 1:
            data = orient_1(max_c, noise)
        elif tt == 2:
            data = orient_2(max_c, noise)
        elif tt == 3:
            data = orient_3(max_c, noise)
        elif tt == 4:
            data = orient_4(max_c, noise)

        # Append the generated data to the corresponding list in the dictionary
        pairs[tt].append(data)
        counter += 1

    # Return the dictionary containing the pairs
    return pairs


# The orientations correspond to the four possible orientations of each of the two axes.

def orient_1(max_c, noise_rate):
    # Generate a list of random delimitation points
    del_points = []
    for _ in range(random.randint(3, 8)):  # Generate 3 to 8 delimitation points
        x = random.random()
        y = random.random()
        del_points.append([x, y])

    # Initialize data list and counters for positive and negative samples
    data = []
    c_0 = 0  # Counter for negative samples
    c_1 = 0  # Counter for positive samples

    # Loop to generate the specified number of samples
    while c_0 < max_c or c_1 < max_c:
        x = random.random()
        y = random.random()
        label = 1  # Default label is positive

        # Check if the point falls below any of the delimitation points
        for dp in del_points:
            z1 = dp[0]
            z2 = dp[1]
            if x < z1 and y < z2:
                label = 0  # If it does, mark it as negative

        # Introduce noise by flipping the label with some probability
        if random.random() < noise_rate: 
            label = 1 - label  # Flip the label

        # Append the point to the data list with its label
        if label == 0:
            c_0 += 1
            if c_0 <= max_c:
                data.append([x, y, label])
        else:
            c_1 += 1
            if c_1 <= max_c:
                data.append([x, y, label])

    # Return the generated data
    return data

def orient_2(max_c, noise_rate):
    del_points = []
    # Generate 3 to 8 delimitation points
    for _ in range(random.randint(3, 8)):
        x = random.random()
        y = random.random()
        del_points.append([x, y])

    data = []
    c_0 = 0
    c_1 = 0

    while c_0 < max_c or c_1 < max_c:
        x = random.random()
        y = random.random()
        label = 1

        # Check if the point falls in the specified region
        for dp in del_points:
            z1 = dp[0]
            z2 = dp[1]
            if x > z1 and y < z2:
                label = 0

        if random.random() < noise_rate:  
            label = 1 - label

        if label == 0:
            c_0 += 1
            if c_0 <= max_c:
                data.append([x, y, label])
        else:
            c_1 += 1
            if c_1 <= max_c:
                data.append([x, y, label])

    return data

def orient_3(max_c, noise_rate):
    del_points = []
    # Generate 3 to 8 delimitation points
    for _ in range(random.randint(3, 8)):
        x = random.random()
        y = random.random()
        del_points.append([x, y])

    data = []
    c_0 = 0
    c_1 = 0

    while c_0 < max_c or c_1 < max_c:
        x = random.random()
        y = random.random()
        label = 1

        # Check if the point falls in the specified region
        for dp in del_points:
            z1 = dp[0]
            z2 = dp[1]
            if x > z1 and y > z2:
                label = 0

        if random.random() < noise_rate:  
            label = 1 - label

        if label == 0:
            c_0 += 1
            if c_0 <= max_c:
                data.append([x, y, label])
        else:
            c_1 += 1
            if c_1 <= max_c:
                data.append([x, y, label])

    return data

def orient_4(max_c, noise_rate):
    del_points = []
    # Generate 3 to 8 delimitation points
    for _ in range(random.randint(3, 8)):
        x = random.random()
        y = random.random()
        del_points.append([x, y])

    data = []
    c_0 = 0
    c_1 = 0

    while c_0 < max_c or c_1 < max_c:
        x = random.random()
        y = random.random()
        label = 1

        # Check if the point falls in the specified region
        for dp in del_points:
            z1 = dp[0]
            z2 = dp[1]
            if x < z1 and y > z2:
                label = 0

        if random.random() < noise_rate: 
            label = 1 - label

        if label == 0:
            c_0 += 1
            if c_0 <= max_c:
                data.append([x, y, label])
        else:
            c_1 += 1
            if c_1 <= max_c:
                data.append([x, y, label])

    return data


def main():
    output_path = sys.argv[1]    # name of the output directory
    
    for max_feat in [30, 50, 100, 150, 200]:
        for noise_rate in [0.05, 0.1, 0.2, 0.5]:
            for max_c in [15,25,50, 80]:
                try:

                    df = pd.DataFrame()

                    pairs = compute_pairs(15, max_c, noise_rate)

                    name_pairs = ['Gene1/Gene2/1']

                    df = pd.DataFrame({'Gene1':np.asarray(pairs[1][0])[:,0], 'Gene2': np.asarray(pairs[1][0])[:,1], 'target': np.asarray(pairs[1][0])[:,2]})
                    df.sort_values(by=['target'], inplace=True)
                    df.reset_index(drop=True, inplace=True)


                    pairs[1].remove(pairs[1][0])
                    c = 3




                    for tt in pairs.keys():
                        for pair in pairs[tt]:
                            temp = pd.DataFrame({f'Gene{c}':np.asarray(pair)[:,0], f'Gene{c+1}': np.asarray(pair)[:,1], 'target2': np.asarray(pair)[:,2]})
                            temp.sort_values(by=['target2'], inplace=True)
                            temp.reset_index(drop=True, inplace=True)
                            temp.drop(['target2'],axis=1, inplace=True)
                            df = pd.concat([df, temp], axis=1)

                            name_pairs.append(f'Gene{c}/Gene{c+1}/{tt}')



                            c+=2


                    for k in range(c, max_feat+1):
                        df[f'Gene{k}'] = np.random.random(max_c*2)



                    df.to_csv(f"{output_path}/simulated_data_noise{noise_rate}_feat{max_feat}_obs{max_c*2}.csv")
                except:
                    print(f'Error when simulating data with noise:{noise_rate}, feat: {max_feat}, and obs:{max_c*2}')
                    pass
                
                

if __name__ == "__main__":
    main()
