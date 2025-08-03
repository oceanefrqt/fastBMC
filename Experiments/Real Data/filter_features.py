import pandas as pd
import sys


def main():
    file_path = sys.argv[1]      # name the dataset CSV
    output_path = sys.argv[2]    # name of the output CSV
    rang = int(sys.argv[3])      # Number of genes to keep

    df = pd.read_csv(file_path, index_col=0, low_memory=False)

    var = list(df.var(numeric_only=True).sort_values().index)
    var.remove('target')
    filt = list(var[-rang:])
    filt.reverse()

    df2 = df[filt]
    df2['target'] = df['target']
    df2.to_csv(output_path, index=False)
    
    

if __name__ == "__main__":
    main()