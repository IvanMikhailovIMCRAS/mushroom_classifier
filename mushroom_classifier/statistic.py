import pandas as pd
from tabulate import tabulate

if __name__ == '__main__':
    df = pd.read_csv("spectra_metadata.csv")
    # print(df.head(10))
    print(df.describe())
    # print(df['species'].value_counts(normalize=False).sort_index())
    # print(df.groupby(['species', 'part']).size().unstack(fill_value=0))
    
    result = pd.crosstab(index=df['species'], columns=df['part'])

    # Сброс индекса, чтобы сделать species колонкой
    result = result.reset_index()
    # Временная копия с новым индексом для вывода
    result_reindexed = result.copy()
    result_reindexed.index = range(1, len(result_reindexed) + 1)

    # Выводим с tabulate
    print(tabulate(result_reindexed, headers='keys', tablefmt='psql', showindex=True))
    
    