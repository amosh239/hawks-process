import pandas as pd

def load_and_prep_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['CustomerID'])
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNo'].str.startswith('C')]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    transactions = df.groupby(['CustomerID', 'InvoiceNo'])['InvoiceDate'].first().reset_index()
    return transactions.groupby('CustomerID')['InvoiceDate'].apply(lambda x: sorted(list(x)))