def describe(df):
    return df.describe()

def missing_values(df):
    return df.isnull().sum()
