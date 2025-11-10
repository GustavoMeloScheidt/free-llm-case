import sqlite3, pandas as pd, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--db", default="data/db.sqlite")
parser.add_argument("--out", default="data/csat.csv")
args = parser.parse_args()

con = sqlite3.connect(args.db)
df = pd.read_sql_query("SELECT * FROM csat_extract", con)
df.replace({"\\N": None}, inplace=True) #replace \N with null values 
print(f" Rows: {len(df)}, Colums: {list(df.columns)}")
df.to_csv(args.out, index=False)
print(df.shape, df.columns.tolist())
