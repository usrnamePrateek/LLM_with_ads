import pandas as pd

URL = "https://raw.githubusercontent.com/patcg-individual-drafts/topics/main/taxonomy_v2.md"

df = pd.read_csv(
    URL,
    sep="|",
    skipinitialspace=True
)

# Clean Markdown table formatting
df.columns = [c.strip() for c in df.columns]
df = df.iloc[:, :2]

df["ID"] = (
    df["ID"]
    .astype(str)
    .str.strip()
    .astype(int)
)

df["Topic"] = (
    df["Topic"]
    .astype(str)
    .str.strip()
)

print(df.head())
print(f"Total topics: {len(df)}")
