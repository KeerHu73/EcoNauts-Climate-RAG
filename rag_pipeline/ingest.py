import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

DATA_PATH = Path("data/etl_cleaned_dataset.csv")

def load_cleaned_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["country"].isin(["United States", "Canada"])]
    df = df[df["year"] >= 1990]
    return df

def build_country_documents(df: pd.DataFrame) -> List[Dict[str, Any]]:
    documents = []
    for country, sub in df.groupby("country"):
        sub = sub.sort_values("year")
        lines = []
        lines.append(f"Country: {country}")
        lines.append(f"Years covered: {sub['year'].min()}–{sub['year'].max()}")
        lines.append("")
        lines.append("Annual CO2 emissions (million tonnes):")
        for _, row in sub.iterrows():
            lines.append(f"{int(row['year'])}: {row['co2']:.2f}")
        lines.append("")
        lines.append("Annual temperature anomaly (°C):")
        for _, row in sub.iterrows():
            lines.append(f"{int(row['year'])}: {row['temp_anomaly']:.3f}")

        doc = {
            "text": "\n".join(lines),
            "metadata": {
                "source": "etl_cleaned_dataset.csv",
                "country": country
            }
        }
        documents.append(doc)

    return documents

if __name__ == "__main__":
    df = load_cleaned_data()
    docs = build_country_documents(df)
    print(f"Built {len(docs)} documents")
    print("\nExample document:\n")
    print(docs[0]["text"][:1200])
