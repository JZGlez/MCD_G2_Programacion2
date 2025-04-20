
import pandas as pd
from src.main import run_nlp_pipeline

# ----------------------------------------------------------------------------------------------------------------
# |                                      Natural Langauge Processing                                             |
# ----------------------------------------------------------------------------------------------------------------

# Dataframe loading
df_original = pd.read_csv("data/glassdoor_reviews.csv")

# Dataframe cleaning
df = df_original.copy() 
df.drop(df[df['recommend'] == 'o'].index, inplace=True)
df.dropna(subset=['recommend'], inplace=True)

# Sampling of the dataframe
df_sample = df.sample(n=2000, random_state=42)

# Pipeline execution
run_nlp_pipeline(df_sample, label_col='recommend')
