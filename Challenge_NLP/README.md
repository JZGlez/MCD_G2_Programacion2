# Challenge 02
# Analysis of comments on Glassdoor. 
# An NLP Pipeline with MLflow application

Based on a dataset "Glassdoor Job Reviews" which contains job descriptions for various industries in the UK.  
This project analyzes the **headline** column of the dataset a general conclusion of the appreciation from other users doing a pre-processing of the text in this column, and the classification and sentiment analysis. 
As classfication label, we will be using the column **recomend** which uses the scale v - Positive, r - Mild, x - Negative, o - No opinion 


## Structure 
- `src/`: contains the main pipeline code. 
- `data/`: dataset used in the analysis. 
- `notebooks/`: exploratory notebooks. 
- `run_pipeline.py`: execution program. 

## Execution

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run pipeline:
   ```bash
   python run_pipeline.py
   ```

3. Run MLflow UI:
   ```bash
   mlflow ui
   ```

## Dataset
Debe tener las columnas:
- `text`: headline column
- `lang`: ''en'
- `label`: recommend column
