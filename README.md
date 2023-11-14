# Controversy Detection using LLMs
data.csv contains the handpicked training examples. There are 142 examples in total, with 71 being controversial, and 71 non-controversial.

# Instructions to run:
1. Open your terminal and cd to this directory
2. Run 'pip install -r requirements.txt'
3. Add your OpenAI API key to constants.py
4. You can choose which model to run: 
{   1: "Logistic Regression on Edit Counts", 
    2: "Naive Bayes'", 
    3: Logistic Regression on Edit Counts and Content,
    4: Using Controversy Score,
    5: Using Similarity-Based LLMs
    }
    Enter your choice as the number corresponding to each model: i.e. choice = 2 for Naive Bayes'
5. Add the input path (path to data.csv) and output path (path where the wordcloud will be downloaded)
6. If you wish to generate the wordcloud, set WORDCLOUD = True, otherwise False
7. run python3 models.py
