import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import openai
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time

from constants import API_KEY, MODEL_CHOICE, INPUT_PATH, OUTPUT_FOLDER, WORDCLOUD

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def generate_word_clouds(df):
    controversial = list(df[df['Controversial']=='controversial']['Content'])
    text = ""
    for txt in controversial:
        text+=txt

    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    wordcloud.to_file(OUTPUT_FOLDER+"controversial.png")

    non_controversial = list(df[df['Controversial']=='non-controversial']['Content'])
    text = ""
    for txt in non_controversial:
        text+=txt

    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    wordcloud.to_file(OUTPUT_FOLDER+"non_controversial.png")
    
def controversial_score(df):
    m = 0
    for threshold in range(20000, 2000000, 1000):
        cont = df[df['Controversy Score'] > threshold]
        cont2 = df[df['Controversy Score'] <= threshold]
        correct_yes = cont['Controversial'].eq('controversial').sum()
        correct_no = cont2['Controversial'].eq('non-controversial').sum()
        acc = (correct_yes+correct_no)/len(df)
        m = max(m, acc)
    return m

def logistic_regression(df, split = False):
    x = np.array(df['Number of Edits'])
    y = np.array(df['Controversial'])
    # Reshape x to a 2D array as required by scikit-learn
    x = x.reshape(-1, 1)

    # Initialize the logistic regression model
    model = LogisticRegression()

    if split:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
    else:
        model.fit(x, y)
        y_pred = model.predict(x)
        accuracy = accuracy_score(y, y_pred)

    return accuracy

def naive_bayes(df, split = False):
    sentences = df['Content']
    labels = df['Controversial']
    vectorizer = CountVectorizer()
    classifier = MultinomialNB()

    if split:
        sentences_train, sentences_test, labels_train, labels_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)
        X_train = vectorizer.fit_transform(sentences_train)
        X_test = vectorizer.transform(sentences_test)
        classifier.fit(X_train, labels_train)
        labels_pred = classifier.predict(X_test)
        accuracy = accuracy_score(labels_test, labels_pred)
    else:
        X = vectorizer.fit_transform(sentences)
        classifier.fit(X, labels)
        labels_pred = classifier.predict(X)
        accuracy = accuracy_score(labels, labels_pred)

    return accuracy

def logistic_pipelined(df, split = False):
    X_values = df['Number of Edits']
    X_sentences = df['Content']
    y = df['Controversial']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(np.column_stack((X_values, X_sentences)), y, test_size=0.2, random_state=42, shuffle=True)

    # Define the column transformer
    column_transformer = ColumnTransformer(
        transformers=[
            ('values', 'passthrough', [0]),  # Pass through the values column
            ('sentences', CountVectorizer(), 1)  # Use CountVectorizer for the sentences column
        ])

    # Create the pipeline with logistic regression
    pipeline = Pipeline([
        ('preprocessor', column_transformer),
        ('classifier', LogisticRegression())
    ])

    # Train the classifier
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def few_shot(examples, test_article):
    openai.api_key = API_KEY

    prompt = "Classify the following text as either 'controversial' or 'non-controversial':\n"

    # Generate a few-shot prompt from examples
    few_shot_prompt = prompt + "\n".join([f'"Article title: {title}, Number of edits: {edits}, Content: {content[:1500]}" is {label}.' for edits, content, title, label in examples])

    # Complete the few-shot prompt with the input text for classification
    edits, content, title = test_article
    full_prompt = few_shot_prompt + f'\nTherefore, "Article title: {title}, Number of edits: {str(edits)}, Content: {content[:1500]}" is'

    # Call the OpenAI API to get the model's completion
    global timer
    wait_time = max(20-(time.time()-timer), 0)
    time.sleep(wait_time)
    timer = time.time()
    response = openai.Completion.create(
        engine="text-davinci-002",  # Choose the engine based on your requirements
        prompt=full_prompt,
        temperature=0.7,
        max_tokens=100
    )

    # Extract the model's generated text
    generated_text = response['choices'][0]['text'].strip()

    return generated_text

def get_bert_embeddings(data):
    global model
    global tokenizer
    # Tokenize and obtain embeddings for each sentence
    content_embedding = []
    edit_embedding = []
    title_embedding = []

    for edits, content, title in data:
        inputs = tokenizer(content, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        content_embedding.append(embedding)

        edit_embedding.append(edits)

        inputs2 = tokenizer(title, return_tensors="pt", truncation=True)
        outputs2 = model(**inputs2)
        embedding2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()
        title_embedding.append(embedding2)

    content_array = np.vstack(content_embedding)
    edit_array = np.vstack(edit_embedding)
    title_array = np.vstack(title_embedding)
    
    return content_array

def get_most_similar_articles(test_article, all_articles, top_n=5):
    # Get BERT embeddings for all articles
    article_embeddings = get_bert_embeddings(all_articles)
    test_embedding = get_bert_embeddings([test_article])

    # Calculate cosine similarities between the test article and all other articles
    similarities = cosine_similarity(test_embedding, article_embeddings)[0]

    # Get the indices of the top N most similar articles
    most_similar_indices = similarities.argsort()[-top_n:][::-1]

    # Get the actual articles corresponding to the indices
    most_similar_articles = [all_articles[i] for i in most_similar_indices]

    return most_similar_articles

def hash(edits, content, title):
    return str(edits)+'#'+content+"#"+title

def convert(edit, content, title):
    global label_map
    label = label_map[hash(edit, content, title)]
    return (str(edit), content, title, label)

def in_context_LLM(df):
    X_edits = df['Number of Edits']
    X_content = df['Content']
    X_titles = df['Title']
    y = df['Controversial']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(np.column_stack((X_edits, X_content, X_titles)), y, test_size=0.2, random_state=42, shuffle=True)
        
    y_test = np.array(y_test)
    predictions = []
    acc = 0
    for i in tqdm(range(y_test.shape[0])):
        topn_similar_articles = get_most_similar_articles(X_test[i], X_train, top_n=9)
        examples = [convert(*(art)) for art in topn_similar_articles]
        prediction = few_shot(examples, X_test[i])
        predictions.append(prediction)
        if y_test[i]=='controversial':
            if 'controversial' in predictions[i] and 'non-controversial' not in predictions[i]:
                acc+=1
        else:
            if 'non-controversial' in predictions[i]:
                acc+=1

        print(predictions[i], y_test[i])
        
    acc/=y_test.shape[0]
    return acc, predictions


df = pd.read_csv(INPUT_PATH, index_col = 0)
if WORDCLOUD:
    generate_word_clouds(df)
label_map = {}
for row in df.itertuples(index=False):
    label_map[hash(row[1], row[2], row[0])] = row[4]
    
timer = time.time()
if MODEL_CHOICE==1:
    print(logistic_regression(df, True))
elif MODEL_CHOICE==2:
    print(naive_bayes(df, True))
elif MODEL_CHOICE==3:
    print(logistic_pipelined(df, True))
elif MODEL_CHOICE==4:
    print(controversial_score(df))
elif MODEL_CHOICE==5:
    print(in_context_LLM(df))
else:
    print("Invalid Choice")