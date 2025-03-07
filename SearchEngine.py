import requests
from bs4 import BeautifulSoup
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from math import log

#Step 1: Web Crawler
def search_arxiv(query):
    max_results=50
    params = {
        'searchtype': 'all',
        'query': query,
        'max_results': max_results
    }

    base_url = 'https://arxiv.org/search/'
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        return response.content
    else:
        return None

def parse_arxiv_results(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    search_results = []
    for entry in soup.select('.arxiv-result'):
        title = entry.select_one('.title').get_text(strip=True)
        authors = [author.get_text(strip=True) for author in entry.select('.authors a')]
        abstract = entry.select_one('.abstract-full').get_text(strip=True)

        search_result = {
            'title': title,
            'authors': authors,
            'abstract': abstract
        }

        search_results.append(search_result)

    return search_results

def save_results_to_json(search_results):
    filename='dataset.json'
    with open(filename, 'w') as json_file:
        json.dump(search_results, json_file, indent=2)

#Step 2: Pre-processing text
def preprocess_text(text):
    tokens = word_tokenize(text)
    
    tokens = [token for token in tokens if token.isalnum()]
    
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words]

    wnl = WordNetLemmatizer()
    tokens = [wnl.lemmatize(token) for token in tokens]

    return tokens

#Step 3: Indexing
def create_inverted_index(documents):
    inverted_index = {}
    
    for doc_id, document in enumerate(documents):
        tokens = preprocess_text(document['abstract'])
        
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append(doc_id)
    return inverted_index

#Step 4: Search Engine
def search_engine():
    while True:
        user_query = input("\nEnter your search query or type 'exit' if you want to quit: ")

        if user_query.lower() == 'exit':
            break
        
        print("\nChoose a search retrieval algorithm:")
        print("1. Boolean Retrieval")
        print("2. Vector Space Model (VSM)")
        print("3. Probabilistic Retrieval (Okapi BM25)")
        chosen_algorithm = int(input("Enter the number of your choice (1 or 2 or 3): "))
        
        if chosen_algorithm == 1:
            if ' AND ' in user_query or ' OR ' in user_query or ' NOT ' in user_query or user_query.startswith('NOT '):
                matching_results = boolean_retrieval(user_query, inverted_index, search_results)
            else:
                tokens = preprocess_text(user_query)
                matching_papers = set()
        
                for token in tokens:
                    if token in inverted_index:
                        matching_papers.update(inverted_index[token])
        
                matching_results  = [search_results[i] for i in matching_papers]
        
            if not matching_results:
                print("\nNo matching results found.")
            else:
                print("\nMatching Results (Unfiltered):\n")
                for i, match_result in enumerate(matching_results , start=1):
                    print(f"\nMatch Result {i}:")
                    print(f"\nTitle: {match_result['title']}")
                    print(f"\nAuthors: {', '.join(match_result['authors'])}")
                    print(f"\nAbstract: {match_result['abstract']}")
                    print("\n" + "=" * 100 + "\n")

        elif chosen_algorithm == 2:
            ranked_documents_tf_idf = vsm_retrieval(user_query, search_results)
            print("\nTF-IDF Ranking Results (Unfiltered):\n")
            for i, (doc_id, score) in enumerate(ranked_documents_tf_idf, start=1):
                if score > 0:
                    match_result = search_results[doc_id - 1]
                    print(f"\nRank {i} (TF-IDF Score: {score}):")
                    print(f"\nTitle: {match_result['title']}")
                    print(f"\nAuthors: {', '.join(match_result['authors'])}")
                    print(f"\nAbstract: {match_result['abstract']}")
                    print("\n" + "=" * 100 + "\n")

        elif chosen_algorithm == 3:
            ranked_documents_bm25 = bm25_retrieval(user_query, search_results)
            print("\nBM25 Ranking Results (Unfiltered):\n")
            for i, (doc_id, score) in enumerate(ranked_documents_bm25, start=1):
                if score > 0:
                    match_result = search_results[doc_id - 1]
                    print(f"\nRank {i} (BM25 Score: {score}):")
                    print(f"\nTitle: {match_result['title']}")
                    print(f"\nAuthors: {', '.join(match_result['authors'])}")
                    print(f"\nAbstract: {match_result['abstract']}")
                    print("\n" + "=" * 100 + "\n")
            
        apply_filters = input("Do you want to filter by author? (yes/no): ").lower() == 'yes'
        if apply_filters:
            author_name = input("Enter author's name: ")
            if chosen_algorithm == 1:
                filtered_results = filter_by_author(matching_results, author_name)
                if not filtered_results:
                    print("\nNo matching results found after filtering by author.")
                else:
                    print("\nMatching Results (Filtered):\n")
                    for i, match_result in enumerate(filtered_results , start=1):
                        print(f"\nMatch Result {i}:")
                        print(f"\nTitle: {match_result['title']}")
                        print(f"\nAuthors: {', '.join(match_result['authors'])}")
                        print(f"\nAbstract: {match_result['abstract']}")
                        print("\n" + "=" * 100 + "\n")

            elif chosen_algorithm == 2:
                filtered_results = filter_by_author(search_results, author_name)
                ranked_documents_tf_idf = vsm_retrieval(user_query, filtered_results)
                if not filtered_results:
                    print("\nNo matching results found after filtering by author.")
                else:
                    print("\nTF-IDF Ranking Results (Filtered):\n")
                    for i, (doc_id, score) in enumerate(ranked_documents_tf_idf, start=1):
                        if score > 0:
                            match_result = filtered_results[doc_id - 1]
                            print(f"\nRank {i} (TF-IDF Score: {score}):")
                            print(f"\nTitle: {match_result['title']}")
                            print(f"\nAuthors: {', '.join(match_result['authors'])}")
                            print(f"\nAbstract: {match_result['abstract']}")
                            print("\n" + "=" * 100 + "\n")

            elif chosen_algorithm == 3:
                filtered_results = filter_by_author(search_results, author_name)
                ranked_documents_bm25 = bm25_retrieval(user_query, filtered_results)
                if not filtered_results:
                    print("\nNo matching results found after filtering by author.")
                else:
                    print("\nBM25 Ranking Results (Filtered):\n")
                    for i, (doc_id, score) in enumerate(ranked_documents_bm25, start=1):
                        if score > 0:
                            match_result = filtered_results[doc_id - 1]
                            print(f"\nRank {i} (BM25 Score: {score}):")
                            print(f"\nTitle: {match_result['title']}")
                            print(f"\nAuthors: {', '.join(match_result['authors'])}")
                            print(f"\nAbstract: {match_result['abstract']}")
                            print("\n" + "=" * 100 + "\n")

def boolean_retrieval(query, inverted_index, documents):
    query_tokens = preprocess_boolean_query(query)
    
    result_set = set(range(len(documents)))
    current_operation = 'AND'

    for token in query_tokens:
        if token in ['AND', 'OR', 'NOT']:
            current_operation = token
        elif token in inverted_index:
            token_result_set = set(inverted_index[token])

            if current_operation == 'AND':
                result_set.intersection_update(token_result_set)
            elif current_operation == 'OR':
                result_set.update(token_result_set)
            elif current_operation == 'NOT':
                result_set.difference_update(token_result_set)

    matching_results = [documents[i] for i in result_set]
    
    return matching_results

def preprocess_boolean_query(boolean_query):
    tokens = word_tokenize(boolean_query)

    tokens = [token.upper() for token in tokens]

    tokens = [token for token in tokens if token.isalnum()]

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token in ['AND', 'OR', 'NOT'] or token not in stop_words]
    tokens = [token if token in ['AND', 'OR', 'NOT'] else token.lower() for token in tokens]
    
    wnl = WordNetLemmatizer()
    tokens = [wnl.lemmatize(token) if token not in ['AND', 'OR', 'NOT'] else token for token in tokens]

    return tokens

def vsm_retrieval(query, documents):
    query_tokens = preprocess_text(query)
    
    query_string = ' '.join(query_tokens)

    corpus = [query_string] + [' '.join(preprocess_text(doc['abstract'])) for doc in documents]

    vectorizer = TfidfVectorizer()
    tfidf_space = vectorizer.fit_transform(corpus)

    similarity_scores = cosine_similarity(tfidf_space)

    query_similarity_scores = similarity_scores[0, 1:]

    ranked_documents = sorted(enumerate(query_similarity_scores, start=1), key=lambda x: x[1], reverse=True)

    return ranked_documents

def bm25_retrieval(query, documents):
    k1 = 1.2
    b = 0.75

    query_tokens = preprocess_text(query)
    corpus = [preprocess_text(doc['abstract']) for doc in documents]

    if len(corpus) == 0:
        avg_dl = 0
    else:
        avg_dl = sum(len(doc) for doc in corpus) / len(corpus)

    scores = []
    for doc_id, doc in enumerate(corpus, start=1):
        score = 0
        for token in query_tokens:
            tf = doc.count(token)
            idf = log((len(corpus) - corpus.count(token) + 0.5) / (corpus.count(token) + 0.5) + 1.0)
            score += (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (len(doc) / avg_dl))) * idf

        scores.append((doc_id, score))

    ranked_documents = sorted(scores, key=lambda x: x[1], reverse=True)
    
    return ranked_documents

def filter_by_author(results, author_name):
    formatted_author_name = author_name.lower()
    
    filtered_results = [
        result for result in results if any(formatted_author_name in author.lower() for author in result['authors'])
    ]
    
    return filtered_results

#main
query = 'machine learning'
html_content = search_arxiv(query)

if html_content:
    search_results = parse_arxiv_results(html_content)
    
    save_results_to_json(search_results)
    
    inverted_index = create_inverted_index(search_results)
    
    search_engine()
else:
    print('Error in request')
