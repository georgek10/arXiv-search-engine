# search-engine
A simple search engine that retrieves academic papers from arXiv.org based on user queries and displays them.

# Features
- Web Scraper: Given an initial query it fetches a list of academic papers (50) and creates a data set with information such as the title, authors and abstract for each paper. Moreover, it saves the contents on a json file.
- Text Preprocessing: Using techniques such as tokenization, lemmatization, and stopword removal to improve search efficiency.
- Inverted Indexing: Creates an index for each key word's position in the documents it appears in and used for faster document retrieval.
- Search Algorithms: The user can select between Boolean Retrieval, Vector Space Model (TF-IDF) and BM25 Probabilistic Retrieval for document retrieval.
  Last two also calculate the score of each document based on their relevancy to the user query and rank them accordingly.
- Filtering: The filtering option supported is by author and the user is prompted to filter after receiving the search results.

# To run
Make sure to have python 3.x installed in order to run the source code.
Through the terminal:
1. Enter your search query
2. Choose search retrieval algorithm
3. Choose to filter by author or no
4. Repeat from step one or exit
