# News RAG

This demo is an example of a simple RAG implementation from scraping a news (Wordpress) site.

## Usage
To try this out, you'll need to create a Vector database by scraping a website. You can do so using the `scraper` script:

```
# Create virtual environment and install requirements
virtualenv .venv
pip3 install -r requirements.txt
source .venv/bin/activate

# Set the Wordpress URL
export WP_BASE_URL=""

# Scrape the Wordpress site and populate the posts locally
# By default this will only scrape two pages (to discourage abuse), 
# but you can adjust as needed
python3 scraper.py
```

After the files are loaded locally in the `posts` directory, you can use the RAG script to test out the questions. This will create a local vectorstore and provide the given questions to the LLMs after fetching the most relevant posts using a similarity search in the Vector database.

You can change the questions in the script.
```
opeyemionikute$ python3 rag.py

Question: What is an example of a fraud case in Kenya?
Creating vector store...
Loading existing vector store...
Setting up QA chain...
[...]

> Finished chain.

Answer: 

One example of a fraud case in Kenya is when one of the country's largest lenders lost KSh1.5 billion to fraud, as reported in 2020. This was mentioned in the article "Africa's Fraud Landscape: The Rise of AI-Powered Attacks" (https://www.smileidentity.com/africas-fraud-landscape-the-rise-of-ai-powered-attacks/).

Sources:
- Sample article
  URL: https://website.com
```

## Notes
- Creating the vector database can take some time. For a full dataset of 1000 posts, we'd probably need 30 minutes to one hour.