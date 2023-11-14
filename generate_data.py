import urllib.parse
import mwclient
import requests
import random
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

def get_content(url):
    r = requests.get(url)

    # Get body content
    soup = BeautifulSoup(r.text,'html.parser').select('body')[0]

    # Initialize variable
    content = ""

    # Iterate through all tags
    for tag in soup.find_all():
        if tag.name=="p":
            content+=(tag.text).strip()

    content = content.replace('\n', " ").replace('  ', " ").strip()

    return content

def get_links_from_wikipedia_page(url, max_links_per_topic=5):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    topics = {}
    topic_sections = soup.find_all('span', {'class': 'mw-headline'})

    for section in topic_sections:
        topic_name = section.get_text()
        ul_tag = section.find_next('ul')

        if ul_tag:
            links_data = [{'text': a.get_text(), 'link': a['href']} for a in ul_tag.find_all('a', href=True) if not a['href'].startswith('#')]
            topics[topic_name] = links_data[:max_links_per_topic]

    return topics

def get_total_number_of_edits(article_title):
    """
    Get the total number of edits for a given Wikipedia article using mwclient.
    """
    # Connect to Wikipedia
    site = mwclient.Site('en.wikipedia.org')

    # Initialize count
    total_revisions = 0
    edit_count = {}
    revisions = []
    reverts = set()
    mutual_reverts = {}
    m = 0

    try:
        # Fetch the page
        page = site.pages[article_title]
        # Loop through revisions (mwclient handles continuation internally)
        for revision in page.revisions(prop = 'ids|timestamp|flags|comment|user|sha1'):
            user_id = revision['user']
            total_revisions += 1
            if 'sha1' in revision:
                content = revision['sha1']
                for old_content, old_user_id in revisions:
                    if old_content==content:
                        if user_id + "#" + old_user_id in reverts:
                            mutual_reverts[user_id + "#" + old_user_id] = 0  
                        reverts.add(old_user_id + "#" + user_id)
                revisions.append((content, user_id))
            edit_count[user_id] = edit_count[user_id]+1 if user_id in edit_count else 1

        for p in mutual_reverts:
            users = p.split('#')
            mutual_reverts[p] = min(edit_count[users[0]], edit_count[users[1]])
        vals = list(mutual_reverts.values())
        vals.sort()
        vals = vals[:-1]
        s = sum(vals)
        m = len(edit_count)*s 
    except:
        pass

    return total_revisions, m

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Wikipedia:List_of_controversial_issues"

# Assuming you have already defined and run the get_links_from_wikipedia_page() function.

topics_and_links = get_links_from_wikipedia_page(WIKIPEDIA_URL, 5)

article_to_edits = {}
article_to_topics = {}
article_to_links = {}
article_to_contr = {}

for topic, links_data in tqdm(topics_and_links.items()):
    for linked_data in links_data:
        num_edits, controversy_score = get_total_number_of_edits(linked_data['text'])
        article_to_edits[linked_data['text']] = num_edits
        article_to_topics[linked_data['text']] = topic
        article_to_links[linked_data['text']] = linked_data['link']
        article_to_contr[linked_data['text']] = controversy_score

data = {'article_title': list(article_to_edits.keys()),
        'num_edits': list(article_to_edits.values()),
        'topic': list(article_to_topics.values()),
        'Controversy Score': list(article_to_contr.values()),
        'url': list(article_to_links.values())}
df = pd.DataFrame.from_dict(data)

l = []
for article, link in article_to_links.items():
  l.append(get_content("https://en.wikipedia.org"+link))
df['content'] = l
df.rename(columns = {'article_title':'Title', 'num_edits':'Number of Edits', 'topic':'Category', 'Controversy Score':'Controversy Score', 'url': 'Article Link', 'content': 'Content'}, inplace = True)
df.head(-4).to_csv('article_data.csv')
