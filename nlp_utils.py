from nltk.corpus import stopwords
from langdetect import detect
import spacy

STOPWORDS = set(
    stopwords.words('finnish') +
    stopwords.words('swedish') 
  #  stopwords.words('english')
)

import spacy

nlp_fi = spacy.load("fi_core_news_sm")
nlp_sv = spacy.load("sv_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

def preprocess_text(text):
    lang = detect(text)

    if lang == "fi":
        nlp = nlp_fi
    elif lang == "sv":
        nlp = nlp_sv
    else:     
        nlp = nlp_en

    
    doc = nlp(text)

    tokens = [
    token.lemma_.lower()
    for token in doc
    if not token.is_stop
    and not token.is_punct
    and not token.is_space
    and token.is_alpha
    and len(token) > 2
]

    # lowercase
    #text = text.lower()

    # remove punctuation
    #import re
    #text = re.sub(r"[^\w\s]", "", text)

    # tokenize
    #Xtokens = text.split()

    # remove stopwords
    #tokens = [word for word in tokens if word not in STOPWORDS]

    return tokens

if __name__ == "__main__":
    sample_fi = "Yhtiö on ilmoittanut että se laajentaa toimintaansa Euroopassa"
    sample_swe = "Företaget har meddelat att det vill utöka sin verksamhet i Europa"
    print(preprocess_text(sample_fi))
    print(preprocess_text(sample_swe))
