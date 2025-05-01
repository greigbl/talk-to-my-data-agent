import pandas as pd
import spacy
from collections import Counter
from wordcloud import WordCloud
import os
import streamlit as st
from dotenv import load_dotenv
from datarobot_drum.runtime_parameters import RuntimeParameters


load_dotenv(".env")

#WORDCLUD_FONT
try:
    font_path = RuntimeParameters.get("WORDCLOUD_FONT")
except Exception as e:
    font_path = os.environ.get("WORDCLOUD_FONT", None)
print("FP:",font_path)


def _tokenize_japanese(text:str) -> list[str]:
    nlp = spacy.load("ja_core_news_sm-3.8.0/ja_core_news_sm/ja_core_news_sm-3.8.0", disable=["ner", "tagger","parser","textcat"])
    from spacy.lang.ja import stop_words
    stopwords = list(stop_words.STOP_WORDS) + ["の","どう","か","が","って","?","？","何","場合","的","ましょう"]
    CHUNK_SIZE = 1_000
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    docs = []
    for chunk in chunks:
        doc = nlp(chunk)
        docs.extend([t.text for t in doc if t.text.lower() not in stopwords and t.is_alpha])
    return docs

def _tokenize_english(text:str) -> list[str]:
    from spacy.lang.en import stop_words
    tokens = text.split(" ")
    tokens = [t.lower() for t in tokens if t.lower() not in stop_words.STOP_WORDS]
    return tokens

def _get_wordcloud(tokens):
    from wordcloud import WordCloud
    token_counts = Counter(tokens)
    wordcloud = WordCloud(
        max_words=200,
        font_path=font_path,
        height=600,
        width=1200,
    ).generate_from_frequencies(token_counts)
    return wordcloud


def create_wordloud(df:pd.DataFrame, column:str, language:str="ja") -> str:
    """
     Generate a word cloud for a specific column of a Pandas dataframe in a given language.
    
    #     Args:
    #         df (pd.DataFrame): Input DataFrame.
    #         column (str): target column with text in it.
    #         language (str): Input language.
    
    #     Returns:
    #         A string (base64) representation of the wordcloud image (JPEG).    
    """
    if language == "ja":
        st.toast("⏱️ Generating word clouds for large documents may be slow")
        all_text = "".join([str(i) for i in df[column].tolist()])
        tokens = _tokenize_japanese(all_text)
    else:
        all_text = ".".join([str(i) for i in df[column].tolist()])  
        tokens = _tokenize_english(all_text)
    
    final_wordcloud =  _get_wordcloud(tokens)
    import base64
    from io import BytesIO
    buffered = BytesIO()
    final_wordcloud.to_image().save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue())