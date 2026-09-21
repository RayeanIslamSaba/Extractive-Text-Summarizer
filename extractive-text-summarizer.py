# tkinter libraries for GUI
import tkinter as tk
from tkinter import scrolledtext, messagebox

# nltk libraries 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

# spacy libraries
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

# sumy libraries 
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# web scrapping libraries
from bs4 import BeautifulSoup
from urllib.request import urlopen

# web scrapping functions
def get_text(raw_url):
    try:
        page = urlopen(raw_url)
        soup = BeautifulSoup(page, 'html.parser')
        fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
        return fetched_text
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch URL: {e}")
        return ""

# NLTK summarizer functions
def read_article(text):
    sentences = sent_tokenize(text)
    sentences = [sentence.replace("[^a-zA-Z]", " ") for sentence in sentences]
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = word_tokenize(sent1)
    sent2 = word_tokenize(sent2)
    
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for w in sent1:
        if w not in stopwords:
            vector1[all_words.index(w)] += 1
    
    for w in sent2:
        if w not in stopwords:
            vector2[all_words.index(w)] += 1
            
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
                
    return similarity_matrix

def nltk_summarizer(text):
    stop_words = stopwords.words('english')
    summarize_text = []
    
    sentences = read_article(text)
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    sentence_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_graph)
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)
    top_n = 2
    for i in range(min(top_n, len(ranked_sentences))):
        summarize_text.append(ranked_sentences[i][1])
    
    return " ".join(summarize_text)

# SpaCy summarizer functions
def spacy_summarizer(raw_docx):
    docx = nlp(raw_docx)
    stopwords = list(STOP_WORDS)
    word_frequencies = {}
    for word in docx:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

    maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / maximum_frequency

    sentence_list = [sentence for sentence in docx.sents]
    sentence_scores = {}
    for sentence in sentence_list:
        for word in sentence:
            if word.text.lower() in word_frequencies.keys():
                if len(sentence.text.split(' ')) < 30:
                    if sentence not in sentence_scores.keys():
                        sentence_scores[sentence] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sentence] += word_frequencies[word.text.lower()]
    
    summarized_sentences = nlargest(2, sentence_scores, key=sentence_scores.get)
    final_sentences = [w.text for w in summarized_sentences]
    summary = ' '.join(final_sentences)
    return summary

# Sumy summarizer functions
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 2)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

# counting input and summarized sentences
def count_sentences(input_text, summarized_text):
    input_sentence_count = len(sent_tokenize(input_text))
    summarized_sentence_count = len(sent_tokenize(summarized_text))
    return input_sentence_count, summarized_sentence_count

def count_words(text):
    return len(text.split())

def summarize_text():
    text = text_area.get("1.0", "end-1c").strip()
    if not text:
        messagebox.showwarning("Input Error", "Please enter some text to summarize.")
        return
     
    input_words = count_words(text)

    summarizer_choice = summary_choice.get()

    if summarizer_choice == 'NLTK':
        summary_result = nltk_summarizer(text)
    elif summarizer_choice == 'SpaCy':
        summary_result = spacy_summarizer(text)
    elif summarizer_choice == 'Sumy Lex rank':
        summary_result = sumy_summarizer(text)
    
    summarized_words = count_words(summary_result)
    input_sentences, summarized_sentences = count_sentences(text, summary_result)

    stats_label.config(
        text=f"Input Sentences: {input_sentences} | Summarized Sentences: {summarized_sentences} | "
             f"Input Words: {input_words} | Summarized Words: {summarized_words}"
    )

    result_area.config(state=tk.NORMAL)
    result_area.delete(1.0, tk.END)
    result_area.insert(tk.END, summary_result)
    result_area.config(state=tk.DISABLED)

# handle URL summarization
def summarize_url():
    url = url_entry.get().strip()
    if not url:
        messagebox.showwarning("Input Error", "Please enter a URL.")
        return

    content = get_text(url)
    input_words = count_words(content)
    if content:
        summarizer_choice = summary_choice.get()
        if summarizer_choice == 'NLTK':
            summary_result = nltk_summarizer(content)
        elif summarizer_choice == 'SpaCy':
            summary_result = spacy_summarizer(content)
        elif summarizer_choice == 'Sumy Lex rank':
            summary_result = sumy_summarizer(content)

        summarized_words = count_words(summary_result)
        input_sentences, summarized_sentences = count_sentences(content, summary_result)

        stats_label.config(
            text=f"Input Sentences: {input_sentences} | Summarized Sentences: {summarized_sentences} | "
                 f"Input Words: {input_words} | Summarized Words: {summarized_words}"
        )

        result_area.config(state=tk.NORMAL)
        result_area.delete(1.0, tk.END)
        result_area.insert(tk.END, summary_result)
        result_area.config(state=tk.DISABLED)

def reset_inputs():
    text_area.delete(1.0, tk.END)
    url_entry.delete(0, tk.END)
    stats_label.config(text="Input Sentences: 0 | Summarized Sentences: 0 | Input Words: 0 | Summarized Words: 0")

def clear_results():
    result_area.config(state=tk.NORMAL)
    result_area.delete(1.0, tk.END)
    result_area.config(state=tk.DISABLED)

# Creating the GUI using Tkinter
root = tk.Tk()
root.title("Extractive Text Summarizer")
root.geometry("700x400")

# Created frames
frame = tk.Frame(root)
frame.pack(pady=20)

# text area for user input
text_area_label = tk.Label(frame, text="Enter Text Here:")
text_area_label.grid(row=0, column=0, padx=5, pady=5)

text_area = scrolledtext.ScrolledText(frame, width=60, height=10)
text_area.grid(row=1, column=0, padx=5, pady=5)

# buttons to summarize
summarize_button = tk.Button(frame, text="Summarize Text", command=summarize_text, bg="skyblue", fg="black")
summarize_button.grid(row=2, column=0, padx=5, pady=10)

# dopdown menu for summarization choice
summary_choice_label = tk.Label(frame, text="Choose Summarization Method:")
summary_choice_label.grid(row=3, column=0, padx=5, pady=5)

summary_choice = tk.StringVar()
summary_choice.set("NLTK")

summary_choice_menu = tk.OptionMenu(frame, summary_choice, "NLTK", "SpaCy", "Sumy Lex rank")
summary_choice_menu.config(bg="lightgreen", fg="black", activebackground="blue", activeforeground="white")
summary_choice_menu.grid(row=4, column=0, padx=5, pady=5)

# URL input section
url_label = tk.Label(frame, text="Enter URL for Summarization:")
url_label.grid(row=5, column=0, padx=5, pady=5)

url_entry = tk.Entry(frame, width=50)
url_entry.grid(row=6, column=0, padx=5, pady=5)

url_button = tk.Button(frame, text="Summarize URL", command=summarize_url,bg="skyblue", fg="black")
url_button.grid(row=7, column=0, padx=5, pady=10)

# area for displaying results
result_area_label = tk.Label(frame, text="Summarized Text:")
result_area_label.grid(row=8, column=0, padx=5, pady=5)

result_area = scrolledtext.ScrolledText(frame, width=60, height=10, wrap=tk.WORD, state=tk.DISABLED)
result_area.grid(row=9, column=0, padx=5, pady=5)

stats_label = tk.Label(frame, text="Input Sentences: 0 | Summarized Sentences: 0 | Input Words: 0 | Summarized Words: 0")
stats_label.grid(row=10, column=0, padx=5, pady=5)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

reset_button = tk.Button(button_frame, text="Reset Inputs", command=reset_inputs, bg="green", fg="white")
reset_button.grid(row=0, column=0, padx=10, pady=5)

clear_result_button = tk.Button(button_frame, text="Clear Results", command=clear_results, bg="blue", fg="white")
clear_result_button.grid(row=0, column=1, padx=10, pady=5)

root.mainloop()