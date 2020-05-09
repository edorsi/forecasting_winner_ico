import datetime
import os
import re
import time
import warnings
from collections import Counter, OrderedDict
from itertools import cycle
from itertools import groupby
from os import listdir
from os.path import isfile, join

import PyPDF2
import gensim
import gensim.corpora as corpora
from gensim.models import FastText
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from PIL import Image
from gensim.models import Phrases, CoherenceModel
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from pdf2image import convert_from_path
from scipy.spatial import distance
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE

import pytesseract


#no need
# from statsmodels.formula.api import ols
# import statsmodels.api as sm
# from tabulate import tabulate
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
# from sklearn.naive_bayes import BernoulliNB, MultinomialNB
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
# from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
# from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", 999)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
chromedriver_path = "C:/chromedriver_win32/chromedriver.exe"
ner_path = r"C:\stanford-ner"


def download_whitepaper(icodrops_download_path, icodrops_ended_details_path, verbose=False):
    def every_downloads_chrome(driver):
        if not driver.current_url.startswith("chrome://downloads"):
            driver.get("chrome://downloads/")
        return driver.execute_script("""
            var items = downloads.Manager.get().items_;
            if (items.every(e => e.state === "COMPLETE"))
                return items.map(e => e.fileUrl || e.file_url);
            """)

    df = pd.read_csv(icodrops_ended_details_path, index_col=0)
    print(df.info()) if verbose else False
    print(df.head(5)) if verbose else False

    df_errors = pd.DataFrame()
    icos = len(df['ico_name'])
    for ico_num, ico in enumerate(df['ico_name']):
        path = 'not_available'
        directory = icodrops_download_path + ico.replace(":", "_").replace(u'\xa0', u'')
        print(datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - ICO pdf to be downlaoded into: %s' % directory) if verbose else False
        if not os.path.exists(directory):
            os.makedirs(directory)
        options = webdriver.ChromeOptions()
        prefs = {"plugins.plugins_disabled": ["Chrome PDF Viewer"],
                 "plugins.always_open_pdf_externally": True,
                 "download.default_directory": directory,
                 "download.extensions_to_open": "applications/pdf"}
        options.add_experimental_option("prefs", prefs)
        driver = webdriver.Chrome(chromedriver_path, options=options)
        onlyfiles = len([f for f in listdir(directory) if isfile(join(directory, f))])
        if not onlyfiles:
            try:
                pdf_url = df.loc[ico_num, 'whitepaper_link']
                if pdf_url:
                    print(pdf_url) if verbose else False
                    driver.get(pdf_url)
                    driver.minimize_window()
                    time.sleep(15)
                    path = WebDriverWait(driver, 15, 1).until(every_downloads_chrome)
            except:
                path = 'not_available'
                data = {'ico_name': df.loc[ico_num, 'ico_name'],
                        'whitepaper_link': df.loc[ico_num, 'whitepaper_link']}
                df_errors = df_errors.append(pd.DataFrame(data, index=[0]), ignore_index=True)
                pass
        print('\r' + datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - Completed ICO num %d - %3.2f%% - in %s' % (ico_num + 1, (ico_num + 1) / icos * 100, path), end='')
        driver.close()

    df_errors.to_csv(icodrops_download_path + 'errors.csv', index_label='index')
    print('\n' + datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - Collected ICOs white paper')


def check_whitepaper_download(icodrops_ended_details_path, icodrops_download_path, icodrops_ended_checked_path, verbose=False):
    df = pd.read_csv(icodrops_ended_details_path, index_col=0)
    print(df.info()) if verbose else False
    print(df.head(5)) if verbose else False

    max_ico = len(df['ico_name'])
    df_check = pd.DataFrame()
    for ico_num, ico in enumerate(df['ico_name']):
        directory = icodrops_download_path + ico.replace(":","_").replace(u'\xa0', u'')
        onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
        len_onlyfiles = len(onlyfiles)
        if len_onlyfiles:
            data = {'wp_downlaoded': 'Y', 'wp_dir': directory + '\\'}
        else:
            data = {'wp_downlaoded': 'N', 'wp_dir': 'not available'}
        df_check = df_check.append(pd.DataFrame(data, index=[0]), ignore_index=True)
        print('\r' + datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - Completed ICO num %d - %3.2f%%' % (ico_num + 1, (ico_num + 1) / max_ico * 100), end='')
    df = pd.concat([df, df_check], axis='columns', join='outer', ignore_index=False, sort=False)
    df.to_csv(icodrops_ended_checked_path, index_label='index')


def convert_whitepaper(icodrops_ended_checked_path, icodrops_download_path, icodrops_whitepaper_converted_path, verbose=False):
    df = pd.read_csv(icodrops_ended_checked_path, index_col=0)
    print(df.info()) if verbose else False
    print(df.head(5)) if verbose else False

    max_ico = len(df['ico_name'])
    df_check = pd.DataFrame()
    for ico_num, ico in enumerate(df['ico_name']):
        dir = icodrops_download_path + ico.replace(":", "_").replace(u'\xa0', u'')
        onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
        print(onlyfiles) if verbose else False
        onlyfilesext = [f.split(".")[-1] for f in listdir(dir) if isfile(join(dir, f))]
        print(onlyfilesext) if verbose else False
        directory = dir + '\\'
        data = {'whitepaper_name': 'not_available'}

        if ('pdf' in onlyfilesext) and ('txt' not in onlyfilesext):
            print('Check 1 True') if verbose else False
            for file in onlyfiles:
                if (file.split(".")[-1] == 'jpg' or file.split(".")[-1] == 'jpeg'):
                    os.remove(os.path.join(directory, file))

            for file in onlyfiles:
                if (file.split(".")[-1] == 'pdf'):
                    try:
                        file_name = str(directory) + str(file.split(".")[0]) + '.txt'
                        print(file_name) if verbose else False
                        try:
                            pdfFileReader = PyPDF2.PdfFileReader(directory + file)
                            maxPages = pdfFileReader.numPages + 1
                        except:
                            maxPages = 200
                        print('Max pages: %d' % maxPages) if verbose else False
                        for page in range(1, maxPages, 10):
                            images_from_path = convert_from_path(directory + file, output_folder=directory, output_file='page', fmt='.jpg', first_page=page, last_page=min(page + 10 - 1, maxPages))
                            print('\r' + datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - ICO %s num %d - %3.2f%% - Images processing - page %d / %d' % (ico, ico_num + 1, (ico_num + 1) / max_ico * 100, page, maxPages), end='')
                        f = open(file_name, "a", encoding='utf-8')
                        for page_no in range(1, maxPages):
                            try:
                                filename = directory + 'page0001-' + str(page_no).zfill(1) + '.jpg'
                                text = str((pytesseract.image_to_string(Image.open(filename))))
                                f.write(text)
                            except:
                                pass
                            try:
                                filename = directory + 'page0001-' + str(page_no).zfill(2) + '.jpg'
                                text = str((pytesseract.image_to_string(Image.open(filename))))
                                f.write(text)
                            except:
                                pass
                            try:
                                filename = directory + 'page0001-' + str(page_no).zfill(3) + '.jpg'
                                text = str((pytesseract.image_to_string(Image.open(filename))))
                                f.write(text)
                            except:
                                pass
                            print('\r' + datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - ICO %s num %d - %3.2f%% - Text processing - page %d / %d' % (ico, ico_num + 1, (ico_num + 1) / max_ico * 100, page_no, maxPages), end='')
                        f.close()
                        data = {'whitepaper_name': file_name}
                    except:
                        data = {'whitepaper_name': 'to_review'}
                        pass
        elif 'txt' in onlyfilesext:
            for file in onlyfiles:
                if (file.split(".")[-1] == 'txt'):
                    file_name = str(directory) + str(file)
                    data = {'whitepaper_name': file_name}
                    print('\r' + datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - ICO %s num %d - %3.2f%% - WHite paper folder - %s' % (ico, ico_num + 1, (ico_num + 1) / max_ico * 100, file_name), end='')
                if (file.split(".")[-1] == 'jpg' or file.split(".")[-1] == 'jpeg'):
                    os.remove(os.path.join(directory, file))
        else:
            data = {'whitepaper_name': 'not_available'}
            print('\r' + datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - ICO %s num %d - %3.2f%% - White paper not available' % (ico, ico_num + 1, (ico_num + 1) / max_ico * 100), end='')

        df_check = df_check.append(pd.DataFrame(data, index=[0]), ignore_index=True)
    df = pd.concat([df, df_check], axis='columns', join='outer', ignore_index=False, sort=False)
    df.to_csv(icodrops_whitepaper_converted_path, index_label='index')


def topic_modeling(icodrops_whitepaper_converted_path, icodrops_topic_path, tech_words, verbose=False):
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None

    def lda_models(corpus, id2word, num_topics):
        print(datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - LDA model %d - # topics' %num_topics) if verbose else False
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                eval_every=None,
                                                per_word_topics=True)
        print(datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - LDA model %d - model created' % num_topics) if verbose else False
        coherence_model = CoherenceModel(model=model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        print(datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - LDA model %d - cohorence calculated' % num_topics) if verbose else False
        coherence_values.append(round(coherence_model.get_coherence(), 4))
        print(datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - LDA model %d - cohorence appended' % num_topics) if verbose else False
        model_list.append((num_topics, round(coherence_model.get_coherence(), 4), model))
        print(datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - LDA model %d - model appended' % num_topics) if verbose else False

    stop_words = stopwords.words('english')

    df = pd.read_csv(icodrops_whitepaper_converted_path, index_col=0)
    print(df.info()) if verbose else False
    print(df.head(5)) if verbose else False

    max_ico = len(df['whitepaper_name'])
    df_topics = pd.DataFrame()
    for wp_num, wp in enumerate(df['whitepaper_name']):
        ico = df['ico_name'][wp_num]
        print('\r' + datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - ICO %s num %d - %3.2f%% - Doc analysis' % (ico, wp_num + 1, (wp_num + 1) / max_ico * 100), end='')
        if wp != 'not_available':
            try:
                try:
                    document = open(wp, 'r', encoding="utf-8")
                    document = document.read()
                except:
                    document = open(wp, 'r', encoding="ISO-8859-1")
                    document = document.read()
                document = re.sub('\s+', ' ', document)
                document = re.sub('\S*@\S*\s?', '', document)
                document = re.sub("\'", "", document)
                sentences = nltk.sent_tokenize(document)
                sentences = [sentence.strip() for sentence in sentences]
                print(datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - Total sentences in text:', len(sentences)) if verbose else False

                data_words =  [gensim.utils.simple_preprocess(str(sentence), deacc=True) for sentence in sentences]
                data_words = list(filter(None, data_words))
                data_words = [token for token in data_words if token not in stop_words]
                print(datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - Data words:') if verbose else False

                bigram = Phrases(data_words, min_count=5, threshold=100)
                trigram = Phrases(bigram[data_words], threshold=100)
                bigram_mod = gensim.models.phrases.Phraser(bigram)
                trigram_mod = gensim.models.phrases.Phraser(trigram)
                text = [trigram_mod[bigram_mod[doc]] for doc in data_words]
                print(datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - Biagrams and Triagrams:') if verbose else False

                lemmatized_0 = [nltk.pos_tag(token) for token in text]
                lemmatized_1 = [[t for t in token if t[1].startswith('N') or t[1].startswith('J') or t[1].startswith('V') or t[1].startswith('R')] for token in lemmatized_0]
                lemmatized_1 = [[(t[0].lower(), get_wordnet_pos(t[1])) for t in token] for token in lemmatized_1]
                wnl = WordNetLemmatizer()
                data_lemmatized = [[wnl.lemmatize(word, pos_tag) for word, pos_tag in token] for token in lemmatized_1]
                print(datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - Data lemmatized:') if verbose else False

                id2word = corpora.Dictionary(data_lemmatized)
                texts = data_lemmatized
                corpus = [id2word.doc2bow(text) for text in texts]
                print(datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - Corpus:') if verbose else False

                start = 20
                limit = 41
                step = 2
                model_list = []
                coherence_values = []
                topic_range = list(range(start, limit, step))

                # for num_topics in range(start, limit, step):
                #     lda_models(corpus, id2word, num_topics)
                lda_models(corpus, id2word, 10)

                index = coherence_values.index(max(coherence_values))
                print(datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - Choosen model and statistics:') if verbose else False
                print(model_list) if verbose else False
                print(coherence_values) if verbose else False
                print(max(coherence_values)) if verbose else False
                print(index) if verbose else False

                x = model_list[index][2].show_topics(num_topics=model_list[index][0], num_words=10, formatted=False)
                topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]
                print(datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - Topics:') if verbose else False
                print(topics_words) if verbose else False

                sentence_topic = []
                for d in sentences:
                    bow = id2word.doc2bow(d.split())
                    t = model_list[index][2].get_document_topics(bow)
                    topics_woight = [tw[1] for tw in t]
                    # sentence_topic_weight.append((topics_woight.index(max(topics_woight)), max(topics_woight)))
                    sentence_topic.append(topics_woight.index(max(topics_woight)))
                print(sentence_topic) if verbose else False
                print(len(sentence_topic)) if verbose else False
                sentences_per_topic = Counter(sentence_topic)
                sentences_per_topic = OrderedDict(sorted(sentences_per_topic.items()))
                print(sentences_per_topic) if verbose else False
                print('\r' + datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - ICO %s num %d - %3.2f%% - Topic modelling completed' % (ico, wp_num + 1, (wp_num + 1) / max_ico * 100), end='')
                data = {'sentences': len(sentences), 'num_topics': model_list[index][0], 'coherence_values': model_list[index][1]}
                for i, w in enumerate(topics_words):
                    topic_words = ', '.join(topics_words[i][1])
                    data.update({'topic_words_' + str(i).zfill(3) : topic_words})
                    topic_tech = 1 if len(list(set(tech_words).intersection(topics_words[i][1]))) > 0 else 0
                    print(len(list(set(tech_words).intersection(topics_words[i][1])))) if verbose else False
                    data.update({'topic_tech_' + str(i).zfill(3): str(topic_tech)})
                for key, value in sentences_per_topic.items():
                    data.update({'sentences_in_topic_' + str(key).zfill(3) : str(value)})
                result = denom = 0
                for i, w in enumerate(topics_words):
                    try:
                        result = result + int(data['topic_tech_' + str(i).zfill(3)]) * int(data['sentences_in_topic_' + str(i).zfill(3)])
                    except:
                        pass
                    try:
                        denom = denom + int(data['sentences_in_topic_' + str(i).zfill(3)])
                    except:
                        pass
                result_pct = round(result / denom, 4)
                data.update({'tech_sentences' : str(result)})
                data.update({'tech_sen_pct' : str(result_pct)})
            except:
                data = {'sentences': 'error', 'num_topics': 'error','coherence_values': 'error'}
        else:
            data = {'sentences': 'not available', 'num_topics': 'not available', 'coherence_values': 'not available'}
        df_topics = df_topics.append(pd.DataFrame(data, index=[0]), ignore_index=True, sort=True)

    df = pd.concat([df, df_topics], axis='columns', join='outer', ignore_index=False, sort=False)
    df.to_csv(icodrops_topic_path, index_label='index')


def ner(icodrops_topic_path, icodrops_ner_path, remove_ner, remove_ner_person, remove_ner_organization, remove_ner_location, verbose=False):
    df = pd.read_csv(icodrops_topic_path, index_col=0)
    print(df.info()) if verbose else False
    print(df.head(5)) if verbose else False

    max_ico = len(df['whitepaper_name'])
    df_ner = pd.DataFrame()
    for wp_num, wp in enumerate(df['whitepaper_name']):
        ico = df['ico_name'][wp_num]
        print(wp) if verbose else False
        print(ico) if verbose else False
        try:
            try:
                document = open(wp, 'r', encoding="utf-8")
                document = document.read()
            except:
                document = open(wp, 'r', encoding="ISO-8859-1")
                document = document.read()
            document = re.sub(ico, ' ', document)
            document = re.sub('\s+', ' ', document)
            document = re.sub('\S*@\S*\s?', '', document)
            document = re.sub("\'", "", document)
            for w in remove_ner:
                document = re.sub(w, '', document)

            ner_person = []
            ner_organization = []
            ner_location = []
            st = StanfordNERTagger(ner_path + '/classifiers/english.all.3class.distsim.crf.ser.gz',
                                   ner_path + '/stanford-ner.jar',
                                   encoding='utf-8')
            tokenized_text = word_tokenize(document)
            classified_text = st.tag(tokenized_text)
            print(classified_text) if verbose else False
            for tag, chunk in groupby(classified_text, lambda x: x[1]):
                words = " ".join(w for w, t in chunk)
                if tag == 'PERSON':
                    ner_person.append(words)
                elif tag == 'ORGANIZATION':
                    ner_organization.append(words)
                elif tag == 'LOCATION':
                    ner_location.append(words)

            ner_person = list(set(ner_person)-set(remove_ner_person))
            ner_organization = list(set(ner_organization) - set(remove_ner_organization))
            ner_location = list(set(ner_location) - set(remove_ner_location))
            ner_person = list(dict.fromkeys(sorted(ner_person)))
            ner_organization = list(dict.fromkeys(sorted(ner_organization)))
            ner_location = list(dict.fromkeys(sorted(ner_location)))

            ner_w_person = ', '.join(ner_person)
            ner_w_organization = ', '.join(ner_organization)
            ner_w_location = ', '.join(ner_location) if ner_location not in remove_ner_location else False
            print('Person:') if verbose else False
            print(ner_w_person) if verbose else False
            print('Location:') if verbose else False
            print(ner_w_location) if verbose else False
            print('Organization:') if verbose else False
            print(ner_w_organization) if verbose else False
            print('\r' + datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - ICO %s num %d - %3.2f%% - NER completed' % (ico, wp_num + 1, (wp_num + 1) / max_ico * 100), end='')

            data = {'person': ner_w_person, 'organization': ner_w_organization, 'location': ner_w_location,
                    'person_num': len(ner_person), 'organization_num': len(ner_organization), 'location_num': len(ner_location)}
            print('\n') if verbose else False
            print(data) if verbose else False
        except:
            data = {'person': 'not available', 'organization': 'not available', 'location': 'not available'}
            print('\r' + datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - ICO %s num %d - %3.2f%% - NER completed' % (ico, wp_num + 1, (wp_num + 1) / max_ico * 100), end='')
        df_ner = df_ner.append(pd.DataFrame(data, index=[0]), ignore_index=True, sort=True)
    df = pd.concat([df, df_ner], axis='columns', join='outer', ignore_index=False, sort=False)
    df.to_csv(icodrops_ner_path, index_label='index')


def ico_date_process(input_csv, output_csv, verbose=False):
    """
    Review the downloaded range date period of the ICO and split it into multiple columns day, month, year, quarter,
    duration
    :param input_csv:
    :param output_csv:
    :param verbose:
    :return: the initial data frame csv with additional columns
    """

    def time_x(element):
        try:
            return time.strptime(element[-3:], '%b').tm_mon
        except:
            pass

    def day_x(element):
        x = re.findall(r'\d+', element)
        if x:
            return int(x[0])
        else:
            return 0

    def change_year(quarter_end, quarter_shift):
        if quarter_end > quarter_shift:
            return 1
        else:
            return 0

    def year_start(year_start, year_end):
        try:
            if int(year_start) >= 0:
                return int(year_end)
            else:
                return int(year_end) - 1
        except:
            pass

    def quarter(element):
        if element < 4:
            q = 1
        elif element < 7:
            q = 2
        elif element < 10:
            q = 3
        else:
            q = 4
        return q

    df = pd.read_csv(input_csv, index_col=0)
    print(df.info()) if verbose else False
    print(df.head(5)) if verbose else False

    df['full'] = df['token'].apply(lambda element: re.sub(r'\u2011', '-', element))
    df['full'] = df['full'].apply(lambda element: re.sub(r'\u2012', '-', element))
    df['full'] = df['full'].apply(lambda element: re.sub(r'\u2013', '-', element))
    df['full'] = df['full'].apply(lambda element: re.sub(r'\u2014', '-', element))
    df['ico_start'] = df['full'].apply(lambda element: element.split('-')[0])
    df['ico_end'] = df['full'].apply(lambda element: element.split('-')[-1])
    df['day_start'] = df['ico_start'].apply(lambda element: day_x(element))
    df['day_end'] = df['ico_end'].apply(lambda element: day_x(element))
    df['month_start'] = df['ico_start'].apply(lambda element: time_x(element))
    df['month_end'] = df['ico_end'].apply(lambda element: time_x(element))
    df['quarter_start'] = df['month_start'].apply(lambda element: quarter(element))
    df['quarter_end'] = df['month_end'].apply(lambda element: quarter(element))
    df['quarter_shift'] = df['quarter_end'].shift()
    df['change_year'] = df.apply(
        lambda element: change_year(quarter_end=element['quarter_end'], quarter_shift=element['quarter_shift']), axis=1)
    df['change_year_cum'] = df['change_year'].cumsum()
    df['year_end'] = df['sale_date'].apply(lambda element: str(element)[-4:])
    df['year_start'] = df['month_end'] - df['month_start']
    df['year_start'] = df.apply(lambda element: year_start(element['year_start'], element['year_end']), axis=1)
    df['year_start'] = np.where(df.month_end.isnull(), np.nan, df['year_start'])
    df['year_end'] = np.where(df.month_end.isnull(), np.nan, df['year_end'])
    df['month_start'] = df['month_start'].fillna(0).astype(np.int64)
    df['month_end'] = df['month_end'].fillna(0).astype(np.int64)
    df['quarter_start'] = df['quarter_start'].fillna(0).astype(np.int64)
    df['quarter_end'] = df['quarter_end'].fillna(0).astype(np.int64)
    df['year_start'] = df['year_start'].fillna(0).astype(np.int64)
    df['year_end'] = df['year_end'].fillna(0).astype(np.int64)
    df['date_start'] = df['day_start'].map(str) + '/' + df['month_start'].map(str) + '/' + df['year_start'].map(str)
    df['date_end'] = df['day_end'].map(str) + '/' + df['month_end'].map(str) + '/' + df['year_end'].map(str)
    df['date_start'] = pd.to_datetime(df['date_start'], errors='coerce', format='%d/%m/%Y')
    df['date_end'] = pd.to_datetime(df['date_end'], errors='coerce', format='%d/%m/%Y')
    df['duration_days'] = (df['date_end'] - df['date_start']) / np.timedelta64(1, 'D')
    df['duration_days'] = df['duration_days'].fillna(0).astype(np.int64)

    df.to_csv(output_csv, index_label='index')
    print(df.info()) if verbose else False
    print(df) if verbose else False


def load_whitepapers(df_file_path, wp_analysis_path, stop_words=nltk.corpus.stopwords.words('english'), verbose=False):
    def normalize_document(doc, stop_words):
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I | re.A)
        doc = doc.lower()
        doc = doc.strip()
        # tokenize document
        tokens = nltk.word_tokenize(doc)
        # filter stopwords out of document
        filtered_tokens = [token for token in tokens if token not in stop_words]
        # re-create document from filtered tokens
        doc = ' '.join(filtered_tokens)
        # print(len(doc))
        return doc

    df = pd.read_csv(df_file_path, index_col=0)
    # df = df[500:503]
    print(df.info()) if verbose else False
    print(df.head(5)) if verbose else False
    icos = len(df['whitepaper_name'])
    # df_temp = pd.DataFrame(df.to_numpy(), columns=list(df.columns.values))

    df_whitepaper = pd.DataFrame()
    documents = [doc for doc in df['whitepaper_name']]
    for ico_num, doc in enumerate(documents):
        if doc != 'not_available':
            f = open(doc, "r", encoding="utf8", errors="ignore")
            norm_doc = normalize_document(f.read(), stop_words)
            print("Read %d %s: len %d" % (ico_num, doc, len(norm_doc))) if verbose else False
            data_dic = {'whitepaper_original': f.read(), 'whitepaper_for_cluster': norm_doc}
            print(norm_doc) if verbose else False
            f.close()
        else:
            data_dic = {'whitepaper_original': 'not_available', 'whitepaper_for_cluster': 'not_available'}
            print("Read %d %s: len 0" % (ico_num, doc)) if verbose else False
        df_whitepaper = df_whitepaper.append(pd.DataFrame(data_dic, index=[0]), ignore_index=True, sort=False)
        print('\r' + datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - Completed ICO num %d/%d - %3.2f%%' % (ico_num + 1, icos, (ico_num + 1) / icos * 100), end='')

    df = pd.concat([df, df_whitepaper], axis='columns', join='outer', ignore_index=False, sort=False)
    print(df.head(-5)) if verbose else False
    df.to_csv(wp_analysis_path, index_label='index')


def aff_prop(df_file_path, af_file_path, w2v_not_processed, do_plot=False, verbose=False):
    def averaged_word_vectorizer(corpus, model, num_features):
        def average_word_vectors(words, model, num_features, verbose=False):
            feature_vector = np.zeros((num_features,), dtype="float64")
            nwords = 0
            not_processed = []
            n_words = len(words)
            for w_num, word in enumerate(words):
                try:
                    print(word) if verbose else False
                    nwords = nwords + 1.
                    feature_vector = np.add(feature_vector, model.wv[word])
                except:
                    not_processed.append(word)
                    pass
            if nwords:
                feature_vector = np.divide(feature_vector, nwords)
            with open(w2v_not_processed, 'a+', encoding="utf-8") as f:
                for item in not_processed:
                    f.write("%s\n" % item)
            return feature_vector

        # vocabulary = set(model.wv.index2word)
        try:
            os.remove(w2v_not_processed)
        except:
            pass
        features = [average_word_vectors(tokenized_sentence, model, num_features) for tokenized_sentence in corpus]
        return np.array(features)

    df = pd.read_csv(df_file_path, index_col=0)
    print(df.info()) if verbose else False
    print(df.head(5)) if verbose else False
    print(datetime.datetime.utcnow().strftime("%H:%M:%S") + ' - Dataframe loaded')

    norm_corpus = df['whitepaper_for_cluster'].tolist()
    wpt = nltk.WordPunctTokenizer()
    tokenized_corpus = [wpt.tokenize(str(document)) for document in norm_corpus]
    print(tokenized_corpus) if verbose else False
    print(datetime.datetime.utcnow().strftime("%H:%M:%S") + ' - Tokenized corpus: ' + str(len(tokenized_corpus)) + ' documents')

    # Set values for various parameters
    feature_size = 5  # Word vector dimensionality
    window_context = 10  # Context window size
    min_word_count = 1  # Minimum word count
    sample = 1e-3  # Downsample setting for frequent words
    seed = 42
    iter = 100
    sentences = tokenized_corpus
    w2v_model = FastText(sentences=sentences, seed=seed, size=feature_size, window=window_context, min_count=min_word_count, sample=sample, iter=iter)
    print(datetime.datetime.utcnow().strftime("%H:%M:%S") + ' - W2V model completed')

    w2v_feature_array = averaged_word_vectorizer(corpus=tokenized_corpus, model=w2v_model, num_features=feature_size)
    w2v_df = pd.DataFrame(w2v_feature_array)
    w2v_df = w2v_df.add_prefix('w2v_avg_')
    df = pd.concat([df, w2v_df], axis='columns', join='outer', ignore_index=False, sort=False)
    df.to_csv(af_file_path)
    print(datetime.datetime.utcnow().strftime("%H:%M:%S") + ' - W2V average model completed')

    af = AffinityPropagation(affinity='euclidean', damping=0.5, max_iter=1000, convergence_iter=30).fit(w2v_feature_array)
    df['af_cluster_labels'] = af.labels_
    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)
    for i in range(len(df)):
        df.loc[i, 'af_cluster_centers_indices'] = cluster_centers_indices[df.loc[i, 'af_cluster_labels']]
    print(datetime.datetime.utcnow().strftime("%H:%M:%S") + ' - Affinity model completed')

    tsne_model_en_2d = TSNE(perplexity=len(cluster_centers_indices), n_components=2, init='pca', n_iter=3500, random_state=42)
    tsne = tsne_model_en_2d.fit_transform(w2v_feature_array)
    tsne_df = pd.DataFrame(tsne)
    tsne_df = tsne_df.add_prefix('tsne_')
    df = pd.concat([df, tsne_df], axis='columns', join='outer', ignore_index=False, sort=False)
    print(datetime.datetime.utcnow().strftime("%H:%M:%S") + ' - t-SNE model completed')

    for i in range(len(df)):
        centre_0 = df.loc[df.loc[i, 'af_cluster_centers_indices'], 'tsne_0']
        centre_1 = df.loc[df.loc[i, 'af_cluster_centers_indices'], 'tsne_1']
        node_0 = df.loc[i, 'tsne_0']
        node_1 = df.loc[i, 'tsne_1']
        centre = (centre_0, centre_1)
        node = (node_0, node_1)
        df.loc[i, 'centre_cluster_euclidean_distance'] = distance.euclidean(centre, node)
    print(datetime.datetime.utcnow().strftime("%H:%M:%S") + ' - Euclidean distance completed')

    df.to_csv(af_file_path)
    print(datetime.datetime.utcnow().strftime("%H:%M:%S") + ' - Affinity analysis completed')

    if do_plot:
        plt.close('all')
        plt.figure(1)
        plt.clf()
        X = tsne
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            class_members = af.labels_ == k
            cluster_center = X[cluster_centers_indices[k]]
            plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, label='Cluster #' + str(k),
                     markeredgecolor='k', markersize=14)
            plt.text(cluster_center[0], cluster_center[1], 'Cluster #' + str(k), fontsize=9)
            for x in X[class_members]:
                plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        l = plt.legend(loc='upper right')
        l.set_zorder(20)  # put the legend on top
        plt.show()


def sentiment(df_file_path, df_sentiment, verbose=False):
    def tokenize_text(text):
        tokens = nltk.word_tokenize(text)
        tokens = [token.strip() for token in tokens]
        return tokens

    def pos_tag_text(text, verbose=False):
        def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return wn.ADJ
            elif treebank_tag.startswith('V'):
                return wn.VERB
            elif treebank_tag.startswith('N'):
                return wn.NOUN
            elif treebank_tag.startswith('R'):
                return wn.ADV
            else:
                return None

        tokens = tokenize_text(text)
        tagged_text = nltk.pos_tag(tokens)
        tagged_lower_text = [[word.lower(), pos_tag]
                             for word, pos_tag in
                             tagged_text]
        print('Total words in tagged text:', len(tagged_lower_text)) if verbose == True else False
        return tagged_lower_text

    def analyze_sentiment_sentiwordnet_lexicon(text, verbose=False):
        tokens = tokenize_text(text)
        print(tokens) if verbose else False
        tagged_text = pos_tag_text(text)
        print(tagged_text) if verbose else False
        pos_score = neg_score = token_count = obj_score = 0
        for word, tag in tagged_text:
            ss_set = None
            if 'NN' in tag and list(swn.senti_synsets(word, 'n')):
                ss_set = list(swn.senti_synsets(word, 'n'))[0]
            elif 'VB' in tag and list(swn.senti_synsets(word, 'v')):
                ss_set = list(swn.senti_synsets(word, 'v'))[0]
            elif 'JJ' in tag and list(swn.senti_synsets(word, 'a')):
                ss_set = list(swn.senti_synsets(word, 'a'))[0]
            elif 'RB' in tag and list(swn.senti_synsets(word, 'r')):
                ss_set = list(swn.senti_synsets(word, 'r'))[0]
            if ss_set:
                # add scores for all found synsets
                pos_score += ss_set.pos_score()
                neg_score += ss_set.neg_score()
                obj_score += ss_set.obj_score()
                token_count += 1

        # aggregate final scores
        final_score = pos_score - neg_score
        if token_count > 0:
            norm_final_score = round(float(final_score) / token_count, 2)
            norm_obj_score = round(float(obj_score) / token_count, 2)
            norm_pos_score = round(float(pos_score) / token_count, 2)
            norm_neg_score = round(float(neg_score) / token_count, 2)
        else:
            norm_obj_score = norm_pos_score = norm_neg_score = norm_final_score = 0
        final_sentiment = 'positive' if norm_final_score >= 0 else 'negative'

        return token_count, norm_obj_score, norm_pos_score, norm_neg_score, norm_final_score, final_sentiment

    df = pd.read_csv(df_file_path, index_col=0)
    print(df.info()) if verbose else False
    print(df.head(5)) if verbose else False

    sentiment_frame = pd.DataFrame()
    documents = [doc for doc in df['whitepaper_for_cluster']]
    for doc in documents:
        print(doc) if verbose else False
        token_count, norm_obj_score, norm_pos_score, norm_neg_score, norm_final_score, final_sentiment = analyze_sentiment_sentiwordnet_lexicon(str(doc))
        data = {'tokens_in_sentiment_analysis': token_count, 'Avg_obj_score': norm_obj_score, 'Avg_pos_score': norm_pos_score,
                'Avg_neg_score': norm_neg_score,
                'Avg_overall_score': norm_final_score, 'final_sentiment': final_sentiment}
        sentiment_frame = sentiment_frame.append(pd.DataFrame(data, index=[0]), ignore_index=True)

    df = pd.concat([df, sentiment_frame], axis='columns', join='outer', ignore_index=False, sort=False)
    df.to_csv(df_sentiment, index_label='index')
    print(datetime.datetime.utcnow().strftime("%H:%M:%S") + ' - Sentiment analysis completed')
    print(df.info()) if verbose else False



