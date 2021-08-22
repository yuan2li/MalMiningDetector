import re
import joblib
import warnings
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

data_path = '/mnt/diskd/datasets/mining'
base_path = './idata'
pic_path = './pic'

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.figsize'] = (7.0, 5.0)
plt.rcParams['savefig.dpi'] = 500


def get_training_data(category_path, category):
    """Get string feature of certain category samples.

    :param category_path: path of certain category samples
    :param category: string-format flag of category
    :return: string of certain category samples
    """

    with open(f'{base_path}/{category}.txt', 'r') as fp:
        id_ = fp.read().split()
    list_ = []
    # Get printable strings joined by space in each sample of the category
    for path in id_:
        with open(f'{data_path}/{category_path}/{path}', 'rb') as fp:
            strings = fp.read().decode('utf-8', errors='ignore')
        raw_words = re.findall('[a-zA-Z]+', strings)
        words_space = ' '.join(w for w in raw_words if 4 < len(w) < 20)
        list_.append(words_space)
    df = pd.DataFrame()
    df['words'] = list_
    df.to_csv(f'{base_path}/{category}.csv', index=False)
    return df


def merge(black, white):
    """Get training set by merge black and white sample.

    :param black: dataframe-format of strings in black samples
    :param white: dataframe-format of strings in white samples
    """

    df = black.append(white)
    df['labels'] = [1 for _ in range(black.shape[0])] + [0 for _ in range(white.shape[0])]
    df.to_csv(f'{base_path}/train_data.csv', index=False)


def plot_wordcloud(strings, category):
    """Plot word cloud picture for strings in black and white samples respectively.

    :param strings: string of certain category samples
    :param category: string-format flag of category
    """

    string_of_all = ' '.join(strings['words'].tolist())
    wordc = WordCloud(width=700,
                      height=500,
                      min_font_size=10,
                      max_font_size=100).generate(string_of_all)
    plt.imshow(wordc)
    plt.axis('off')
    plt.savefig(f'{pic_path}/{category}_wordcloud.pdf', dpi=500)


def display_top_strings(strings, number, category):
    """Display top-N most common strings of each category.

    :param strings: string of certain category samples
    :param number: number of top strings to be displayed
    :param category: string-format flag of category
    """

    string_of_all = []
    for string_single in strings.words.tolist():
        string_of_all.extend(string_single.split())
    string_counts = Counter(string_of_all)
    string_counts_top = string_counts.most_common(number)
    axis_y = np.arange(number, 0, -1)
    axis_x = np.array([s[1] for s in string_counts_top])
    label_x = [s[0] for s in string_counts_top]
    plt.barh(axis_y, width=axis_x, tick_label=label_x)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.savefig(f'{pic_path}/{category}_topstring.pdf', bbox_inches='tight')


def tfidf_vectorize(df):
    """Vectorize training data by tf-idf algorithm.

    :param df: string features of data
    """

    vectorizer = TfidfVectorizer(min_df=3, max_df=0.9, max_features=3000)
    tfidf_features = vectorizer.fit_transform(df.words.tolist())
    with open(f'{base_path}/tfidf_features.pkl', 'wb') as fp:
        joblib.dump(tfidf_features, fp)
    with open(f'{base_path}/train_labels.pkl', 'wb') as fp:
        joblib.dump(df.labels, fp)


def tsne_plot(feature_vector, labels):
    """Visualize feature vector by T-SNE algorithm.

    :param feature_vector: feature vector obtained by tf-idf
    :param labels:category label of data
    """

    tsne = TSNE(n_components=2, perplexity=300, random_state=42).fit_transform(feature_vector)
    plt.scatter(tsne[:, 0], tsne[:, 1], alpha=0.6,
                c=labels, cmap=plt.cm.get_cmap('rainbow', 2))
    plt.title("Features Visualization", fontweight='bold')
    plt.ylim([-30, 50])
    plt.xlim([-50, 30])
    plt.grid(False)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.savefig(f'{pic_path}/tsne.pdf', bbox_inches='tight')
