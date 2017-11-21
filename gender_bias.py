import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

b_dir = '../glove-global-vectors-for-word-representation/'
b2_dir = './ted-talks/'

# preparing embedding index
embeddings_index = {}
with open(b_dir + 'glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


speaker_gender = pd.read_csv('../ted_speaker_gender.csv')
df = pd.read_csv(b2_dir + 'ted_main.csv')

df_with_gender = pd.merge(df, speaker_gender, how='inner',
                          left_on='main_speaker', right_on='name_in_profile')

df_with_gender['tags_eval'] = df_with_gender['tags'].apply(
    lambda x: ast.literal_eval(x))

df_with_gender['he_she_count'] = df_with_gender['he_count'] > df_with_gender['she_count']


def get_tag_word_list(tag_list):
    word_list = []
    for tag in tag_list:
        word_list.extend(tag.lower().split(' '))
    return word_list


def get_similarity(word_list, he_she):
    total = 0
    for word in word_list:
        if word in embeddings_index:
            cosine_similarity = np.dot(embeddings_index[word], embeddings_index[he_she]) / (
                np.linalg.norm(embeddings_index[word]) * np.linalg.norm(embeddings_index[he_she]))
            total += cosine_similarity
    return total


def get_similarity_max(word_list, he_she):
    sim_list = [0]
    for word in word_list:
        if word in embeddings_index:
            cosine_similarity = np.dot(embeddings_index[word], embeddings_index[he_she]) / (
                np.linalg.norm(embeddings_index[word]) * np.linalg.norm(embeddings_index[he_she]))
            sim_list.append(cosine_similarity)
    return max(sim_list)


def get_similarity_mean(word_list, he_she):
    vec_list = []
    for word in word_list:
        if word in embeddings_index:
            vec_list.append(embeddings_index[word])
    if len(vec_list) > 0:
        mean_vec = np.array(vec_list).mean(axis=0)
        cosine_similarity = np.dot(mean_vec, embeddings_index[he_she]) / (
            np.linalg.norm(mean_vec) * np.linalg.norm(embeddings_index[he_she]))
    else:
        cosine_similarity = 0
    return cosine_similarity


colors = {False: 'red', True: 'blue'}

import matplotlib.lines as mlines


def plot_legend():
    blue_line = mlines.Line2D([], [], color=colors[True], marker='o',
                              markersize=15, label='male')
    red_line = mlines.Line2D([], [], color=colors[False], marker='o',
                             markersize=15, label='female')
    plt.legend(handles=[blue_line, red_line])


def bias(col, split_func=lambda x: x.split()):
    he_col = 'he_%s' % col
    she_col = 'she_%s' % col
    df_with_gender[he_col] = df_with_gender[col].apply(
        lambda x: get_similarity_mean(split_func(x), 'he'))
    df_with_gender[she_col] = df_with_gender[col].apply(
        lambda x: get_similarity_mean(split_func(x), 'she'))
    fig, ax = plt.subplots()
    ax.scatter(df_with_gender[he_col], df_with_gender[she_col],
               c=df_with_gender['he_she_count'].apply(lambda x: colors[x]))
    plt.xlabel(he_col)
    plt.ylabel(she_col)
    plot_legend()
    # plt.show()
    plt.savefig('bias_%s.png' % col, dpi=200)
    plt.clf()
    plt.cla()
    plt.close()


bias('title')
bias('tags_eval', split_func=get_tag_word_list)

fig, ax = plt.subplots()
ax.scatter(df_with_gender['views'], df_with_gender['comments'],
           c=df_with_gender['he_she_count'].apply(lambda x: colors[x]))

plt.xlabel('views')
plt.ylabel('comments')
plot_legend()
# plt.show()
plt.savefig('bias_%s.png' % "views_comments", dpi=200)
plt.clf()
plt.cla()
plt.close()
