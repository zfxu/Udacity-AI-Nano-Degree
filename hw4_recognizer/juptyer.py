import math
import time
import timeit
import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from my_model_selectors import SelectorCV
from asl_utils import test_features_tryit
from sklearn.model_selection import KFold
from my_model_selectors import SelectorBIC
from my_model_selectors import SelectorDIC
from my_model_selectors import SelectorConstant

from asl_data import AslDb
from asl_utils import show_errors
from asl_utils import test_std_tryit

import my_model_selectors
from my_recognizer import recognize

# from matplotlib import (cm, pyplot as plt, mlab)


warnings.filterwarnings("ignore", category=RuntimeWarning)


def zScoreScaling(feature_name):
    X = asl.df[feature_name]
    Xmean = asl.df['speaker'].map(df_means[feature_name])
    Xstd = asl.df['speaker'].map(df_std[feature_name])
    return (X - Xmean) / Xstd


def unitScaling(feature_name):
    a = -1
    b = 1
    X = asl.df[feature_name]
    X_max = asl.df['speaker'].map(df_max[feature_name])
    X_min = asl.df['speaker'].map(df_min[feature_name])
    return a + (X - X_min) * (b - a) / (X_max - X_min)
    # return (X - min(X)) / (max(X)-min(X))


def unitScaling2(feature_name):
    a = -1
    b = 1
    X = asl.df[feature_name]
    X_max = max(X)
    X_min = min(X)
    return a + (X - X_min) * (b - a) / (X_max - X_min)
    # return (X - min(X)) / (max(X)-min(X))


def cart_raduis_to_polar(x, y):
    return np.sqrt(np.power(x, 2) + np.power(y, 2))


def cart_angle_to_polar(x, y):
    return np.arctan2(x, y)


def calc_delta(x):
    return x.diff().fillna(0)


# def train_a_word(word, num_hidden_states, features):
#     warnings.filterwarnings("ignore", category=DeprecationWarning)
#     training = asl.build_training(features)
#     X, lengths = training.get_word_Xlengths(word)
#     model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
#     logL = model.score(X, lengths)
#     return model, logL


def show_model_stats(word, model):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        print()


# def visualize(word, model):
#     """ visualize the input model for a particular word """
#     variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
#     figures = []
#     for parm_idx in range(len(model.means_[0])):
#         xmin = int(min(model.means_[:,parm_idx]) - max(variance[:,parm_idx]))
#         xmax = int(max(model.means_[:,parm_idx]) + max(variance[:,parm_idx]))
#         fig, axs = plt.subplots(model.n_components, sharex=True, sharey=False)
#         colours = cm.rainbow(np.linspace(0, 1, model.n_components))
#         for i, (ax, colour) in enumerate(zip(axs, colours)):
#             x = np.linspace(xmin, xmax, 100)
#             mu = model.means_[i,parm_idx]
#             sigma = math.sqrt(np.diag(model.covars_[i])[parm_idx])
#             ax.plot(x, mlab.normpdf(x, mu, sigma), c=colour)
#             ax.set_title("{} feature {} hidden state #{}".format(word, parm_idx, i))
#
#             ax.grid(True)
#         figures.append(plt)
#     for p in figures:
#         p.show()


def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict


# Counting time for feature extraction
startTime = time.time()

# Load time
asl = AslDb() # initializes the database
# asl.df.head() # displays the first five rows of the asl database, indexed by video and frame\

# Feature extraction - ground
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']

# Feature extraction - norm
# training = asl.build_training(features_ground)
df_means = asl.df.groupby('speaker').mean()
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
df_std = asl.df.groupby('speaker').std()
asl.df['norm-rx'] = zScoreScaling('right-x')
asl.df['norm-ry'] = zScoreScaling('right-y')
asl.df['norm-lx'] = zScoreScaling('left-x')
asl.df['norm-ly'] = zScoreScaling('left-y')
features_norm = ['norm-rx', 'norm-ry', 'norm-lx', 'norm-ly']

df_max = asl.df.groupby('speaker').max()
df_min = asl.df.groupby('speaker').min()
asl.df['unitNorm-rx'] = unitScaling('right-x')
asl.df['unitNorm-ry'] = unitScaling('right-y')
asl.df['unitNorm-lx'] = unitScaling('left-x')
asl.df['unitNorm-ly'] = unitScaling('left-y')
features_unitNorm = ['unitNorm-rx', 'unitNorm-ry', 'unitNorm-lx', 'unitNorm-ly']

# Feature extraction - polar
asl.df['polar-rr'] = cart_raduis_to_polar(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-lr'] = cart_raduis_to_polar(asl.df['grnd-lx'], asl.df['grnd-ly'])
asl.df['polar-rtheta'] = cart_angle_to_polar(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-ltheta'] = cart_angle_to_polar(asl.df['grnd-lx'], asl.df['grnd-ly'])
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

asl.df['unitPolar-rr'] = unitScaling2('polar-rr')
asl.df['unitPolar-lr'] = unitScaling2('polar-lr')
asl.df['unitPolar-rtheta'] = unitScaling2('polar-rtheta')
asl.df['unitPolar-ltheta'] = unitScaling2('polar-ltheta')
features_unitPolar = ['unitPolar-rr', 'unitPolar-lr', 'unitPolar-rtheta', 'unitPolar-ltheta']

# Feature extraction - delta
asl.df['delta-rx'] = calc_delta(asl.df['right-x'])
asl.df['delta-ry'] = calc_delta(asl.df['right-y'])
asl.df['delta-lx'] = calc_delta(asl.df['left-x'])
asl.df['delta-ly'] = calc_delta(asl.df['left-y'])
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

asl.df['unitDelta-rx'] = unitScaling2('delta-rx')
asl.df['unitDelta-ry'] = unitScaling2('delta-ry')
asl.df['unitDelta-lx'] = unitScaling2('delta-lx')
asl.df['unitDelta-ly'] = unitScaling2('delta-ly')
features_unitDelta = ['unitDelta-rx', 'unitDelta-ry', 'unitDelta-lx', 'unitDelta-ly']

features_norm_grnd = ['norm-grnd-rx', 'norm-grnd-ry', 'norm-grnd-lx','norm-grnd-ly']
df_std = asl.df.groupby('speaker').std()
df_mean = asl.df.groupby('speaker').mean()
for feature, root_feat in zip(features_norm_grnd, features_ground):
    asl.df[feature] = (asl.df[root_feat] - asl.df['speaker'].map(df_means[root_feat])) / asl.df['speaker'].map(df_std[root_feat])

# Delta Values for Normalized Grounded Features
features_delta_norm_grnd = ['delta-norm-grnd-rx', 'delta-norm-grnd-ry', 'delta-norm-grnd-lx', 'delta-norm-grnd-ly']

asl.df[features_delta_norm_grnd] = asl.df[features_norm_grnd].fillna(0).diff().fillna(0)
endTime = time.time()

# define a list named 'features_custom' for building the training set
# features_custom = features_norm_grnd + features_delta_norm_grnd + features_polar
features_custom1 = features_ground + features_norm + features_polar + features_delta
features_custom2 = features_norm + features_polar + features_delta + features_norm_grnd + features_delta_norm_grnd
features_custom3 = features_unitNorm + features_unitPolar + features_unitDelta
print("Feature extraction finishes in " + str(endTime-startTime) + " seconds.\n")

# startTime = time.time()
# features = features_ground # change as needed
# model_selector = my_model_selectors.SelectorBIC # change as needed
# models = train_all_words(features, model_selector)
# test_set = asl.build_test(features)
# probabilities, guesses = recognize(models, test_set)
# endTime = time.time()
# print("features_ground + BIC finishes in " + str(endTime-startTime) + " seconds.")
# show_errors(guesses, test_set)
# print(" ")
#
# startTime = time.time()
# features = features_norm # change as needed
# model_selector = my_model_selectors.SelectorBIC # change as needed
# models = train_all_words(features, model_selector)
# test_set = asl.build_test(features)
# probabilities, guesses = recognize(models, test_set)
# endTime = time.time()
# print("features_norm + BIC finishes in " + str(endTime-startTime) + " seconds.")
# show_errors(guesses, test_set)
# print(" ")
#
# startTime = time.time()
# features = features_polar # change as needed
# model_selector = my_model_selectors.SelectorBIC # change as needed
# models = train_all_words(features, model_selector)
# test_set = asl.build_test(features)
# probabilities, guesses = recognize(models, test_set)
# endTime = time.time()
# print("features_polar + BIC finishes in " + str(endTime-startTime) + " seconds.")
# show_errors(guesses, test_set)
# print(" ")
#
# startTime = time.time()
# features = features_delta # change as needed
# model_selector = my_model_selectors.SelectorBIC # change as needed
# models = train_all_words(features, model_selector)
# test_set = asl.build_test(features)
# probabilities, guesses = recognize(models, test_set)
# endTime = time.time()
# print("features_delta + BIC finishes in " + str(endTime-startTime) + " seconds.")
# show_errors(guesses, test_set)
# print(" ")
#
# startTime = time.time()
# features = features_norm_grnd # change as needed
# model_selector = my_model_selectors.SelectorBIC # change as needed
# models = train_all_words(features, model_selector)
# test_set = asl.build_test(features)
# probabilities, guesses = recognize(models, test_set)
# endTime = time.time()
# print("features_norm_grnd + BIC finishes in " + str(endTime-startTime) + " seconds.")
# show_errors(guesses, test_set)
# print(" ")
#
# startTime = time.time()
# features = features_delta_norm_grnd # change as needed
# model_selector = my_model_selectors.SelectorBIC # change as needed
# models = train_all_words(features, model_selector)
# test_set = asl.build_test(features)
# probabilities, guesses = recognize(models, test_set)
# endTime = time.time()
# print("features_delta_norm_grnd + BIC finishes in " + str(endTime-startTime) + " seconds.")
# show_errors(guesses, test_set)
# print(" ")

# startTime = time.time()
# features = features_unitNorm # change as needed
# model_selector = my_model_selectors.SelectorBIC # change as needed
# models = train_all_words(features, model_selector)
# test_set = asl.build_test(features)
# probabilities, guesses = recognize(models, test_set)
# endTime = time.time()
# print("features_unitNorm + BIC finishes in " + str(endTime-startTime) + " seconds.")
# show_errors(guesses, test_set)
# print(" ")
#
# startTime = time.time()
# features = features_unitPolar # change as needed
# model_selector = my_model_selectors.SelectorBIC # change as needed
# models = train_all_words(features, model_selector)
# test_set = asl.build_test(features)
# probabilities, guesses = recognize(models, test_set)
# endTime = time.time()
# print("features_unitPolar + BIC finishes in " + str(endTime-startTime) + " seconds.")
# show_errors(guesses, test_set)
# print(" ")
#
# startTime = time.time()
# features = features_unitDelta # change as needed
# model_selector = my_model_selectors.SelectorBIC # change as needed
# models = train_all_words(features, model_selector)
# test_set = asl.build_test(features)
# probabilities, guesses = recognize(models, test_set)
# endTime = time.time()
# print("features_unitDelta + BIC finishes in " + str(endTime-startTime) + " seconds.")
# show_errors(guesses, test_set)
# print(" ")

startTime = time.time()
features = features_ground # change as needed
model_selector = my_model_selectors.SelectorDIC # change as needed
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
endTime = time.time()
print("features_custom3 + DIC finishes in " + str(endTime-startTime) + " seconds.")
show_errors(guesses, test_set)
print(" ")

