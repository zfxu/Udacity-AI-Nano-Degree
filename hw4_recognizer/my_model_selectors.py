import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self,all_word_sequences:dict,all_word_Xlengths:dict, this_word:str,n_constant=3,min_n_components=2, max_n_components=10,random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states,
                                    covariance_type="diag",
                                    n_iter=1000,
                                    random_state=self.random_state,
                                    verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """
    select the model with value self.n_constant
    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """
    select the model with the lowest Baysian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components+1

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        bic = {}
        for num_components in range(self.min_n_components, self.max_n_components+1, 1):
            n = num_components
            d = len(self.X[0])
            p = np.power(n,2) + 2*d*n -1
            try:
                # Bayesian information criteria: BIC = -2 * logL + p * logN
                hmm_model = self.base_model(n)
                logL = hmm_model.score(self.X, self.lengths)
                logN = np.log(d)
                bic[num_components] = -2 * logL + p * logN
            except:
                continue
        # print(bic)
        if bic:
            best_model = min(bic, key=bic.get)
            return self.base_model(best_model)
        else:
            return None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        dic = {}
        for num_components in range(self.min_n_components, self.max_n_components + 1, 1):
            try:
                # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
                hmm_model = self.base_model(num_components)
                logL = hmm_model.score(self.X, self.lengths)
                logLlist = []
                for w in self.hwords:
                    if w != self.this_word:
                        X, lengths = self.hwords[w]
                        logLlist.append(hmm_model.score(X, lengths))
                logLavg = np.average(logLlist)
                dic[num_components] = logL - logLavg
            except:
                continue

        if dic:
            best_model = min(dic, key=dic.get)
            return self.base_model(best_model)
        else:
            return None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        if len(self.sequences) < 2:
            return self.base_model(self.n_constant)
        elif len(self.sequences) == 2:
            noFold = 2
        elif len(self.sequences) >= 3:
            noFold = 3
        else:
            return None
        cv = {}
        for num_components in range(self.min_n_components, self.max_n_components + 1):
            splits = KFold(n_splits=noFold)
            scores = []
            for train, test in splits.split(self.sequences):
                train_X, train_lengths = combine_sequences(train, self.sequences)
                test_X, test_lengths = combine_sequences(test, self.sequences)
                try:
                    hmm_model = GaussianHMM(n_components=num_components,
                                            covariance_type="diag",
                                            n_iter=1000,
                                            random_state=self.random_state,
                                            verbose=False).fit(train_X, train_lengths)
                    scores.append(hmm_model.score(test_X, test_lengths))
                except:
                    pass
            cv[num_components] = np.average(scores)

        if cv:
            best_model = max(cv, key=cv.get)
            return self.base_model(best_model)
        else:
            return None




