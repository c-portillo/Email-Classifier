import os
import re
import random
import operator
import time
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from functools import reduce


# parses an email file
def parse_emails(file_list, folder):
    for i in tqdm(range(len(file_list)), leave=False, desc="Parsing Emails"):
        file = file_list[i]
        file_path = folder + file
        with open(file_path, "r", errors='ignore') as email:
            email_message = email.read().replace("\n", " ")

            content = get_words(email_message)
            email_message_dict[file] = content

            true_labels[file] = folder[:-1]
    time.sleep(0.01)    # added for tqdm progress-bar performance purposes, can be removed


# extracts the words from an email, returns list of words
def get_words(message):
    ps = PorterStemmer()
    temp_dict = {}
    temp = re.sub('[^A-Za-z]', ' ', message).split()
    for x in range(len(temp)):
        w = temp[x].lower()
        temp[x] = ps.stem(w)    # stem words to get root word
    words = [word for word in temp if word not in stop_words]
    for w in words:
        if w in temp_dict:
            temp_dict[w] += 1
        else:
            temp_dict[w] = 1
    return temp_dict


# takes in a collection of emails, creates a word list to be used for Bag-of-Words vectors
def get_word_list(file_list):
    word_dict = {}
    for file in file_list:
        email_message = email_message_dict[file]
        for word in email_message:
            if word in word_dict.keys():
                word_dict[word] += 1
            else:
                word_dict[word] = 1

    sorted_words = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)
    ignore_threshold = 100    # words with frequency less than ignore_threshold will be ignored
    for i in range(len(sorted_words)):
        if sorted_words[i][1] < ignore_threshold:
            sorted_words = sorted_words[:i]     # slice here, words beyond sorted_words[i] are less than threshold
            break

    list_of_words = [w[0] for w in sorted_words]
    return list_of_words


# creates vectors representing emails
def get_vectors(file_list, wrd_lst):
    #for file in file_list:
    for x in tqdm(range(len(file_list)), leave=False, desc="Generating Vectors"):
        file = file_list[x]
        vector = []
        content_dict = email_message_dict[file]

        for j in range(len(wrd_lst)):
            if wrd_lst[j] in content_dict.keys():
                vector.append(content_dict[wrd_lst[j]])
            else:
                vector.append(0)

        email_vector_dict[file] = vector
    time.sleep(0.01)


# partitions data into training and testing
def get_train_test(file_list):
    train = []
    test = []
    size = len(file_list)
    test_size = int(0.2 * size)     # adjust test size as preferred, default setting: training 80%, testing 20%
    test_indices = random.sample(range(size), test_size)    # randomly select test data

    for x in range(size):
        if x in test_indices:
            test.append(file_list[x])
        else:
            train.append(file_list[x])

    return [train, test]


# Implementation of naive bayes, returns accuracy rate
def naive_bayes(all_data, printer):
    if printer:
        print('Running Naive Bayes')
    dimension = len(email_vector_dict[train_data[0]])  # get dimension of vectors
    below_mean_spam = []
    below_mean_ham = []
    train = all_data[0]
    test = all_data[1]
    train_spam = []
    train_ham = []
    means = []

    # find the mean value for each attribute (AKA for each dimension)
    for d in range(dimension):
        below_mean_spam.append(0)
        below_mean_ham.append(0)
        means.append(0)

    for file in train:
        email = email_vector_dict[file]
        for d in range(dimension):
            means[d] += email[d]
        if true_labels[file] == 'spam':
            train_spam.append(email)
        else:
            train_ham.append(email)

    for m in range(len(means)):
        means[m] = means[m]/len(train_data)

    # P(A | B) = (P(B | A) * P(A)) / P(B)
    for email in train_spam:
        for d in range(dimension):
            if email[d] <= means[d]:
                below_mean_spam[d] += 1/(len(train_spam))

    for email in train_ham:
        for d in range(dimension):
            if email[d] <= means[d]:
                below_mean_ham[d] += 1/(len(train_ham))

    # make predictions and check with true labels
    predictions = []
    correct_labels = []

    for file in test:
        spam_mults = []
        ham_mults = []
        vect = email_vector_dict[file]

        for d in range(dimension):
            if vect[d] <= means[d]:
                spam_mults.append(below_mean_spam[d])
                ham_mults.append(below_mean_ham[d])
            else:
                spam_mults.append(1 - below_mean_spam[d])
                ham_mults.append(1 - below_mean_ham[d])

        prob_spam = reduce(lambda x, y: x*y, spam_mults)
        prob_ham = reduce(lambda x, y: x*y, ham_mults)

        if prob_spam > prob_ham:
            predictions.append('spam')
        else:
            predictions.append('ham')

        # add true label to correct_labels
        correct_labels.append(true_labels[file])

    correct = 0
    for c in range(len(correct_labels)):
        if predictions[c] == correct_labels[c]:
            correct += 1

    correct = correct/len(predictions)              # accuracy as a float between 0 and 1
    correct_percent = round((correct * 100), 2)       # accuracy represented as a percent

    if printer:
        print('Predictions: ' + str(predictions))
        print('True labels: ' + str(correct_labels))

    print('Accuracy: ' + str(correct_percent) + '%')

    return correct


# Data: 'spam/' is a folder with spam emails, 'ham/' is a folder with non-spam email
spam_files = os.listdir('spam/')    # list of all file names in spam folder
ham_files = os.listdir('ham/')      # list of all file names in ham folder
all_files = spam_files + ham_files  # list of all email file names
random.shuffle(all_files)

true_labels = {}    # dictionary containing true labels - will be used to test predictions
email_message_dict = {}     # dictionary with email info    {email_ID: [message, label, data_vector]}
email_vector_dict = {}

stop_words = set(stopwords.words('english'))    # set stopwords, common words to be ignored
stop_words.add('subject')                       # add 'subject' to stopwords, as 'subject' is in all emails

parse_emails(ham_files, 'ham/')
parse_emails(spam_files, 'spam/')


I = 200  # number of iterations
print_predictions = False
accuracy_sum = 0

for i in range(I):

    print('Iteration ' + str(i))
    # partition all_files into training and testing data, data = [training_data, testing_data]
    data = get_train_test(all_files)
    train_data = data[0]
    test_data = data[1]

    word_list = get_word_list(train_data)
    get_vectors(train_data+test_data, word_list)
    ith_accuracy = naive_bayes(data, print_predictions)
    accuracy_sum += ith_accuracy

average_accuracy = accuracy_sum/I
average_accuracy_percentage = round((average_accuracy * 100), 2)

print("\nAverage Accuracy Over " + str(I) + " Iterations: " + str(average_accuracy_percentage) + "%")