"""@authors: Ruben Pacheco-Caldera, Xander Koo, Gabriel Alzate
A sentiment analyzer for ASPC course reviews
"""
# from vader_sentiment.vader_sentiment import *
from vader_sentiment.vader_sentiment import SentimentIntensityAnalyzer
import string
import math
import random
import re
import time
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    # return the compound sentiment analysis score
    return score


def analyze_file(inFile, outFile, startIndex):
    """
    Analyzes the file
    :param inFile: file to read from
    :param outFile: file to write to
    :param startIndex: used to create the review number (id)
    :return: none
    """
    csv_file = open(inFile, "r")
    out_file = open(outFile, "w")
    out_file.write("Review Number, Class ID, Comment Sentiment Score, Rating \n")
    # read header
    csv_file.readline()
    count = 0
    word_with_sentiment = {}

    start = time.time()
    for review in csv_file:

        review = review.split(",")
        class_id = review[0]
        comment = review[1]
        rating = int(review[5])
        # if they wrote a comment and gave a course review
        if comment != "n/a" and rating != "-1":
            comment = comment.split()
            for word in comment:
                word_with_sentiment[word] = sentiment_analyzer_scores(word)["compound"]
            # uncomment to shuffle
            # comment = comment.split()
            # random.shuffle(comment)
            comment = " ".join(comment)
            sentiment_score = sentiment_analyzer_scores(comment)



            max = -1
            prediction = -1
            label = -1
            compound = sentiment_score["compound"]

            if compound >= 0.05:
                prediction = 1
            elif compound >= -0.05:
                prediction = 0
            else:
                prediction = -1

            if rating < 3:
                label = -1
            elif rating == 3:
                label = 0
            else:
                label = 1
            if label == prediction:
                count += 1

            # out_file.write(str(startIndex) + ", " + class_id + ", " + str(sentiment_score) + ", " + str(rating) )

            startIndex += 1
    end = time.time()
    print("finished in " + str(end - start))
    print({k: v for k, v in sorted(word_with_sentiment.items(), key=lambda item: item[1])})
    print(count / startIndex)
    out_file.close()
    print("Finished analyzing. Wrote to file")

def tricky_sentence():
    tricky_sentences = ["Its a love hate relationship",
                        "The class is fucking cool",
                        "this kung fu class is cool",
                        "Must-take class! Professor is VERY insightful about war and violence",
                        "Sentiment analysis has never been good.",
                        "Sentiment analysis has never been this good!",
                        "Most automated sentiment analysis tools are shit.",
                        "With VADER, sentiment analysis is the shit!",
                        "Other sentiment analysis tools can be quite bad.",
                        "On the other hand, VADER is quite bad ass",
                        "VADER is such a badass!",  # slang with punctuation emphasis
                        "Without a doubt, excellent idea.",
                        "Roger Dodger is one of the most compelling variations on this theme.",
                        "Roger Dodger is at least compelling as a variation on the theme.",
                        "Roger Dodger is one of the least compelling variations on this theme.",
                        "Not such a badass after all.",  # Capitalized negation with slang
                        "Without a doubt, an excellent idea."  # "without {any} doubt" as negation
                        ]
    print("----------------------------------------------------")
    print(" - Analyze examples of tricky sentences that cause trouble to other sentiment analysis tools.")
    print("  -- special case idioms - e.g., 'never good' vs 'never this good', or 'bad' vs 'bad ass'.")
    print("  -- special uses of 'least' as negation versus comparison \n")
    for sentence in tricky_sentences:
        vs = sentiment_analyzer_scores(sentence)
        print(sentence, vs)

def preProcessFile(inFile, outFile):
    """
    Analyzes the file
    :param inFile: file to read from
    :param outFile: file to write to
    :param startIndex: used to create the review number (id)
    :return: none
    """
    csv_file = open(inFile, "r")
    out_file = open(outFile, "w")

    # read header
    csv_file.readline()

    for review in csv_file:

        review = review.split("\t")
        class_id = review[0]
        comment = review[1].lower()

        rating = int(review[5])
        count = 0
        # label = 0
        #
        # # if they wrote a comment and gave a course review
        if comment != "n/a" and rating != "-1":
            comment = addSpace(comment)



            if rating < 3:
                label = -1
            elif rating == 3:
                label = 0
            else:
                label = 1



            out_file.write(str(rating) + "\t" + comment + "\n")
            # out_file.write(", " + class_id + ", " + str(sentiment_score) + ", " + str(rating))

    out_file.close()

def addSpace(words):
    words = words.replace("'", "")
    words = [item for item in map(str.strip, re.split("(\W+)", words)) if len(item) > 0]
    return " ".join(words)



def main():
    # analyze_file("course_reviews.csv", "analyzedFile.txt", 1)
     # preProcessFile("course_reviews2.csv", "preprocessed.txt")
     tricky_sentence()
if __name__ == '__main__':
    main()
    # sample = "(Effort you put in) x100 = What you get out of it don't be afraid of her ; don't be afraid of seeming stupid, she knows the material a heck ton better than you do ; go talk to her after class ; do the work!"
    # sia = SentimentIntensityAnalyzer()
    # sentiment_map = sia.polarity_scores(sample)
    # print(sentiment_map)