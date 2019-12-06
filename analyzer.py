"""@authors: Ruben Pacheco-Caldera, Xander Koo, Gabriel Alzate
A sentiment analyzer for ASPC course reviews
"""
# from vader_sentiment.vader_sentiment import *
from vader_sentiment.vader_sentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    # return the compound sentiment analysis score
    return score["compound"]
    # print("{:-<40} {}".format(sentence, str(score)))

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

    for review in csv_file:

        review = review.split(",")
        class_id = review[0]
        comment = review[1]
        rating = review[5]
        # if they wrote a comment and gave a course review
        if comment != "n/a" and rating != "-1":
            sentiment_score = sentiment_analyzer_scores(comment)

            out_file.write(str(startIndex) + ", " + class_id + ", " + str(sentiment_score) + ", " + str(rating))

            startIndex += 1
    out_file.close()
    print("Finished analyzing. Wrote to file")
def main():
    analyze_file("course_reviews.csv", "analyzedFile.txt", 1)

if __name__ == '__main__':
    main()