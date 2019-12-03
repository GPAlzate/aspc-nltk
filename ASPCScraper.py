import requests
import os
import csv
import string
from dotenv import load_dotenv
from bs4 import BeautifulSoup

class StudentReview:
    """Contains individual student's comments and ratings for a class

    All data defaults to -1 if there are no reviews for the course

    Args:
        course_num (str):       the name and code fo the course
        course_rating (int):    1-5 star rating of the course quality
        course_comment (str):   Review of the course by the student
        hours (int):            number of hours spent working per week
        chlng_rating (int):     1-5 star rating of challenge
        inclu_rating (int):     1-5 star rating of inclusivity

    Attributes:
        num_empty (int):    number of courses with no reviews
    """

    num_empty = 0

    def __init__(self, course_num = "???", course_comment = "n/a", course_rating = -1,
            chlng_rating = -1, inclu_rating = -1, hours = -1):

        self.course_num = course_num
        self.course_rating = course_rating
        self.course_comment = course_comment
        self.hours = hours
        self.chlng_rating = chlng_rating
        self.inclu_rating = inclu_rating

        if course_rating < 0:
            StudentReview.num_empty += 1


    def __str__(self):
        """ Returns a string representation of a StudentReview.

        e.g. 
            Course rating: <Rating>/5.0
            Course comment: <Some comment followed by ellipses> ... 
            Hours per week: <Hours> 
            Challenge Rating: <Challenge>/5.0 
            Inclusivity Rating: <Inclusivity>/5.0 
        """
        num = f"Course code: {self.course_num} \n"
        rat = f"Course rating: {self.course_rating}/5.0 \n"
        com = f"Course comment: {' '.join(self.course_comment.split(' ')[:10])}... \n"
        hrs = f"Hours per week: {self.hours} \n"
        cha = f"Challenge Rating: {self.chlng_rating}/5.0 \n"
        inc = f"Inclusivity Rating: {self.inclu_rating}/5.0 \n"

        return num + rat + com + hrs + cha + inc

class ASPCScraper:
    """Scrapes all the reviews for every course from ASPC course reviews.

    Collects individual students' course rating, challenge rating, inclusivity
    rating, and general comments. Also includes estimates for hours of work per week.
    """

    def GetLoginData(self):
        """ Get payload information for logging into ASPC.

        This takes username, display name (school), password, and execution code
        from some .env file
        """
        # load login info (from a .env because it contains login info)
        load_dotenv()
        username = os.getenv("USERNAME")
        dispname = os.getenv("DISPNAME")
        password = os.getenv("PASSWORD")
        execution = os.getenv("EXECUTION")

        # return dictionary that is payload of post request
        return {
            "username" : username,
            "dispname": dispname,
            "password": password,
            "execution": execution,
            "_eventId": "submit"
        }

    def GetCourseReviews(self, i, reviews, misc, ratings, comments):
        """Creates a StudentReview from a course review and adds it to a list of
        reviews.

        Args:
            reviews (list):     the list to which we add reviews of a course
            misc (list):        challenge/inclusivity/hours per week ratings
            ratings (list):     array of 1-5 stars that a student rated a course
            comments (list):    comment of a course
            i (int):            count of which review 'block' we are on
        """
        # get course quality (1-5 stars)
        course_rating = len(ratings[i].findAll("i", class_="fa fa-star"))

        # student comment
        course_comment = comments[i].find("p", class_="subtitle").string

        # remove punctuation
        # stripped = course_comment.translate(str.maketrans('', '', string.punctuation))

        # convert each whitespace to single space
        course_comment = ' '.join(course_comment.split())

        # each p corresponds to weekly hours, challenge (1-5) inclusivity (1-5)
        other_ratings = misc[i].findAll("p")
        hours = float(other_ratings[0].find("span").string.split(' ')[0])
        chlng_rating = len(other_ratings[1].findAll("i", class_="fa fa-star"))
        inclu_rating = len(other_ratings[2].findAll("i", class_="fa fa-star"))

        # add a review for this class
        return StudentReview(
                course_num = course_num,
                course_rating = course_rating,
                course_comment = course_comment,
                hours = hours,
                chlng_rating = chlng_rating,
                inclu_rating = inclu_rating
            )
    
    def write_to_csv(self, course_review_data):
        """Writes all reviews to a csv file.

        The values are ordered as course number, student comment, challenge (1-5),
        inclusivity (1-5), weekly hours, rating

        Args:
            course_review_data (dict):  a mapping from each course to it reviews

        """
        with open("course_reviews2.csv", "w") as file:

            # create CSV file
            writer = csv.writer(file, delimiter = "\t", quotechar = "|",
                                quoting = csv.QUOTE_MINIMAL)

            # and fill it
            writer.writerow(["number", "comment", "challenge",
                            "inclusivity", "hours", "rating" ])
            for data in course_review_data:
                for course in course_review_data[data]:
                    writer.writerow(
                        [data, course.course_comment, course.chlng_rating,
                         course.inclu_rating, course.hours, course.course_rating]
                    )
            


# scraper object for methods
aspc = ASPCScraper()

# login data for CAS authentication
data = aspc.GetLoginData()

# URL to send login info to
loginURL = "https://webauth.claremont.edu/cas/login?service=https%3A%2F%2Fssocas.campus.pomona.edu%2Fcas%2Flogin%3Fclient_name%3DCasClient"

# map from course number to course review
course_review_data = {}

# request session to keep login cookies
with requests.Session() as s:

    # CAS authentication process to login to ASPC
    cas = s.post(loginURL, data=data)
    response = s.get("https://pomonastudents.org/login/cas")

    # scrape reviews/rating for all courses
    for page in range(1, 7767):

        # soup: course review page parser
        review_page = s.get(f"https://pomonastudents.org/courses/{page}").text
        soup = BeautifulSoup(review_page, "lxml")

        # get course number
        course_num = soup.find("h2", class_="subtitle").string.strip()
        course_num = ' '.join(course_num.split(' ')[2:])

        # ignore first 6 and last 4 'column' tags,
        # and delete every 4th element (review written <date>)
        # will eventually just hold miscellaneous course ratings after slicing
        misc = soup.findAll("div", class_="column")[6:-4]
        del misc[3::4]

        # if no reviews, delegate to default ctor w/ no data
        if not len(misc):
            course_review_data[course_num] = [StudentReview(course_num=course_num)]
            continue

        # get course ratings (every 3rd 'column' div)
        ratings = misc[::3]
        del misc[::3]

        # get course comments (every 2nd 'column' div)
        comments = misc[::2]
        del misc[::2]

        # list of all reviews for a class
        reviews = []

        # new object for every review; add course and all its reviews to map
        num_ratings = len(ratings)
        for i in range(num_ratings):
            reviews.append(
                aspc.GetCourseReviews(i=i, reviews = reviews, misc = misc,
                    ratings=ratings, comments=comments
                )
            )
        # map the course to its reviews
        course_review_data[course_num] = reviews
        
    print(StudentReview.num_empty)

    aspc.write_to_csv(course_review_data)


