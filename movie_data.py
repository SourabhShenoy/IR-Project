import numpy as np
class User:
    def __init__(self,user_id,age,sex,occupation,zipcode):
        self.id = user_id
        self.age = age
        self.sex = sex
        self.occupation = occupation
        self.zipcode = zipcode
        self.avg_rating = 0
        self.pref_genre = []

class Movie:
    def __init__(self,movie_id,name,release_data,imdb_link,genre):
        self.id = movie_id
        self.name = name
        self.release_date = release_data
        self.imdb_link = imdb_link
        self.genre = genre


class Rating:
    '''
    def __init__(self,user_id,movie_id,rating):
        self.user_id = user_id
        self.movie_id = movie_id
        self.rating = rating
    '''
    def rating_matrix(self,rate_matrix,filename):
        # f = open(".\ml-100k\u1.base", "r")
        f = open(filename, "r")
        ratings = f.readlines()
        for r in ratings:
            r = r.split("\t")
            rate_matrix[int(r[0])-1][int(r[1])-1] = int(r[2])

class Data:
    def __init__(self):
        self.genres_list = self.get_genres()
        self.genre_corr = [[0 for col in range(19)] for row in range(19)]

    def user_data(self):
        users = []
        f = open(".\ml-100k\u.user","r")
        lines = f.readlines()
        for line in lines:
            data = line.split("|")
            new_user = User(int(data[0])-1,int(data[1]),data[2],data[3],data[4])
            users.append(new_user)
        return users

    def movies_data(self):
        movies = []
        f = open(".\ml-100k\u.item","r")
        lines = f.readlines()
        # making a movie object
        for line in lines:
            new_movie = []
            data = line.split("|")
            genre = {}
            i = 5
            for g in self.genres_list:
                genre[g] = int(data[i])
                i += 1
            new_movie = Movie(int(data[0])-1,data[1],data[2],data[3],genre)
            movies.append(new_movie)
        return movies

    def get_genres(self):
        f_genre = open(".\ml-100k\u.genre", "r")
        genres = []
        # getting all the genres
        lines = f_genre.readlines()
        for line in lines:
            line = line.split("|")
            genres.append(line[0])
        return genres[:-1]

    def create_rating_matrix(self,rate_matrix,filename):
        r = Rating()
        #rate_matrix = np.zeros((943, 1682))
        r.rating_matrix(rate_matrix,filename)
        #print rate_matrix[0]

    def genre_correlation(self):
        f = open(".\ml-100k\u.item", "r")
        lines = f.readlines()
        for i in range(19):
            self.genre_corr[i][i] = 1
        for line in lines:
            data = line.split("|")
            genre = data[5:]
            for i in range(19):
                if genre[i] == 1:
                    for j in range(i+1,19):
                        if genre[j] == 1:
                            self.genre_corr[i][j] += 1


#d = Data()
#d.create_rating_matrix()