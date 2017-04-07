from movie_data import *
from sklearn.cluster import KMeans
import numpy as np

class MoviePredict():
    def __init__(self):
        self.data = Data()
        self.users = []
        self.movies = []
        self.train = []
        self.test = []
        self.genres = []
        self.movie_cluster = None

    def load_data(self):
        self.users = self.data.user_data()
        self.movies = self.data.movies_data()
        self.genres = self.data.get_genres()
        self.train = np.zeros((len(self.users),len(self.movies)))
        self.data.create_rating_matrix(self.train)

    def movieClustering(self):
        movie_genre = []
        for m in self.movies:
            mg = []
            for g in self.genres:
                mg.append(m.genre[g])
            movie_genre.append(mg)

        movie_genre = np.array(movie_genre)
        self.movie_cluster = KMeans(n_clusters=len(self.genres)).fit_predict(movie_genre)

    def create_ratingmatrix(self):
        print "Generating Rating matrix"
        avg = np.zeros(len(self.genres))
        ratings = [[ [] for j in range(len(self.genres))] for i in range(len(self.users))]
        for user in self.users:
            for m in self.movies:
                rate = self.train[user.id][m.id]
                if rate != 0:
                    genre = self.movie_cluster[m.id]
                    ratings[user.id][genre].append(rate)

            for i in range(len(self.genres)):
                if ratings[user.id][i] != []:
                    ratings[user.id][i] = np.mean(ratings[user.id][i])
                else:
                    ratings[user.id][i] = 0

        self.ratings = ratings

    def calculate_avgUserRating(self):
        for user in self.users:
            rating = self.ratings[user.id]
            user.avg_rating = np.mean(rating)

    def normalize_rating(self):
        norm = [[[] for j in range(len(self.genres))] for i in range(len(self.users))]
        for user in self.users:
            for i in range(len(self.genres)):
                norm[user.id][i] = [x - user.avg_rating for x in self.ratings[user.id]]

        self.normRating = norm

    def calculate_pearsonCC(self):
        print "Generating Similarity Matrix"
        pcs = [[0 for j in range(len(self.users))] for i in range(len(self.users))]
        for userA in self.users:
            for userB in self.users:
                if userA != userB:
                    rateA = [ x - userA.avg_rating for x in self.ratings[userA.id]]
                    rateB = [ x - userB.avg_rating for x in self.ratings[userB.id]]
                    pcs[userA.id][userB.id] = np.dot(rateA,rateB)


    def guess(self,user,movie,top_n):
        gid = self.movie_cluster[movie -1]
        top_similar = np.argsort(self.normRating[user-1])[-top_n:]
        s,c = 0,0
        for t in top_similar:
            if self.normRating[t][gid] != 0:
                s += self.normRating[t][gid]
                c += 1

        rate = self.users[user-1].avg_rating + float(s)/c
        if rate < 1.0:
            return 1.0
        elif rate > 5.0:
            return 5.0
        else:
            return rate

    def load_allData(self):
        self.load_data()
        self.movieClustering()
        self.create_ratingmatrix()
        self.calculate_avgUserRating()
        self.normalize_rating()
        self.calculate_pearsonCC()


obj = MoviePredict()
obj.load_allData()