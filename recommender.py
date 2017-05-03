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
        self.test = np.zeros((len(self.users), len(self.movies)))
        self.data.create_rating_matrix(self.train,"./ml-100k/ub.base")
        self.data.create_rating_matrix(self.test,"./ml-100k/ub.test")

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
            norm[user.id] = [x - user.avg_rating for x in self.ratings[user.id]]

        self.normRating = norm

    def calculate_pearsonCC(self):
        print "Generating Similarity Matrix"
        maxage = 73
        minage = 7
        pcs = [[0 for j in range(len(self.users))] for i in range(len(self.users))]
        for userA in self.users:
            agerange = max(abs(userA.age - minage), abs(userA.age - maxage))
            for userB in self.users:
                if userA != userB:
                    rateA = [x - userA.avg_rating for x in self.ratings[userA.id]]
                    rateB = [x - userB.avg_rating for x in self.ratings[userB.id]]
                    pcs[userA.id][userB.id] = np.dot(rateA, rateB)
                    # including age bias
                    pcs[userA.id][userB.id] *= float(agerange - abs(userA.age - userB.age)) / agerange
                    # including gender bias
                    if userA.sex == userB.sex:
                        pcs[userA.id][userB.id] *= 1.1
                    else:
                        pcs[userA.id][userB.id] *= 0.9
        self.pcs = pcs

    def guess(self,user,movie,top_n):
        gid = self.movie_cluster[movie]
        pearson = self.pcs[user]
        # agerange = max(abs(self.users[user].age - 7), abs(self.users[user].age - 73))
        # for i in range(len(pearson)):
        #     pearson[i] *= float(agerange - abs(self.users[user].age - self.users[i].age)) / agerange
        #     if self.users[i].sex == self.users[user].sex:
        #         pearson[i] *= 1.1
        #     else:
        #         pearson[i] *= 0.9
        top_similar = np.argsort(pearson)[-top_n:]
        s, c = 0, 0
        for t in top_similar:
            if self.normRating[t][gid] != 0:
                s += self.normRating[t][gid] * pearson[t]
                c += pearson[t]

        rate = self.users[user].avg_rating + float(s) / c
        if rate < 1.0:
            return 1.0
        elif rate > 5.0:
            return 5.0
        else:
            return rate

    def get_rmse(self):
        error = 0.00
        cnt = 0
        err1, err2, c1, c2 = 0, 0, 0, 0

        for user in self.users:
            uid = user.id
            for mov in self.movies:
                mid = mov.id
                if self.test[uid][mid] != 0:
                    pred1 = self.guess(uid, mid, 150)
                    pred2 = self.guess2(uid, mid)
                    pred = 0.85 * pred1 + 0.15 * pred2
                    error += (pred - self.test[uid][mid]) ** 2
                    cnt += 1
                    err1 += (pred1 - self.test[uid][mid]) ** 2
                    err2 += (pred2 - self.test[uid][mid]) ** 2
                    c1 += 1
                    c2 += 1

        print "RMSE=", (float(error) / cnt) ** 0.5
        print "RMSE 1=", (float(err1) / c1) ** 0.5
        print "RMSE 2=", (float(err2) / c2) ** 0.5

    def decide_usergenre(self,top_n=30,genre_no = 3):
        for user in self.users:
            pg = [0]*len(self.genres)
            res = np.argsort(self.normRating[user.id])[-genre_no:]
            for r in res:
                pg[r] += 1
            pearson = self.pcs[user.id]
            # for i in range(len(pearson)):
            #     if self.users[i].sex != self.users[user.id].sex:
            #         pearson[i] *= 0.9
            top_similar = np.argsort(pearson)[-top_n:]
            for t in top_similar:
                res = np.argsort(self.normRating[t])[-genre_no:]
                for r in res:
                    pg[r] += 1

            user.pref_genre = np.argsort(pg)[-genre_no:]
            # user.pref_genre = np.argsort(self.normRating[user.id])[-genre_no:]

    def calculate_movieAvgRating(self):
        for m in self.movies:
            rating = 0
            cnt = 0
            for u in self.users:
                if self.train[u.id][m.id] != 0:
                    rating += self.train[u.id][m.id]
                    cnt += 1
            if cnt == 0:
                m.avg_rating = 0
            else:
                m.avg_rating = float(rating)/cnt

    def guess2(self,user,movie):
        up = self.users[user].pref_genre
        gen = self.movies[movie].genre
        mg = []
        for i in range(len(self.genres)):
            if gen[self.genres[i]] != 0:
                mg.append(i)
        # for g in self.genres:
        #     if gen[g] == 1:
        #         mg.append(gen[g])

        rating = 0
        for k in up:
            for j in mg:
                rating += self.genre_corr[k][j]
        rating *= self.movies[movie].avg_rating
        rate = float(rating)/len(up)
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
        self.decide_usergenre()
        self.calculate_movieAvgRating()
        self.data.genre_correlation()
        # self.genre_corr = self.data.genre_corr
        self.genre_corr = self.data.genre_corr_wordnet()
        self.get_rmse()

obj = MoviePredict()
obj.load_allData()
