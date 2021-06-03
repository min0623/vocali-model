import csv
import pandas as pd
import numpy as np
import os
import random
import pickle
import keras
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from surprise import Reader, Dataset
from surprise import BaselineOnly
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from starlette_context import context, plugins
from starlette_context.middleware import ContextMiddleware

class UserInfo(BaseModel):
  prefWeight: Optional[float] = 0.5
  moodWeight: Optional[float] = 0.5
  pitchWeight: Optional[float] = 0.5
  likeList: Optional[List[str]] = []
  dislikeList: Optional[List[str]] = []
  undefinedList: Optional[List[str]] = []
  minPitch: Optional[str] = ''
  maxPitch: Optional[str] = ''
  moods: Optional[List[str]] = []

app = FastAPI()

moods = ['happy', 'energetic', 'depression', 'calm']
int2pitch = {0: ["C"],1: ["C#"],2: ["D"],3: ["D#"],4: ["E"],5: ["F"],
             6: ["F#"],7: ["G"],8: ["G#"],9: ["A"],10: ["A#"],11: ["B"]}
pitch2int = {"C": 0,"B#": 0,"C#":1,"D": 2,"D#": 3,"E": 4,"F": 5,
             "E#": 5,"F#": 6,"G": 7,"G#": 8,"A": 9,"A#": 10,"B": 11}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ContextMiddleware)

# Gets the list of all track ids and names
def get_final_track_list(filename):
    with open(filename, 'r', encoding='utf-8') as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      final_tracks_ids = []
      final_tracks_names = []
      line_count = 0
      
      for row in csv_reader:
        if (line_count > 0):
          final_tracks_ids.append(row[6])
          final_tracks_names.append(row[1])
        line_count += 1

    return final_tracks_ids, final_tracks_names

# Loads the dataset of user-item data
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        items = []
        users = []
        final_tracks_ids, final_tracks_names = get_final_track_list("./songListWithFeatures.csv")
        tracks = []
        ratings = []

        for row in csv_reader:
            if (line_count == 2):
                users = row[3:-2]
            if (line_count > 2):
                if (row[-1] in final_tracks_ids):
                    tracks.append(row[-1])
                    ratings.append(row[3:-2])
            line_count += 1

        for track_id in range(len(tracks)):
            for user_id in range(len(ratings[track_id])):
                if (ratings[track_id][user_id] == 'TRUE'):
                    item = [user_id, track_id, 1]
                else:
                    item = [user_id, track_id, 0.5]
                items.append(item)

        context.update(final_tracks_ids=final_tracks_ids)
        
        return users, tracks, items

# Getting the user-item sparse matrix
def get_user_item_sparse_matrix(data):
    sparse_data = sparse.csr_matrix((data.rating, (data.user, data.track)))
    return sparse_data

# Calculatest the average rating of a user
def get_average_rating(sparse_matrix, is_user):
    ax = 1 if is_user else 0
    sum_of_ratings = sparse_matrix.sum(axis = ax).A1  
    no_of_ratings = sparse_matrix.sum(axis = ax).A1 
    rows, cols = sparse_matrix.shape
    average_ratings = {i: sum_of_ratings[i]/no_of_ratings[i] for i in range(rows if is_user else cols) if no_of_ratings[i] != 0}
    return average_ratings

# Calculates the similarity between users
def compute_user_similarity(sparse_matrix):
    row_index, col_index = sparse_matrix.nonzero()
    rows = np.unique(row_index)
    similar_arr = np.zeros(len(rows) * len(rows)).reshape(len(rows), len(rows))

    for row in rows:
        sim = cosine_similarity(sparse_matrix.getrow(row), context['train_sparse_data']).ravel()
        similar_indices = sim.argsort()
        similar = sim[similar_indices]
        similar_arr[row] = similar
    
    return similar_arr

# Calculates the similarity between tracks
def compute_track_similarity(sparse_matrix, track_id):
    similarity = cosine_similarity(sparse_matrix.T, dense_output=False)
    similar_tracks = context['tracks'][track_id], similarity[track_id]
    return similar_tracks

# Calculates the similar features of top 10 user and tracks for initial dataset
def create_new_similar_features(sample_sparse_matrix):
    global_avg_rating = get_average_rating(sample_sparse_matrix, False)
    global_avg_users = get_average_rating(sample_sparse_matrix, True)
    global_avg_tracks = get_average_rating(sample_sparse_matrix, False)
    sample_train_users, sample_train_tracks, sample_train_ratings = sparse.find(sample_sparse_matrix)
    new_features_csv_file = open("./new_features.csv", mode = "w")
    
    for user, track, rating in zip(sample_train_users, sample_train_tracks, sample_train_ratings):
        similar_arr = list()
        similar_arr.append(user)
        similar_arr.append(track)
        #similar_arr.append(sample_sparse_matrix.sum()/sample_sparse_matrix.count_nonzero())
        
        similar_users = cosine_similarity(sample_sparse_matrix[user], sample_sparse_matrix).ravel()
        indices = np.argsort(-similar_users)[1:]
        ratings = sample_sparse_matrix[indices, track].toarray().ravel()
        top_similar_user_ratings = list(ratings[:5])
        top_similar_user_ratings.extend([global_avg_rating[track]] * (5-len(ratings)))
        similar_arr.extend(top_similar_user_ratings)
        
        similar_tracks = cosine_similarity(sample_sparse_matrix[:,track].T, sample_sparse_matrix.T).ravel()
        similar_tracks_indices = np.argsort(-similar_tracks)[1:]
        similar_tracks_ratings = sample_sparse_matrix[user, similar_tracks_indices].toarray().ravel()
        top_similar_track_ratings = list(similar_tracks_ratings[:5])
        top_similar_track_ratings.extend([global_avg_users[user]] * (5-len(top_similar_track_ratings)))
        similar_arr.extend(top_similar_track_ratings)
        
        #similar_arr.append(global_avg_users[user])
        #similar_arr.append(global_avg_tracks[track])
        similar_arr.append(rating)
        
        new_features_csv_file.write(",".join(map(str, similar_arr)))
        new_features_csv_file.write("\n")
        
    new_features_csv_file.close()
    new_features_df = pd.read_csv('./new_features.csv', names = ["user_id", "track_id", "similar_user_rating1", 
                                                               "similar_user_rating2", "similar_user_rating3", "similar_user_rating4", "similar_user_rating5",
                                                               "similar_track_rating1", "similar_track_rating2", 
                                                               "similar_track_rating3", "similar_track_rating4", "similar_track_rating5",
                                                               "rating"])
    
    return new_features_df

# Calculates the similar features of top 10 user and tracks
def create_new_similar_features_for_new_user(sample_sparse_matrix, new_user):
    global_avg_rating = get_average_rating(sample_sparse_matrix, False)
    global_avg_users = get_average_rating(sample_sparse_matrix, True)
    global_avg_tracks = get_average_rating(sample_sparse_matrix, False)
    sample_train_users, sample_train_tracks, sample_train_ratings = sparse.find(sample_sparse_matrix)
    new_features_csv_file = open("./new_features_for_user.csv", mode = "w")
    
    for user, track, rating in zip(sample_train_users, sample_train_tracks, sample_train_ratings):
      if (user == new_user):
        similar_arr = list()
        similar_arr.append(user)
        similar_arr.append(track)
        #similar_arr.append(sample_sparse_matrix.sum()/sample_sparse_matrix.count_nonzero())
        
        similar_users = cosine_similarity(sample_sparse_matrix[user], sample_sparse_matrix).ravel()
        indices = np.argsort(-similar_users)[1:]
        ratings = sample_sparse_matrix[indices, track].toarray().ravel()
        top_similar_user_ratings = list(ratings[:5])
        top_similar_user_ratings.extend([global_avg_rating[track]] * (5-len(ratings)))
        similar_arr.extend(top_similar_user_ratings)
        
        similar_tracks = cosine_similarity(sample_sparse_matrix[:,track].T, sample_sparse_matrix.T).ravel()
        similar_tracks_indices = np.argsort(-similar_tracks)[1:]
        similar_tracks_ratings = sample_sparse_matrix[user, similar_tracks_indices].toarray().ravel()
        top_similar_track_ratings = list(similar_tracks_ratings[:5])
        top_similar_track_ratings.extend([global_avg_users[user]] * (5-len(top_similar_track_ratings)))
        similar_arr.extend(top_similar_track_ratings)
        
        #similar_arr.append(global_avg_users[user])
        #similar_arr.append(global_avg_tracks[track])
        similar_arr.append(rating)
        
        new_features_csv_file.write(",".join(map(str, similar_arr)))
        new_features_csv_file.write("\n")
        
    new_features_csv_file.close()
    new_features_df = pd.read_csv('./new_features_for_user.csv', names = ["user_id", "track_id", "similar_user_rating1", 
                                                               "similar_user_rating2", "similar_user_rating3", "similar_user_rating4", "similar_user_rating5",
                                                               "similar_track_rating1", "similar_track_rating2", 
                                                               "similar_track_rating3", "similar_track_rating4", "similar_track_rating5",
                                                               "rating"])
    
    return new_features_df

# Calculates the error of the users
def error_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def send_output(newWeight, liked, disliked, undefined, minPitch, maxPitch, newMood):
    song_num = context['song_num']
    df_songs = context['df_songs']
    trained_features = context['trained_features']
    clf = context['clf']
    mood_lrs = context['mood_lrs']
    pitch2int = context['pitch2int']
    data = context['data']
    final_tracks_ids = context['final_tracks_ids']

    # Get input
    weight = newWeight
    user_likeList = liked
    user_dislikeList = disliked
    user_undefinedList = undefined
    user_minPitch = minPitch
    user_maxPitch = maxPitch
    user_mood = newMood[0]
    user_with = newMood[1]

    ## Mood

    # Initializing scores
    mood_score = [0] * song_num

    # Mood analysis for each songs with LR
    features = ['loudness','mode','speechiness','acousticness','instrumentalness',
                'liveness','valence','tempo']

    for i in range(song_num):
        row = df_songs.iloc[[i]]
        # print(str(i)+" : "+row['title'])
        song_features = row[features]
        score = 0
        if user_with == "alone":
            score += mood_lrs[user_mood].predict_proba(song_features)[0][1]
        elif user_with == "together":
            score += mood_lrs[user_mood].predict_proba(song_features)[0][1] * 0.6
            score += mood_lrs["energetic"].predict_proba(song_features)[0][1] * 0.2
            score += mood_lrs["happy"].predict_proba(song_features)[0][1] * 0.2
        mood_score[i] += score

    ## Song Preference
    # Get predictions for the new user for collective filtering
    new_items = []
    for i in range(len(final_tracks_ids)):
        if (final_tracks_ids[i] in user_likeList):
            new_items.append([50, i, 1])
        elif (final_tracks_ids[i] in user_dislikeList):
            new_items.append([50, i, -1])
        else:
            new_items.append([50, i, 0.5])

    new_data = pd.DataFrame(new_items, columns=["user", "track", "rating"])

    new_user_sparse_matrix = get_user_item_sparse_matrix(pd.concat([new_data, data]))
    new_features = create_new_similar_features_for_new_user(new_user_sparse_matrix, 50)
    new_final_features = pd.concat([new_features, trained_features])
    new_test = new_final_features.loc[new_final_features['user_id']==50].drop(["user_id", "track_id", "rating"], axis = 1)
    new_pred_test = clf.predict(new_test)
    preference_score = new_pred_test

    # Pitch Analysis
    pitch_score = [0] * song_num
    # Calculate user key
    user_minKey = 11 * int(user_minPitch[-1]) + pitch2int[user_minPitch[:-1]]
    user_maxKey = 11 * int(user_maxPitch[-1]) + pitch2int[user_maxPitch[:-1]]
    user_key = int((user_minKey + user_maxKey)/2) % 11
    for i in range(song_num):
        song_key = int(df_songs.iloc[i]['key'])
        pitch_score[i] = (12 - min(abs(user_key-song_key), 12 - abs(user_key-song_key))) / 24

    ## Score Calculation
    total_score = [0] * song_num
    for i in range(song_num):
        total_score[i] = weight[0] * preference_score[i] + weight[1] * mood_score[i] + weight[2] * pitch_score[i]
        # 이미 판단한 노래 점수 -1으로 만들기 (TODO: 아예 위에서 계산도 안하게 빼버리면 더 빠를수도?)
        id = df_songs.iloc[i]['id']
        if (id in user_likeList) or (id in user_dislikeList) or (id in user_undefinedList):
            total_score[i] = -1

    # Print Top 10 results
    df_scores = pd.DataFrame(total_score, columns = ['score'])
    top_scores = list(df_scores.sort_values('score', ascending = False).index)
    rec_list = df_songs.iloc[top_scores]
    rec_list = rec_list[~rec_list['id'].isin(user_likeList)]
    rec_list = rec_list[['title', 'artist', 'id']]

    top10_mood_score = []
    top10_preference_score = []
    top10_pitch_score = []
    top10_total_score = []

    top10_indices = list(rec_list.index.values)

    for idx in top10_indices:
        top10_mood_score.append(mood_score[idx])
        top10_preference_score.append(preference_score[idx])
        top10_pitch_score.append(pitch_score[idx])
        top10_total_score.append(total_score[idx])

    rec_list['mood_score'] = top10_mood_score
    rec_list['preference_score'] = top10_preference_score
    rec_list['pitch_score'] = top10_pitch_score
    rec_list['total_score'] = top10_total_score
    # print(rec_list)
    # print(tabulate(rec_list[['title','mood_score','pitch_score','preference_score','total_score']], 
                #  headers='keys', tablefmt='psql'))

    return rec_list

def init_model():
    context.update(mood_lrs={})
    users, tracks, items = load_data('./userData.csv')
    data = pd.DataFrame(items, columns=["user", "track", "rating"])
    context.update(tracks=tracks)

    reader = Reader(rating_scale=(0,1))
    train_data_mf = Dataset.load_from_df(data[['user', 'track', 'rating']], reader)
    trainset = train_data_mf.build_full_trainset()
    svd = SVD(n_factors=100, biased=True, random_state=15, verbose=True)
    svd.fit(trainset)

    # Getting predictions of train set with SVD
    train_preds = svd.test(trainset.build_testset())
    train_pred_mf = np.array([pred.est for pred in train_preds])

    # Splitting train and test data
    split_value = int(len(data) * 0.80)
    train_data = data[:split_value]
    test_data = data[split_value:]

    train_sparse_data = get_user_item_sparse_matrix(train_data)
    test_sparse_data = get_user_item_sparse_matrix(test_data)

    context.update(train_sparse_data = train_sparse_data)

    global_average_rating = train_sparse_data.sum()/train_sparse_data.count_nonzero()

    similar_user_matrix = compute_user_similarity(train_sparse_data)

    train_new_similar_features = create_new_similar_features(train_sparse_data)
    test_new_similar_features = create_new_similar_features(test_sparse_data)

    x_train = train_new_similar_features.drop(["user_id", "track_id", "rating"], axis = 1)
    x_test = test_new_similar_features.drop(["user_id", "track_id", "rating"], axis = 1)
    y_train = train_new_similar_features["rating"]
    y_test = test_new_similar_features["rating"]

    print(x_train, y_train)

    # XGB Model
    clf = xgb.XGBRegressor(n_estimators = 100, silent = False, n_jobs  = 21, random_state=15, objective='binary:logistic', learning_rate=0.05, num_round=200, max_depth=6)
    clf.fit(x_train, y_train, eval_metric = 'rmse')
    context.update(clf = clf)

    y_pred_test = clf.predict(x_test)
    rmse_test = error_metrics(y_test, y_pred_test)
    trained_sparse_matrix = get_user_item_sparse_matrix(data)
    context.update(trained_features = create_new_similar_features(trained_sparse_matrix))

    for mood in moods:
        pkl_filename = "LR_" + mood + ".pkl"
        with open(pkl_filename, 'rb') as file:
            mood_lrs = context['mood_lrs']
            mood_lrs[mood] = pickle.load(file)
            context.update(mood_lrs=mood_lrs)

    df_songs = pd.read_csv('songListWithFeatures.csv',index_col=['num'])
    context.update(song_num = len(df_songs))
    context.update(df_songs = df_songs)

    print("Init Model Finished")

@app.on_event("startup")
async def startup():
    # background_tasks.add_task(init_model)
    print("App Started")

@app.post('/recommendations')                # Just in case if you want to handle a GET request
def index(userInfo: UserInfo):
    userInfoDict = userInfo.dict()

    result = send_output(
        [userInfoDict['prefWeight'], userInfoDict['moodWeight'], userInfoDict['pitchWeight']],
        userInfoDict['likeList'],
        userInfoDict['dislikeList'],
        userInfoDict['undefinedList'],
        userInfoDict['minPitch'],
        userInfoDict['maxPitch'],
        userInfoDict['moods']
    )
    
    return result.to_dict('records')[:10]

@app.get('/')
def getInfo(background_tasks: BackgroundTasks):
    background_tasks.add_task(init_model)
    return "Hello World"
