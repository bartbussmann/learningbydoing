# My first main idea:
# Two estimators: one that estimates Xtarget from t4 and one that estimate t4 from t3 and U.

import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import hyperopt
from zipfile import ZipFile
import multiprocessing

import time


def load_all_training_data():
    path = '../CHEM_trainingdata'
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame


def get_X4_predictor_trainingset(df):
    instance_ids = df["ID"].unique()
    x_list = []
    y_list = []
    for id in instance_ids:
        x_vector = []
        y_vector = []
        instance_df = df.loc[df['ID'] == id]
        t1_df = instance_df.loc[df['t'] == 0]
        t4_df = instance_df.loc[df['t'] == 1.001427]

        for i in range(1, 15):
            column_name = f'X{i}'
            x_vector.append(t1_df[column_name].values[0])
            y_vector.append(t4_df[column_name].values[0])
        x_vector.append(t1_df['Y'].values[0])
        y_vector.append(t4_df['Y'].values[0])

        x_vector.append(int(t1_df['System'].values[0].split('_')[1]))
        for i in range(1, 9):
            column_name = f'U{i}'
            x_vector.append(t1_df[column_name].values[0])


        x_list.append(x_vector)
        y_list.append(y_vector)
    return np.asarray(x_list), np.asarray(y_list)

def get_X18_X19_X20_predictor_trainingset(df):
    instance_ids = df["ID"].unique()
    x_list = []
    y_list = []
    for id in instance_ids:
        x_vector = []
        y_vector = []
        instance_df = df.loc[df['ID'] == id]
        t4_df = instance_df.loc[df['t'] == 1.001427]
        t18_df = instance_df.loc[df['t'] == 50.002546]
        t19_df = instance_df.loc[df['t'] == 63.27446]
        t20_df = instance_df.loc[df['t'] == 80]


        for i in range(1, 15):
            column_name = f'X{i}'
            x_vector.append(t4_df[column_name].values[0])
        x_vector.append(t4_df['Y'].values[0])
        x_vector.append(int(t4_df['System'].values[0].split('_')[1]))

        y_vector.append(t18_df['Y'].values[0])
        y_vector.append(t19_df['Y'].values[0])
        y_vector.append(t20_df['Y'].values[0])


        x_list.append(x_vector)
        y_list.append(y_vector)
    return np.asarray(x_list), np.asarray(y_list)

def train_random_forest(x, y):
    # regressor = RandomForestRegressor(n_estimators=200)
    regressor = MLPRegressor(max_iter=50000)
    regressor.fit(x, y)
    return regressor

def get_train_val_split(x, y, percentage=0.10):
    permutation = np.random.permutation(x.shape[0])
    val_samples = int(x.shape[0]*percentage)
    x_shuffled = x[permutation, :]
    y_shuffled = y[permutation, :]
    x_train = x_shuffled[val_samples:, :]
    x_val = x_shuffled[:val_samples, :]
    y_train = y_shuffled[val_samples:, :]
    y_val = y_shuffled[:val_samples, :]
    return x_train, x_val, y_train, y_val

def load_submission_data():
    df = pd.read_csv('submission_template.csv')
    return df

def submission_data_to_list_of_vectors(df):
    x_list = []
    for index, row in df.iterrows():
        x_vector = []
        for i in range(1, 15):
            column_name = f'X{i}'
            x_vector.append(row[column_name])
        x_vector.append(row['Y'])
        x_vector.append(int(row['System'].split('_')[1]))
        x_list.append(np.asarray(x_vector))
    return x_list

def submission_data_to_list_of_targets(df):
    target_list = []
    for index, row in df.iterrows():
        target_list.append(row['target'])
    return target_list

def get_prediction_for_target_values(initial_data, u_vector, regressor1, regressor2):
    input_vector1 = np.concatenate([initial_data, u_vector])
    prediction1 = regressor1.predict(input_vector1[np.newaxis, :])
    input_vector2 = np.concatenate([prediction1.flatten(), np.asarray([initial_data[-1]])])
    prediction2 = regressor2.predict(input_vector2[np.newaxis, :])
    return prediction2[0]

def find_u_vector_with_lowest_loss(initial_data, regressor1, regressor2, target):
    space = {}
    space['u1'] = hyperopt.hp.uniform('u1', -10, 10)
    space['u2'] = hyperopt.hp.uniform('u2', -10, 10)
    space['u3'] = hyperopt.hp.uniform('u3', -10, 10)
    space['u4'] = hyperopt.hp.uniform('u4', -10, 10)
    space['u5'] = hyperopt.hp.uniform('u5', -10, 10)
    space['u6'] = hyperopt.hp.uniform('u6', -10, 10)
    space['u7'] = hyperopt.hp.uniform('u7', -10, 10)
    space['u8'] = hyperopt.hp.uniform('u8', -10, 10)
    space['initial_data'] = initial_data
    space['regressor1'] = regressor1
    space['regressor2'] = regressor2
    space['target'] = target
    best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=50)
    best['initial_data'] = initial_data
    best['regressor1'] = regressor1
    best['regressor2'] = regressor2
    best['target'] = target

    loss1 = objective(best)
    space['u1'] = space['u2'] = space['u3'] = space['u4'] = space['u5'] = space['u6'] = space['u7'] = space['u8'] = 0
    loss2 = objective(space)
    print(loss2)
    if loss1 < loss2:
        return best
    else:
        return space


def find_u_vector_with_lowest_loss_genetic(initial_data, regressor1, regressor2, target, children_per_generation=5000):
    t = time.time()
    best = np.zeros(8)
    u_vector = best
    prediction = get_prediction_for_target_values(initial_data, u_vector, regressor1, regressor2)
    best_loss = approximation_loss(prediction, target, u_vector)
    for i in range(3):
        children = [get_child(best) for j in range(children_per_generation)]
        list_of_inputs = [(initial_data, children[j], regressor1, regressor2) for j in range(children_per_generation)]
        with multiprocessing.Pool(processes=8) as pool:
            predictions = pool.starmap(get_prediction_for_target_values, list_of_inputs)

        losses = [approximation_loss(predictions[j], target, children[j]) for j in range(children_per_generation)]
        best_index = np.argmin(losses)
        best_child_of_generation = children[best_index]
        if approximation_loss(predictions[best_index], target, best_child_of_generation) < best_loss:
            best_loss = approximation_loss(prediction, target, best_child_of_generation)
            best = best_child_of_generation
    print('time', time.time() - t)
    return best

def get_child(best):
    child = best.copy()
    bit_to_change = np.random.randint(8)
    child += np.random.normal(size=8)
    return child

def objective(space):
    initial_data = space['initial_data']
    regressor1 = space['regressor1']
    regressor2 = space['regressor2']
    target = space['target']
    u_vector = np.asarray([space['u1'], space['u2'], space['u3'], space['u4'], space['u5'], space['u6'], space['u7'], space['u8']])
    prediction = get_prediction_for_target_values(initial_data, u_vector, regressor1, regressor2)
    return approximation_loss(prediction, target, u_vector)


def simple_loss(prediction, target, u_vector):
    return (prediction[0] - target)**2 + (prediction[1] - target)**2 + (prediction[2] - target)**2

def approximation_loss(prediction, target, u_vector):
    c = 1/20
    u_norm_squared = np.linalg.norm(u_vector)**2
    t = np.asarray([50.002546, 63.27446, 80])
    squared_error = (prediction - target)**2
    integral_approximation = np.trapz(squared_error, x=t)
    loss = c*(u_norm_squared/8) + np.sqrt((1/40)*integral_approximation)
    return loss

def make_submission(training_df, submission_df):
    submission_list = submission_data_to_list_of_vectors(submission_df)
    target_list = submission_data_to_list_of_targets(submission_df)

    x, y = get_X4_predictor_trainingset(training_df)
    x_train, x_val, y_train, y_val = get_train_val_split(x, y, percentage=0.10)
    regressor1 = train_random_forest(x, y)
    mse = np.mean((y_train - regressor1.predict(x_train)) ** 2)
    mse_val = np.mean((y_val - regressor1.predict(x_val)) ** 2)
    # assert(mse_val < 0.10)
    print(mse, mse_val)

    x, y = get_X18_X19_X20_predictor_trainingset(training_df)
    x_train, x_val, y_train, y_val = get_train_val_split(x, y, percentage=0.10)
    regressor2 = train_random_forest(x, y)
    mse = np.mean((y_train - regressor2.predict(x_train)) ** 2)
    mse_val = np.mean((y_val - regressor2.predict(x_val)) ** 2)
    # assert(mse_val < 0.20)
    print(mse, mse_val)

    for i, row in submission_df.iterrows():
        print(i)
        best = find_u_vector_with_lowest_loss_genetic(submission_list[i], regressor1, regressor2, target_list[i])
        print(best)
        for j in range(1, 9):
            submission_df.loc[i, f'U{j}'] = best[j-1]

    submission_df.to_csv("submission.csv")
    ZipFile('submission.zip', mode='w').write('submission.csv')

def plot_all_timeseries(df):
    instance_ids = df["ID"].unique()
    for id in instance_ids:
        instance_df = df.loc[df['ID'] == id]
        for i in range(1, 15):
            plt.plot(instance_df["t"], instance_df[f"X{i}"])
        plt.show()

np.random.seed(1)
training_df = load_all_training_data()
# plot_all_timeseries(training_df)

submission_df = load_submission_data()
make_submission(training_df, submission_df)


# x, y = get_X4_predictor_trainingset(df)
# x_train, x_val, y_train, y_val = get_train_val_split(x, y, percentage=0.10)
# regressor1 = train_random_forest(x, y)
# mse = np.mean((y_train - regressor1.predict(x_train))**2)
# mse_val = np.mean((y_val - regressor1.predict(x_val))**2)
# print(mse, mse_val)
#
# x, y = get_X18_X19_X20_predictor_trainingset(df)
# x_train, x_val, y_train, y_val = get_train_val_split(x, y, percentage=0.10)
# regressor2 = train_random_forest(x, y)
# mse = np.mean((y_train - regressor2.predict(x_train))**2)
# mse_val = np.mean((y_val - regressor2.predict(x_val))**2)
# print(mse, mse_val)
#
# prediction = get_prediction_for_target_values(submission_list[0], np.zeros(8), regressor1, regressor2)
# print(prediction)
# target_list = submission_data_to_list_of_targets(submission_df)
# best = find_u_vector_with_lowest_loss(submission_list[0], regressor1, regressor2, target_list[0])
# print(best)