import pandas as pd
import numpy as np
import glob
from sklearn.neural_network import MLPRegressor
from zipfile import ZipFile


def load_training_data():
    path = '../CHEM_trainingdata'
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame


def load_submission_data():
    df = pd.read_csv('submission_template.csv')
    return df


def get_t4_predictor_trainingset(df):
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

        system = int(t1_df['System'].values[0].split('_')[1])
        system_oh = np.zeros(12)
        system_oh[system - 1] = 1
        x_vector += list(system_oh)
        for i in range(1, 9):
            column_name = f'U{i}'
            x_vector.append(t1_df[column_name].values[0])

        x_list.append(x_vector)
        y_list.append(y_vector)

    x_array = np.asarray(x_list)
    y_array = np.asarray(y_list)
    permutation = np.random.permutation(x_array.shape[0])
    x_array = x_array[permutation, :]
    y_array = y_array[permutation, :]
    return x_array, y_array


def get_t18_t19_t20_predictor_trainingset(df):
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
        system = int(t4_df['System'].values[0].split('_')[1])
        system_oh = np.zeros(12)
        system_oh[system - 1] = 1
        x_vector += list(system_oh)

        y_vector.append(t18_df['Y'].values[0])
        y_vector.append(t19_df['Y'].values[0])
        y_vector.append(t20_df['Y'].values[0])

        x_list.append(x_vector)
        y_list.append(y_vector)

    x_array = np.asarray(x_list)
    y_array = np.asarray(y_list)
    permutation = np.random.permutation(x_array.shape[0])
    x_array = x_array[permutation, :]
    y_array = y_array[permutation, :]
    return x_array, y_array


def train_regressor(x, y, params):
    regressor = MLPRegressor(max_iter=50000, batch_size=params["batch_size"],
                             hidden_layer_sizes=params["hidden_layer_sizes"], learning_rate=params["learning_rate"],
                             learning_rate_init=params["learning_rate_init"], alpha=params["alpha"])
    regressor.fit(x, y)
    return regressor


def submission_data_to_list_of_vectors(df):
    x_list = []
    for index, row in df.iterrows():
        x_vector = []
        for i in range(1, 15):
            column_name = f'X{i}'
            x_vector.append(row[column_name])
        x_vector.append(row['Y'])
        system = int(row['System'].split('_')[1])
        system_oh = np.zeros(12)
        system_oh[system - 1] = 1
        x_vector += list(system_oh)
        x_list.append(np.asarray(x_vector))
    return x_list


def submission_data_to_list_of_targets(df):
    target_list = []
    for index, row in df.iterrows():
        target_list.append(row['target'])
    return target_list


def get_prediction_for_target_values(initial_data, u_vectors, regressor1, regressor2):
    initial_data = np.repeat(initial_data[np.newaxis, :], u_vectors.shape[0], axis=0)
    input_vector1 = np.concatenate([initial_data, u_vectors], axis=1)
    prediction1 = regressor1.predict(input_vector1)
    system_number = initial_data[:, -12:]
    input_vector2 = np.concatenate([prediction1, system_number], axis=1)
    prediction2 = regressor2.predict(input_vector2)
    return prediction2


def find_u_vector_with_lowest_loss_genetic(initial_data, regressor1, regressor2, target, children_per_generation=2500):
    best = np.zeros(8)
    u_vector = best
    scales = [2, 0.5, 0.01]
    prediction = get_prediction_for_target_values(initial_data, u_vector[np.newaxis, :], regressor1,
                                                           regressor2)
    best_loss = approximation_loss(prediction, target, u_vector)
    for i in range(3):
        children = np.asarray([get_child(best, scale=scales[i]) for _ in range(children_per_generation)])
        predictions = get_prediction_for_target_values(initial_data, children, regressor1, regressor2)
        losses = [approximation_loss(predictions[j], target, children[j]) for j in range(children_per_generation)]

        best_index = np.argmin(losses)
        best_child_of_generation = children[best_index]

        if approximation_loss(predictions[best_index], target, best_child_of_generation) < best_loss:
            best_loss = approximation_loss(predictions[best_index], target, best_child_of_generation)
            best = best_child_of_generation
    return best


def get_child(best, scale=1):
    child = best.copy()
    child += np.random.normal(0, scale, size=8)
    return np.clip(child, -10, 10)


def approximation_loss(prediction, target, u_vector):
    c = 1/20
    u_norm_squared = np.linalg.norm(u_vector) ** 2
    t = np.asarray([50.002546, 63.27446, 80])
    squared_error = (prediction - target) ** 2
    integral_approximation = np.trapz(squared_error, x=t)
    loss = c * (u_norm_squared / 8) + np.sqrt((1 / 40) * integral_approximation)
    return loss


def make_submission(training_df, submission_df, params):
    x, y = get_t4_predictor_trainingset(training_df)
    regressor1 = train_regressor(x, y, params)
    x, y = get_t18_t19_t20_predictor_trainingset(training_df)
    regressor2 = train_regressor(x, y, params)

    submission_list = submission_data_to_list_of_vectors(submission_df)
    target_list = submission_data_to_list_of_targets(submission_df)

    for i, row in submission_df.iterrows():
        best = find_u_vector_with_lowest_loss_genetic(submission_list[i], regressor1, regressor2,
                                                               target_list[i])
        for j in range(1, 9):
            submission_df.loc[i, f'U{j}'] = best[j - 1]

    submission_df.to_csv("results/submission.csv")
    ZipFile(f'results/champion_oh{params}.zip', mode='w').write('results/submission.csv')


if __name__ == '__main__':
    params = {}
    params['hidden_layer_sizes'] = (8192, 8192)
    params["learning_rate_init"] = 0.0002
    params["learning_rate"] = 'constant'
    params["alpha"] = 0
    params["batch_size"] = 64

    training_df = load_training_data()
    submission_df = load_submission_data()
    make_submission(training_df, submission_df, params)
