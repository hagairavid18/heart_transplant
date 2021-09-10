import collections

import optuna
from scipy.stats import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc

from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv, check_y_survival


def clean_data(new_df):
    new_df = new_df[(new_df['ORGAN'] == 'HR')]
    new_df = new_df[(new_df['NUM_PREV_TX'] < 1)]
    new_df = new_df[(new_df['AGE'] >= 18)]

    new_df = new_df[new_df['TX_DATE'].notnull()]

    # remove irrelevant cols
    cols_to_drop = 'inutero pt_code wl_id_code trr_id_code donor_id perfused_prior perfusion_location perfused_by ' \
                   'total_perfusion_time lu_received lu2_received pretitera pretitera_date pretiterb pretiterb_date ' \
                   'txhrt txint txkid txliv txlng txpan txvca tx_procedur_ty status_tcr prvtxdif retxdate trtrej1y ' \
                   'prev_tx prev_tx_any age_group init_calc_las init_match_las end_calc_las end_match_las ' \
                   'calc_las_listdate tcr_cdc_growth_bmi tcr_cdc_growth_hgt tcr_cdc_growth_wgt init_priority ' \
                   'end_priority  titera titera_date titerb titerb_date '

    measured_after = ' cod cod_ostxt cod2 cod2_ostxt cod3 cod3_ostxt gstatus gtime lastfuno pstatus func_stat_trf'

    cols_to_drop_list = (cols_to_drop + measured_after).upper().split()

    # remove features measured after surgery
    cols_to_drop_list += ['NUM_PREV_TX', 'COMPOSITE_DEATH_DATE', 'SSDMF_DEATH_DATE', 'PX_STAT_DATE', 'LOS',
                          'PST_DIAL', 'GRF_STAT',
                          'GRF_FAIL_DATE', 'GRF_FAIL_CAUSE', 'DISCHARGE_DATE', 'INIT_DATE', 'END_STAT', 'PST_STROKE',
                          'WL_ORG', 'ORGAN']

    # remove date features
    cols_to_drop_list += ['TX_DATE', 'GRF_FAIL_CAUSE_OSTXT', 'REFERRAL_DATE', 'ADMISSION_DATE',
                          'END_DATE', 'ACTIVATE_DATE', 'TX_YEAR',
                          'RECOVERY_DATE_DON', 'ADMIT_DATE_DON', 'REM_CD']

    # removes features with one\zero values:
    cols_to_drop_list += [col for col in new_df.columns if len(new_df[col].value_counts()) <= 1]

    new_df = new_df.drop(cols_to_drop_list, axis=1)

    return new_df


def show_importances(model, df):
    importances = model.feature_importances_
    sorted_index = np.argsort(importances)[len(importances) - 30::]
    x = range(30)
    print(x)
    labels = np.array(df.columns)[sorted_index]
    plt.bar(x, importances[sorted_index], tick_label=labels)
    plt.xticks(rotation=90)
    print(plt.show())


def find_best_features(X, y, col_names, name_of_file):
    scores = np.empty(len(col_names))
    m = CoxPHSurvivalAnalysis()

    for j in range(len(col_names)):
        Xj = X[:, j:j + 1]
        try:
            m.fit(Xj, y)
            scores[j] = m.score(Xj, y)
        except:
            scores[j] = 0

    all_keys = pd.Series(scores, index=col_names).sort_values(ascending=False).keys().tolist()
    all_scores = pd.Series(scores, index=col_names).sort_values(ascending=False).values.tolist()
    with open(name_of_file, "w") as f:
        for i in range(len(all_keys)):
            f.write(str(all_keys[i]) + " " + str(all_scores[i]) + "\n")


def get_k_best(name_of_file, k, cols_names):
    best_features = []
    with open(name_of_file) as f:
        for line in f:
            if len(best_features) >= k:
                return best_features
            elif line.split()[0] in cols_names:
                best_features.append(line.split()[0])
    return best_features


def make_monitor(running_mean_len):
    def monitor(i, self, args):
        if np.mean(self.oob_improvement_[max(0, i - running_mean_len + 1):i + 1]) < 0:
            return True
        else:
            return False

    return monitor


def treat_mis_value_nu(df):
    # get only numeric columns to dataframe
    df_nu = df.select_dtypes(include=['float64', 'int64'])

    # get only columns with NaNs
    df_nu = df_nu.loc[:, df_nu.isnull().any()]

    # get columns for remove with at most 0.7 null
    cols_to_drop = df.columns[df.isnull().mean() >= 0.8]
    print(len(cols_to_drop))
    # replace missing values of original columns and remove above thresh
    df = df.fillna(df_nu.median()).drop(cols_to_drop, axis=1)
    return df


def treat_mis_value_obj(new_df):
    df_ob = new_df.select_dtypes(include=['object'])
    df_nu = df_ob.loc[:, df_ob.isnull().any()]
    for col in df_nu.columns:
        if new_df[col].value_counts().iloc[0] / new_df[col].value_counts().sum() * 100 > 70:
            new_df[col] = new_df[col].fillna(new_df[col].mode().iloc[0])
        else:
            new_df[col] = new_df[col].fillna("unknown")
    return new_df


def remove_outliers(X, y):
    # summarize the shape of the training dataset
    # print(X.shape, y.shape)
    # identify outliers in the training dataset
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X)
    # select all rows that are not outliers
    mask = yhat != -1
    X_train, y_train = X[mask, :], y[mask]
    # summarize the shape of the updated training dataset
    # print(X_train.shape, y_train.shape)
    return X_train, y_train


def label_encoder(new_df):
    # Multiple categorical columns

    categorical_cols = [col for col in new_df.columns if new_df[col].nunique() < 5 and new_df[col].dtype == 'object']
    new_df = pd.get_dummies(new_df, columns=categorical_cols, drop_first=True)
    columns_to_be_encoded = [col for col in new_df.columns if new_df[col].dtype == 'object']

    # Instantiate the encoders
    encoders = {column: LabelEncoder() for column in columns_to_be_encoded}

    for column in columns_to_be_encoded:
        new_df[column] = encoders[column].fit_transform(new_df[column].astype(str))
    return new_df


def objective(trial, X, y, df, features_names):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9, step=0.1),
        # "max_features": trial.suggest_categorical(
        #     "max_features", ["auto", "sqrt", "log2"]
        # ),
        "random_state": 1121218,
        "num_of_features": trial.suggest_int("num_of_features", 100, 250, step=10)
    }
    best_cols = get_k_best('to_remove_10.2018.txt', params["num_of_features"], features_names)
    index_no = [df.columns.get_loc(c) for c in best_cols if c in df]
    X = X[:, index_no]
    # Perform CV
    est_cph_tree = GradientBoostingSurvivalAnalysis(n_estimators=params['n_estimators'],
                                                    learning_rate=params['learning_rate'],
                                                    max_depth=params['max_depth'],
                                                    subsample=params['subsample'])

    scores = []
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in skf.split(X, y["event"]):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_test = X_test[y_test['time'] < y_train['time'].max(), :]
        y_test = y_test[y_test['time'] < y_train['time'].max()]
        X_train, y_train = remove_outliers(X_train, y_train)
        monitor = make_monitor(10)
        est_cph_tree.fit(X_train, y_train, monitor=monitor)
        va_times = np.arange(y_test['time'].min(), y_test['time'].max(), 7)

        rsf_chf_funcs = est_cph_tree.predict_cumulative_hazard_function(
            X_test)

        try:
            rsf_risk_scores = np.row_stack([chf(va_times) for chf in rsf_chf_funcs])
            est_auc, est_mean_auc = cumulative_dynamic_auc(
                y_train, y_test, rsf_risk_scores, va_times
            )
            print(est_cph_tree.score(X_train, y_train))
            print(est_cph_tree.score(X_test, y_test))
            scores.append(est_auc[52])
        except ValueError:
            print("raise error")

        print(scores)

    return np.median(scores)


def main():

    # get data:
    df = pd.read_csv('clean1.csv', low_memory=False)

    # TX_DATE is the day the person was transplanted.
    df = df[(df['TX_DATE'] > '2018-10-01')]  # & (df['TX_DATE'] < '2017-10-01')]
    df = clean_data(df)

    # in survival anlasys, we define for each person if it right censo
    df['event'] = np.where(df['PX_STAT'] == 'D', True, False)

    # treat missing values:
    df = treat_mis_value_obj(df)
    df = treat_mis_value_nu(df)
    print(df.shape)

    y = Surv.from_arrays(df.event, df.PTIME)

    df = df.drop(['PX_STAT', 'PTIME', 'event'], axis=1)

    # label encoder
    df = label_encoder(df)

    X = df.to_numpy()

    # Scaled feature
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=20, stratify=y["event"], shuffle=True)

    X_test = X_test[y_test['time'] < y_train['time'].max(), :]
    y_test = y_test[y_test['time'] < y_train['time'].max()]

    # Applying Transformer
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # find_best_features(X_train, y_train, df.columns, 'to_remove_2018-today-today.txt')

    print(X_train.shape)

    study = optuna.create_study(direction="maximize")
    func = lambda trial: objective(trial, X_train, y_train, df, df.columns)

    # Start optimizing with 100 trials
    study.optimize(func, n_trials=50)

    print("Best params:")

    # for key, value in study.best_params.items():
    #     print(f"\t{key}: {value}")

    best_cols = get_k_best('to_remove_10.2018.txt', study.best_params['num_of_features'], df.columns)
    # best_cols = get_k_best('to_remove_10.2018.txt', 140, df.columns)
    # print(best_cols)
    index_no = [df.columns.get_loc(c) for c in best_cols if c in df]
    X_train = X_train[:, index_no]
    X_test = X_test[:, index_no]
    X_train, y_train = remove_outliers(X_train, y_train)

    est_cph_tree = GradientBoostingSurvivalAnalysis(n_estimators=study.best_params['n_estimators'],
                                                    max_depth=study.best_params['max_depth'],
                                                    learning_rate=study.best_params['learning_rate'],
                                                    subsample=study.best_params['subsample'])
    # est_cph_tree = GradientBoostingSurvivalAnalysis(n_estimators=900, learning_rate=0.0016543559723740892, max_depth=5,
    #                                                 subsample=0.7)
    monitor = make_monitor(10)
    est_cph_tree.fit(X_train, y_train, monitor=monitor)
    # show_importances(est_cph_tree,df)

    va_times = np.arange(y_test['time'].min(), y_test['time'].max(), 7)

    rsf_chf_funcs = est_cph_tree.predict_cumulative_hazard_function(
        X_test)

    rsf_risk_scores = np.row_stack([chf(va_times) for chf in rsf_chf_funcs])
    est_auc, est_mean_auc = cumulative_dynamic_auc(
        y_train, y_test, rsf_risk_scores, va_times
    )
    plt.plot(va_times, est_auc, marker="o")
    plt.axhline(est_mean_auc, linestyle="--")
    plt.xlabel("days from enrollment")
    plt.ylabel("time-dependent AUC")
    plt.grid(True)

    print(plt.show())
    print(est_auc[52])
    print(est_auc[:52])
    print(est_cph_tree.score(X_train, y_train))
    print(est_cph_tree.score(X_test, y_test))


if __name__ == "__main__":
    main()
