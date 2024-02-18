import numpy as np
# from attention_forest.model import *
from attention_forest import TaskType, ForestKind, EAFParams, FWAFParams, EpsAttentionForest, FeatureWeightedAttentionForest
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_friedman1
import pandas as pd
from sklearn.preprocessing import StandardScaler

def run_example():
    print("Attention Forest:")
    #X, y = make_friedman1(random_state=12345)
    df = pd.read_csv("seismic-bumps.csv")
    cleanup_nums = {"seismic": {"a": 0, "b": 1},
                    "seismoacoustic": {"a": 0, "b": 1, "c": 2},
                    "hazard": {"a": 0, "b": 1, "c": 2},
                    "shift": {"W": 1, "N": 0}}
    df = df.replace(cleanup_nums)
    df['seismic'] = df['seismic'].astype (float)
    df['seismoacoustic'] = df['seismoacoustic'].astype (float)
    df['shift'] = df['shift'].astype (float)
    df['genergy'] = df['genergy'].astype (float)
    df['gpuls'] = df['gpuls'].astype (float)
    df['gdenergy'] = df['gdenergy'].astype (float)
    df['gdpuls'] = df['gdpuls'].astype (float)
    df['hazard'] = df['hazard'].astype (float)
    df['energy'] = df['energy'].astype (float)
    df['maxenergy'] = df['maxenergy'].astype (float)
    df['nbumps'] = df['nbumps'].astype (float)
    df['nbumps2'] = df['nbumps2'].astype (float)
    df['nbumps3'] = df['nbumps3'].astype (float)
    df['nbumps4'] = df['nbumps4'].astype (float)
    df['nbumps5'] = df['nbumps5'].astype (float)
    df['nbumps6'] = df['nbumps6'].astype (float)
    df['nbumps7'] = df['nbumps7'].astype (float)
    df['nbumps89'] = df['nbumps89'].astype (float)
    df['class'] = df['class'].astype (float)

    scaler = StandardScaler()



    y = df["class"].values
    #print("y")
    #print(y)
    del df['class']
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    X = df.values
    #print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12345)

    task = TaskType.CLASSIFICATION
    # model = EpsAttentionForest(EAFParams(
    #     kind=ForestKind.RANDOM,
    #     task=task,
    #     forest=dict(
    #         n_estimators=100,
    #         min_samples_leaf=3,
    #         n_jobs=-1,
    #         random_state=12345,
    #     ),
    #     eps=0.5,
    #     tau=1.0,
    # ))

    model = FeatureWeightedAttentionForest(FWAFParams(
        kind=ForestKind.EXTRA,
        task=task,
        forest=dict(
            n_estimators=5,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=12345,
        ),
        no_temp=False,
    ))

    model.fit(X_train, y_train)
    if task == TaskType.REGRESSION:
        metric = r2_score
        forest_predict = lambda x: model.forest.predict(x)
    else:
        metric = lambda a, b: roc_auc_score(a, b[:, 1])
        forest_predict = lambda x: model.forest.predict_proba(x)

    print("Forest score (train):", metric(y_train, forest_predict(X_train)))
    print("Forest score (test):", metric(y_test, forest_predict(X_test)))

    print("Before opt score (train):", metric(y_train, model.predict_proba(X_train)))
    print("Before opt score (test):", metric(y_test, model.predict_proba(X_test)))

    model.optimize_weights_sgd(X_train, y_train)
    print("After opt score (train):", metric(y_train, model.predict_proba(X_train)))
    print("After opt score (test):", metric(y_test, model.predict_proba(X_test)))


if __name__ == "__main__":
    run_example()