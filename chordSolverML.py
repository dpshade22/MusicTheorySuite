from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from joblib import dump, load
import streamlit as st
import pandas as pd
import numpy as np
from streamlit.proto.Radio_pb2 import Radio
from PIL import Image

# st.set_page_config(layout="wide")


dtreeModel = DecisionTreeClassifier()
joblib.dump(dtreeModel, "dtreeModel.joblib")
# Build pandas dataframe
@st.cache
def loadData(vars):
    if vars == "Chord":
        df = pd.read_csv("./DataAndModels/data10000", usecols=["Key", "Notes", "Chord"])
    if vars == "Roman":
        df = pd.read_csv(
            "./DataAndModels/data10000",
            usecols=["Key", "Notes", "RomanNumeral"],
        )
    if vars == "Key":
        df = pd.read_csv(
            "./DataAndModels/data10000",
            usecols=["Key", "Notes", "RomanNumeral"],
        )
    if vars == "Notes":
        df = pd.read_csv(
            "./DataAndModels/data10000",
            usecols=["Key", "Notes", "RomanNumeral"],
        )
    if vars == "ChordNotes":
        df = pd.read_csv(
            "./DataAndModels/data10000",
            usecols=["Chord", "Notes"],
        )
    if vars == "Quality":
        df = pd.read_csv(
            "./DataAndModels/data10000",
            usecols=["Notes", "Quality"],
        )

    return df


# df.head()
# Scikit Learn Classification
ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
# model = neighbors.KNeighborsClassifier(15)
chordModel = load("./DataAndModels/dtreeModel.joblib")
romanModel = load("./DataAndModels/dtreeModel.joblib")
keyModel = load("./DataAndModels/dtreeModel.joblib")
notesModel = load("./DataAndModels/dtreeModel.joblib")
chordNotesModel = load("./DataAndModels/dtreeModel.joblib")
qualityModel = load("./DataAndModels/dtreeModel.joblib")
## Column Transformation and Model


def getCategoircalDataTransformed(columns=[]):
    column_trans = make_column_transformer(
        (ohe, columns),
        remainder="passthrough",
    )
    return column_trans


## Declare Supervised Training Set
## Declare Training and Testing Sets
## Establish Pipeline to Run Encoding and Model Simultaneously

# X_train.shape, X_test.shape


## Fit Data to Model
## Accuracy of Model on X Training Set
## Create Prediction Dataframe
# X_train[:20]
# prediction[:20]
## Predict Random Values
@st.cache
def chordPredictionDF():
    df = loadData("Chord")
    X = df.drop(columns=["Chord"])
    y = df["Chord"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=4, shuffle=True
    )

    pipe = make_pipeline(
        getCategoircalDataTransformed(["Key", "Notes"]),
        chordModel,
    )

    pipe.fit(X_train, y_train)

    score = pipe.score(X_test, y_test)

    return pipe, score


@st.cache
def romanPredictionDF():
    df = loadData("Roman")
    X = df.drop(columns=["RomanNumeral"])
    y = df["RomanNumeral"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=4, shuffle=True
    )

    pipe = make_pipeline(
        getCategoircalDataTransformed(["Key", "Notes"]),
        romanModel,
    )

    pipe.fit(X_train, y_train)

    score = pipe.score(X_test, y_test)

    return pipe, score


@st.cache
def keyPredictionDF():
    df = loadData("Key")
    X = df.drop(columns=["Key"])
    y = df["Key"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=4, shuffle=True
    )

    pipe = make_pipeline(
        getCategoircalDataTransformed(["Notes", "RomanNumeral"]),
        keyModel,
    )

    pipe.fit(X_train, y_train)

    score = pipe.score(X_test, y_test)

    return pipe, score


@st.cache
def notesPredictionDF():
    df = loadData("Notes")
    X = df.drop(columns=["Notes"])
    y = df["Notes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=4, shuffle=True
    )

    pipe = make_pipeline(
        getCategoircalDataTransformed(["Key", "RomanNumeral"]),
        notesModel,
    )

    pipe.fit(X_train, y_train)

    score = pipe.score(X_test, y_test)

    return pipe, score
