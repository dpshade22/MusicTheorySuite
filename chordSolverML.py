from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from joblib import dump, load
import streamlit as st
import pandas as pd
import numpy as np
from streamlit.proto.Radio_pb2 import Radio


# Build pandas dataframe
@st.cache
def loadData(vars):
    if vars == "Chord":
        df = pd.read_csv(
            "./DataAndModels/data1000", usecols=["Key", "Quality", "Notes", "Chord"]
        )
    if vars == "Roman":
        df = pd.read_csv(
            "./DataAndModels/data1000",
            usecols=["Key", "Quality", "Notes", "RomanNumeral"],
        )
    if vars == "Key":
        df = pd.read_csv(
            "./DataAndModels/data1000",
            usecols=["Key", "Notes", "RomanNumeral"],
        )
    if vars == "Notes":
        df = pd.read_csv(
            "./DataAndModels/data1000",
            usecols=["Chord", "Key", "Quality", "Notes", "RomanNumeral"],
        )

    return df


# df.head()
# Scikit Learn Classification
ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
# model = neighbors.KNeighborsClassifier(15)
chordModel = load("./DataAndModels/model.joblib")
romanModel = load("./DataAndModels/model.joblib")
keyModel = load("./DataAndModels/model.joblib")
notesModel = load("./DataAndModels/model.joblib")

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

    X_train, y_train, X_test, y_test = train_test_split(
        X, y, random_state=0, shuffle=True
    )

    pipe = make_pipeline(
        getCategoircalDataTransformed(["Key", "Quality", "Notes"]),
        chordModel,
    )

    pipe.fit(X, y)

    chordScore = pipe.score(y_train, y_test)

    return pipe, chordScore


@st.cache
def romanPredictionDF():
    df = loadData("Roman")
    X = df.drop(columns=["RomanNumeral"])
    y = df["RomanNumeral"]

    X_train, y_train, X_test, y_test = train_test_split(
        X, y, random_state=0, shuffle=True
    )

    pipe = make_pipeline(
        getCategoircalDataTransformed(["Key", "Quality", "Notes"]),
        romanModel,
    )

    pipe.fit(X, y)

    romanScore = pipe.score(y_train, y_test)

    return pipe, romanScore


@st.cache
def keyPredictionDF():
    df = loadData("Key")
    X = df.drop(columns=["Key"])
    y = df["Key"]

    X_train, y_train, X_test, y_test = train_test_split(
        X, y, random_state=0, shuffle=True
    )

    pipe = make_pipeline(
        getCategoircalDataTransformed(["Notes", "RomanNumeral"]),
        keyModel,
    )

    pipe.fit(X, y)

    romanScore = pipe.score(y_train, y_test)

    return pipe, romanScore


@st.cache
def notesPredictionDF():
    df = loadData("Notes")
    X = df.drop(columns=["Notes"])
    y = df["Notes"]

    X_train, y_train, X_test, y_test = train_test_split(
        X, y, random_state=0, shuffle=True
    )

    pipe = make_pipeline(
        getCategoircalDataTransformed(["Key", "Quality", "Chord", "RomanNumeral"]),
        notesModel,
    )

    pipe.fit(X, y)

    romanScore = pipe.score(y_train, y_test)

    return pipe, romanScore


chordPipe, chordScore = chordPredictionDF()
romanPipe, romanScore = romanPredictionDF()
keyPipe, keyScore = keyPredictionDF()
notesPipe, notesScore = notesPredictionDF()


f"""
# Chord Solver 2.0 (ML)
> **_Note:_** 

> The results are **_NOT_** perfect. The more accurate you are at your input, the more accurate the machine will be.

----------------------------------------------------------------
## Predict Roman Numeral
"""
# > Chord Prediction Accuracy = {100*chordScore}%
# > Roman Numeral Prediction Accuracy = {100*romanScore}%

predictions = st.multiselect(
    label="What would you like to predict?",
    options=["Key", "Notes", "Chord", "Roman Numeral"],
    default=["Chord", "Roman Numeral"],
)

col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)
options = [
    "C",
    "D",
    "E",
    "F",
    "G",
    "A",
    "B",
    "Cb",
    "Db",
    "Eb",
    "Fb",
    "Gb",
    "Ab",
    "Bb",
    "C#",
    "D#",
    "E#",
    "F#",
    "G#",
    "A#",
    "B#",
]

if len(predictions) == 0:
    "# ^ Select something to predict"

elif predictions[0] in ["Chord", "Roman Numeral"]:

    with col1:
        KEY = st.selectbox("Key", options)
    with col2:
        QUALITY = st.selectbox("Major or Minor", ["major", "minor"], index=0)
    with col3:
        NOTES = st.text_input("Enter Notes", value="C E G")
        NOTES = NOTES.title()

    y = pd.DataFrame({"Key": [KEY], "Quality": [QUALITY], "Notes": [NOTES]})

    f"""
    ---
    ### Predictions
    > ### Chord Name: {chordPipe.predict(y)[0]}
    > ### Roman Numeral: {romanPipe.predict(y)[0]}
    """

elif predictions == ["Key"]:

    with col1:
        NOTES = st.text_input("Enter Notes", value="C E G")
        NOTES = NOTES.title()
    with col2:
        ROMAN = st.selectbox(
            label="Roman Numeral",
            options=[
                "I",
                "ii",
                "iii",
                "IV",
                "V",
                "vi",
                "vii°",
                "i",
                "ii°",
                "III",
                "iv",
                "v",
                "VI",
                "VII",
            ],
        )
    with col3:
        ALTER = st.radio(label="Altered?", options=["#5", "b5"])

    ROMAN += ALTER
    key_y = pd.DataFrame({"Notes": [NOTES], "RomanNumeral": [ROMAN]})

    f"""
    ---
    ### Predictions
    > ### Key: {keyPipe.predict(key_y)[0]}
    """

elif predictions == "Notes":

    notes_y = pd.DataFrame({"Key": [], "Quality": [], "RomanNumeral": []})

    f"""
    ---
    ### Predictions
    > ### Chord Name: {chordPipe.predict(notes_y)[0]}
    > ### Notes: {notesPipe.predict(notes_y)[0]}
    """

# with col3:
#     RN = st.text_input("Enter Roman Numeral", value="I")
# with col4:
#     CHORD = st.text_input("Enter Chord Name", value="C Major")
