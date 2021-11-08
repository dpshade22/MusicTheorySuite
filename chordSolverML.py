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
            usecols=["Chord", "Key", "Quality", "Notes", "RomanNumeral"],
        )
    if vars == "Quality":
        df = pd.read_csv(
            "./DataAndModels/data10000",
            usecols=["Key", "Quality", "Chord", "RomanNumeral"],
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
qualityModel = load("./DataAndModels/model.joblib")
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
        getCategoircalDataTransformed(["Key", "Quality", "Chord", "RomanNumeral"]),
        notesModel,
    )

    pipe.fit(X_train, y_train)

    score = pipe.score(X_test, y_test)

    return pipe, score


@st.cache
def qualityPredictionDF():
    df = loadData("Quality")
    X = df.drop(columns=["Quality"])
    y = df["Quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=4, shuffle=True
    )

    pipe = make_pipeline(
        getCategoircalDataTransformed(["Key", "Quality", "Chord", "RomanNumeral"]),
        qualityModel,
    )

    pipe.fit(X_train, y_train)

    score = pipe.score(X_test, y_test)

    return pipe, score


chordPipe, chordScore = chordPredictionDF()
romanPipe, romanScore = romanPredictionDF()
keyPipe, keyScore = keyPredictionDF()
notesPipe, notesScore = notesPredictionDF()
qualityPipe, qualityScore = qualityPredictionDF()

f"""
# Chord Solver 2.0 (ML)
> **_Note:_** 

> The results are **_NOT_** perfect. The more accurate you are at your input, the more accurate the machine will be.

----------------------------------------------------------------
## Predict Roman Numeral
"""
# > Chord Prediction Accuracy = {100*chordScore}%
# > Roman Numeral Prediction Accuracy = {100*romanScore}%

predictions = st.selectbox(
    label="What would you like to predict?",
    options=["Key", "Notes", "Chord", "Roman Numeral", "Quality"],
    index=2,
)

col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)
keys = [
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

elif predictions in ["Chord", "Roman Numeral"]:

    with col4:
        KEY = st.selectbox("Key", keys)
    with col5:
        NOTES = st.text_input("Enter Notes", value="C E G")
        NOTES = NOTES.title()

    y = pd.DataFrame({"Key": [KEY], "Notes": [NOTES]})

    f"""
    ---
    ### Predictions

    Accuracy: {round(chordScore*100, 2)}%\n
    Accuracy: {round(romanScore*100, 2)}%

    > ### Chord Name: {chordPipe.predict(y)[0]}
    > ### Roman Numeral: {romanPipe.predict(y)[0]}
    
    """

elif predictions == "Key":

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
        ALTERSHARP5 = st.checkbox(label="#5")
        ALTERFLAT5 = st.checkbox(label="b5")
        SEVENTH = st.checkbox(label="7")
    if ALTERSHARP5:
        ROMAN += "^{#5}"
    elif ALTERFLAT5:
        ROMAN += "^{b5}"
    elif SEVENTH:
        ROMAN += "^7"

    key_y = pd.DataFrame({"Notes": [NOTES], "RomanNumeral": [ROMAN]})
    key_prediction = keyPipe.predict(key_y)[0]

    if ROMAN in [
        "i",
        "ii°",
        "III",
        "iv",
        "v",
        "VI",
        "VII",
    ]:
        key_prediction += " Minor"
    else:
        key_prediction += " Major"

    chord_y = pd.DataFrame(
        {"Key": [key_prediction], "Notes": [NOTES], "RomanNumeral": [ROMAN]}
    )
    chord_prediction = chordPipe.predict(chord_y)[0]

    f"""
    ---
    ### Prediction Accuracy: {round(keyScore*100, 2)}%
    > ### Chord Name: {chord_prediction}
    > ### Key: {key_prediction}

    """

elif predictions == "Notes":

    method = st.radio(label="Method", options=["Chord Name", "Key & Roman Numeral"])

    CHORD: str
    KEY: str
    QUALITY: str

    if method == "Chord Name":

        CHORD = st.text_input("Enter Chord Name", value="C Major")
        CHORD = CHORD.title()

    else:
        KEY = "C Major"
        QUALITY = "Major"

        colX, colY = st.columns(2)

        with colX:
            KEY = st.selectbox("Key", keys, index=0)
        with colY:
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
                index=0,
            )

        KEY = f"$${KEY}$$"

        if ROMAN in [
            "i",
            "ii°",
            "III",
            "iv",
            "v",
            "VI",
            "VII",
        ]:
            KEY += " $$Minor$$"
            QUALITY = " $$Minor$$"
        else:
            KEY += " $$Major$$"
            QUALITY = " $$Major$$"

        ROMAN = f"$${ROMAN}$$"

    notes_y = pd.DataFrame(
        {"Key": [KEY], "Quality": [QUALITY], "Chord": [CHORD], "RomanNumeral": [ROMAN]}
    )
    notes_prediction = notesPipe.predict(notes_y)[0]

    chord_y = pd.DataFrame({"Key": [KEY], "Notes": [notes_prediction]})
    chord_prediction = chordPipe.predict(chord_y)[0]

    f"""
    ---
    ### Prediction Accuracy: {round(notesScore*100, 2)}%
    > ### Chord Name: {chord_prediction}
    > ### Notes: {notes_prediction}
    """
if predictions == "Quality":

    quality_y = pd.DataFrame(
        {"Key": [], "Quality": [], "Chord": [], "RomanNumeral": []}
    )

    quality_prediction = qualityPipe.predict(quality_y)

    f"""
    ---
    ### Prediction Accuracy: {round(chordScore*100, 2)}%
    > ### Chord Name: {chord_prediction}
    """

# with col3:
#     RN = st.text_input("Enter Roman Numeral", value="I")
# with col4:
#     CHORD = st.text_input("Enter Chord Name", value="C Major")
