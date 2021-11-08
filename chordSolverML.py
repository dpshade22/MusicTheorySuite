import joblib
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


chordPipe, chordScore = chordPredictionDF()
romanPipe, romanScore = romanPredictionDF()
keyPipe, keyScore = keyPredictionDF()
notesPipe, notesScore = notesPredictionDF()

image = Image.open("./musicToolSuite.png")

st.image(image, use_column_width=True)

f"""
    # Chord Solver 2.0 (ML)
"""

moreInfo = st.checkbox("More Info")
prediction = st.checkbox("Prediction Accuracy")

if moreInfo:
    """
    > The results are **_NOT_** perfect. There currently isn't a good dataset for music theory, so I did my best to create my own.

    > In the meantime, the more accurate your input is, the more accurate the results will be.

    ----------------------------------------------------------------
    ## Predict Roman Numeral
    """
# > Chord Prediction Accuracy = {100*chordScore}%
# > Roman Numeral Prediction Accuracy = {100*romanScore}%

predictions = st.selectbox(
    label="What would you like to predict?",
    options=["Chord & Roman Numeral", "Key", "Notes"],
    index=2,
)

col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)
keys = [
    "Cb",
    "C",
    "C#",
    "Db",
    "D",
    "D#",
    "Eb",
    "E",
    "E#",
    "Fb",
    "F",
    "F#",
    "Gb",
    "G",
    "G#",
    "Ab",
    "A",
    "A#",
    "Bb",
    "B",
    "B#",
]

if len(predictions) == 0:
    "# ^ Select something to predict"

elif predictions in ["Chord & Roman Numeral"]:

    row2Col1, row2Col2 = st.columns(2)

    with row2Col1:
        KEY = st.selectbox("Key", keys, index=0)
    with row2Col2:
        NOTES = st.text_input("Enter Notes", value="")
        NOTES = NOTES.title()

    if NOTES != "" and KEY is not None:

        y = pd.DataFrame({"Key": [KEY], "Notes": [NOTES]})

        if prediction:
            f"""
            ---
            ### Predictions Accuracy: {round(chordScore*100, 2)}% & {round(romanScore*100, 2)}%
            > ### Chord Name: {chordPipe.predict(y)[0]}
            > ### Roman Numeral: {romanPipe.predict(y)[0]}
            """
        else:
            f"""
            ### Chord Name: {chordPipe.predict(y)[0]}
            ### Roman Numeral: {romanPipe.predict(y)[0]}
            """

elif predictions == "Key":

    with col1:
        NOTES = st.text_input("Enter Notes", value="")
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
        key_prediction += " minor"
    else:
        key_prediction += " major"

    if NOTES != "":
        chord_y = pd.DataFrame(
            {"Key": [key_prediction], "Notes": [NOTES], "RomanNumeral": [ROMAN]}
        )
        chord_prediction = chordPipe.predict(chord_y)[0]

        if prediction:
            f"""
            ---
            ### Prediction Accuracy: {round(keyScore*100, 2)}%
            > ### Chord Name: {chord_prediction}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;Key: {key_prediction}
            """
        else:
            f"""
            ---
            ### Chord Name: {chord_prediction} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;Key: {key_prediction}
            """

elif predictions == "Notes":

    # method = st.radio(label="Method", options=["Chord Name", "Key & Roman Numeral"])

    KEY = "C"

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

    notes_y = pd.DataFrame(
        {
            "Key": [KEY],
            "RomanNumeral": [ROMAN],
        }
    )

    #     CHORD = CHORD.title()
    notes_prediction = notesPipe.predict(notes_y)[0]

    chord_y = pd.DataFrame({"Key": [KEY], "Notes": [notes_prediction]})

    chord_prediction = chordPipe.predict(chord_y)[0]

    if prediction:
        f"""
        ---
        ### Prediction Accuracy: {round(notesScore*100, 2)}%
        > ### Chord: {chord_prediction}
        > ### Notes: {notes_prediction}
        """
    else:
        f"""
        ---
        > ### Chord: {chord_prediction}
        ### Notes: {notes_prediction}
        """

# with col3:
#     RN = st.text_input("Enter Roman Numeral", value="I")
# with col4:
#     CHORD = st.text_input("Enter Chord Name", value="C Major")
