from chordSolverML import (
    chordPredictionDF,
    romanPredictionDF,
    keyPredictionDF,
    notesPredictionDF,
)
import streamlit as st
from joblib import dump, load
import streamlit as st
import pandas as pd
import numpy as np
from streamlit.proto.Radio_pb2 import Radio

chordPipe, chordScore = chordPredictionDF()
romanPipe, romanScore = romanPredictionDF()
keyPipe, keyScore = keyPredictionDF()
notesPipe, notesScore = notesPredictionDF()

st.image("./Assets/musicToolSuite.png", use_column_width=True)

f"""
    # Chord Solver 2.0 (ML)
"""

moreInfo = st.checkbox("More Info")
prediction = st.checkbox("Prediction Accuracy")

if moreInfo:
    """
    > The results are **_NOT_** perfect. There currently isn't a good dataset for music theory, so I did my best to create my own.

    > In the meantime, the more accurate your input is, the more accurate the results will be. Put spaces between the notes.

    ----------------------------------------------------------------
    ## Predict Roman Numeral
    """
# > Chord Prediction Accuracy = {100*chordScore}%
# > Roman Numeral Prediction Accuracy = {100*romanScore}%

predictions = st.selectbox(
    label="What would you like to predict?",
    options=["Chord & Roman Numeral", "Key", "Notes"],
    index=0,
)

"---"

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
        KEY = st.selectbox("Key", keys, index=1)
    with row2Col2:
        NOTES = st.text_input("Enter Notes", value="")
        NOTES = NOTES.title()

    if NOTES != "" and KEY is not None:

        y = pd.DataFrame({"Key": [KEY], "Notes": [NOTES]})

        if prediction:
            f"""
            ---
            ### Predictions Accuracy: {round(chordScore*100, 2)}% & {round(romanScore*100, 2)}%
            > ### Chord Name: {chordPipe.predict(y)[0]} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; Roman Numeral: ${romanPipe.predict(y)[0]}$
            """
        else:
            f"""
            ---
            ### Chord Name: {chordPipe.predict(y)[0]} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; Roman Numeral: ${romanPipe.predict(y)[0]}$
            ###
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
        altered = st.radio("Altered", ["None", "#5", "b5", "7"])
    if altered == "#5":
        ROMAN += "^{\#5}"
    elif altered == "b5":
        ROMAN += "^{ b5}"
    elif altered == "7":
        ROMAN += "^7"

    print(ROMAN)

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
            > ### {chord_prediction} is the &nbsp;&nbsp; ${ROMAN}$     </pre>chord of {key_prediction}
            """
        else:
            f"""
            ---
            > ### {chord_prediction} is the ${ROMAN}$ chord of {key_prediction}
            """

elif predictions == "Notes":

    # method = st.radio(label="Method", options=["Chord Name", "Key & Roman Numeral"])

    KEY = "C"

    colX, colY, colZ = st.columns(3)

    with colX:
        KEY = st.selectbox("Key", keys, index=1)

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
    with colZ:
        altered = st.radio("Altered", ["None", "#5", "b5", "7"])

    if altered == "#5":
        ROMAN += "#5"
    elif altered == "b5":
        ROMAN += "b5"
    elif altered == "7":
        ROMAN += "7"

    print(ROMAN)

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
        ### Chord: {chord_prediction}
        ### Notes: {notes_prediction}
        """

# with col3:
#     RN = st.text_input("Enter Roman Numeral", value="I")
# with col4:
#     CHORD = st.text_input("Enter Chord Name", value="C Major")
