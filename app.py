from chordSolverML import (
    chordPredictionDF,
    romanPredictionDF,
    keyPredictionDF,
    notesPredictionDF,
)
import streamlit as st


chordPipe, chordScore = chordPredictionDF()
romanPipe, romanScore = romanPredictionDF()
keyPipe, keyScore = keyPredictionDF()
notesPipe, notesScore = notesPredictionDF()

st.image("./musicToolSuite.png", use_column_width=True)

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
