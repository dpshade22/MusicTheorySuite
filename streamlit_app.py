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
import streamlit.components.v1 as components

st.set_page_config(
    layout="wide",
    page_title="Music Theory Suite",
    page_icon="./Assets/musicTheoryLogo.jpeg",
)

chordPipe, chordScore = chordPredictionDF()
romanPipe, romanScore = romanPredictionDF()
keyPipe, keyScore = keyPredictionDF()
notesPipe, notesScore = notesPredictionDF()

page = st.sidebar.radio(
    "Choose your page", ["Music Tools v3", "Music Tools v1 & v2", "About"]
)


if page == "Music Tools v3":
    # Display details of page 1

    st.image("./Assets/mustheorysuite.png", width=750, clamp=True)

    f"""
    # Music Tools - v3
    """

    moreInfo = st.expander(
        "More Info",
    )
    # prediction = st.checkbox("Prediction Accuracy")
    prediction = False

    with moreInfo:
        """
        > The results are **_NOT_** perfect. There currently isn't a good dataset for music theory, so I did my best to create my own.

        > Until there is better data, the more accurate your input is, the more accurate the results will be. Put spaces between the notes.
        """
    # > Chord Prediction Accuracy = {100*chordScore}%
    # > Roman Numeral Prediction Accuracy = {100*romanScore}%

    """ 
    ----------------------------------------------------------------
    # Predictor:
    """
    predictions = st.selectbox(
        label="What would you like to predict?",
        options=["Chord & Roman Numeral", "Key", "Notes"],
        index=0,
    )

    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)
    keys = [
        "C♭",
        "C",
        "C♯",
        "D♭",
        "D",
        "D♯",
        "E♭",
        "E",
        "E♯",
        "F♭",
        "F",
        "F♯",
        "G♭",
        "G",
        "G♯",
        "A♭",
        "A",
        "A♯",
        "B♭",
        "B",
        "B♯",
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
                ## Result:
                ### Predictions Accuracy: {round(chordScore*100, 2)}% & {round(romanScore*100, 2)}%
                > ### Chord Name: {chordPipe.predict(y)[0]} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; Roman Numeral: ${romanPipe.predict(y)[0]}$
                """
            else:
                f"""
                ---
                ## Result:
                ### Chord Name: {chordPipe.predict(y)[0]} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; Roman Numeral: ${romanPipe.predict(y)[0]}$
                ###
                """

    elif predictions == "Key":

        with col1:
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
        with col2:
            NOTES = st.text_input("Enter Notes", value="")
            NOTES = NOTES.title()
        with col3:
            altered = st.radio("Altered", ["None", "♯5", "♭5", "7"])

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

            if altered == "♯5":
                ROMAN += "^{\♯5}"
                chord_prediction += " ${\♯5}$"
            elif altered == "♭5":
                ROMAN += "^{ ♭5}"
                chord_prediction += " ${ ♭5}$"
            elif altered == "7":
                ROMAN += "^7"

            chord_prediction = chord_prediction.replace("b", "$♭$")
            chord_prediction = chord_prediction.replace("#", "$♯$")

            key_prediction = key_prediction.replace("b", "$♭$")
            key_prediction = key_prediction.replace("#", "$♯$")

            if prediction:
                f"""
                ---
                ## Result:
                ### Prediction Accuracy: {round(keyScore*100, 2)}%
                > ### {chord_prediction} is the &nbsp;&nbsp; ${ROMAN}$     </pre>chord of {key_prediction}
                """
            else:
                f"""
                ---
                ## Result:
                > ### {chord_prediction} is the ${ROMAN}$ chord of {key_prediction}
                """

    elif predictions == "Notes":

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
            altered = st.radio("Altered", ["None", "♯5", "♭5", "7"])

        if altered == "♯5":
            ROMAN += "♯5"
        elif altered == "♭5":
            ROMAN += "♭5"
        elif altered == "7":
            ROMAN += "7"

        print(ROMAN)

        KEY = KEY.replace("♭", "b")
        KEY = KEY.replace("♯", "#")

        notes_y = pd.DataFrame(
            {
                "Key": [KEY],
                "RomanNumeral": [ROMAN],
            }
        )

        notes_prediction = notesPipe.predict(notes_y)[0]

        chord_y = pd.DataFrame({"Key": [KEY], "Notes": [notes_prediction]})

        chord_prediction = chordPipe.predict(chord_y)[0]

        chord_prediction = chord_prediction.replace("b", "$♭$")
        chord_prediction = chord_prediction.replace("#", "$♯$")

        notes_prediction = notes_prediction.replace("b", "$♭$")
        notes_prediction = notes_prediction.replace("#", "$♯$")

        if prediction:
            f"""
            ---
            ## Result:
            ### Prediction Accuracy: {round(notesScore*100, 2)}%
            > ### Chord: {chord_prediction}
            > ### Notes: {notes_prediction}
            """
        else:
            f"""
            ---
            ## Result:
            ### Chord: {chord_prediction}
            ### Notes: {notes_prediction}
            """

if page == "Music Tools v1 & v2":

    """
    # Chord Solver iOS
    [Click here to download for iOS](https://apps.apple.com/us/app/chord-solver/id1564025162)
    """

    "# Chord Solver Website "

    "[Click here to go to website](http://www.chordsolver.com/) \n"
    st.components.v1.iframe(src="http://chordsolver.com", height=550, scrolling=True)

if page == "About":

    """
    # What is Music Theory Suite?
    > The _**Music Theory Suite**_ is an idea by [Dylan Shade](https://dylan-shade-creations.super.site/) to create a tool kit for those who are interested in working with music theory.


    ---
    # What is this website?
    *This website provides a machine learning implimentation of most of the stuff presented in the previous two apps, but with extended functionality. The current data being used was created by me. This is a current limitation of this website. If there is a better data set, the results have the potential to be even more accurate.*

    ---
    # The Future of Music Theory Suite
    My goal with Music Theory Suite is to add tools that could help you with any music theory questions. As of right now, the following are in consideration of being added:
    - Atonal matrix completion
    - Set completion
    - Aural skills help

    ---

    # Other tools in the Music Theory Suite
    *There are currently two other versions of some basic music theory tools:*\n
    1. The [Chord Solver iOS](https://apps.apple.com/us/app/chord-solver/id1564025162) app that helps you build chords by picking the type of chord and entering the note. The iOS app also allows you to build scales in the same manner.\n
    2. The [Chord Solver Website](http://www.chordsolver.com/) does the inverse of what the app does. Here you can enter notes into a search bar and it will tell you the type of interval, triad, or seventh chord that it creates. \n

    """

"---"
components.html(
    '<script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="dylanshade" data-color="#ffffff" data-emoji=""  data-font="Lato" data-text="Buy me a coffee" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#FFDD00" ></script>'
)
