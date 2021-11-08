from re import A
import sklearn
from sklearn.neighbors import KNeighborsRegressor

import random
import pandas as pd
import numpy


notes = ["C", "D", "E", "F", "G", "A", "B"]
sharp = ["B#", "C#", "C##", "D#", "D##", "E#", "F#", "F##", "G#", "G##", "A#", "A##"]
flat = ["Dbb", "Db", "Ebb", "Eb", "Fb", "Gbb", "Gb", "Abb", "Ab", "Bbb", "Bb", "Cb"]

whiteNotes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

allNotes = [
    "Cbb",
    "Cb",
    "C",
    "C#",
    "C##",
    "Dbb",
    "Db",
    "D",
    "D#",
    "D##",
    "Ebb",
    "Eb",
    "E",
    "E#",
    "E##",
    "Fbb",
    "Fb",
    "F",
    "F#",
    "F##",
    "Gbb",
    "Gb",
    "G",
    "G#",
    "G##",
    "Abb",
    "Ab",
    "A",
    "A#",
    "A##",
    "Bbb",
    "Bb",
    "B",
    "B#",
    "B##",
]

theLists = [sharp, flat, whiteNotes]


def randomNote():
    return random.choice(random.choice(theLists))


baseDist = {1: "2nd", 2: "3rd", 3: "4th", 4: "5th", 5: "6th", 6: "7th"}

intervalsList = ["m2", "M2", "m3", "M3", "P4", "P5", "TT"]


triadChordTypes = {
    "m2": {"TT": "Sus b2"},
    "M2": {"P4": "Sus 2"},
    "m3": {"M2": "Dim b5", "m3": "Dim", "M3": "Minor", "P4": "Minor Augmented"},
    "M3": {"m3": "Major", "M3": "Aug"},
    "P4": {"M2": "Sus 4", "P4": "Quartal"},
    "TT": {"m2": "Sus #4"},
    "P5": {"P5": "Quintal"},
}

seventhChordTypes = {
    "m3": {
        "m3": {
            "m3": "Diminished 7",
            "M3": "Half diminished 7",
            "P5": "Diminished add 9",
        },
        "M3": {"P5": "Minor add 9"},
        "M3": {"M3": "Minor major 7th", "m3": "Minor 7th"},
    },
    "M3": {
        "m3": {
            "M3": "Major major 7th",
            "m3": "Major minor 7th",
            "M2": "Major Dimimished 7",
        }
    },
}

intervals = {
    1: "m2",
    2: "M2",
    3: "m3",
    4: "M3",
    5: "P4",
    6: "TT",
    7: "P5",
    8: "m6",
    9: "M6",
    10: "m7",
    11: "M7",
    -1: "M7",
    -2: "m7",
    -3: "M6",
    -4: "m6",
    -5: "P5",
    -6: "TT",
    -7: "P4",
    -8: "M3",
    -9: "m3",
    -10: "M2",
    -11: "m2",
}


def distOfNotes(a="", b=""):

    a = a.title()
    b = b.title()

    if a in sharp:
        locA = sharp.index(a)
    elif a in flat:
        locA = flat.index(a)
    else:
        locA = whiteNotes.index(a)

    if b in sharp:
        locB = sharp.index(b)
    elif b in flat:
        locB = flat.index(b)
    else:
        locB = whiteNotes.index(b)

    return intervals.get(locB - locA)


class Chord:
    def __init__(self, root, bottom="M3", top="m3", tiptop="m3", isSeventh=False):
        self.root = root.title()
        self.bottom = bottom
        self.top = top
        self.tiptop = tiptop
        self.isSeventh = False
        self.third = None
        self.fifth = None
        self.seventh = None
        self.ninth = None
        self.chord = []

    def createTriad(self, root, third, fifth):
        self.root = root
        try:
            chordName = triadChordTypes.get(self.bottom).get(self.top)
        except:
            chordName = None

        while chordName is None:
            self.bottom, self.top, self.tiptop = (
                random.choice(intervalsList),
                random.choice(intervalsList),
                random.choice(intervalsList),
            )
            try:
                self.bottom = distOfNotes(self.root, third)
                chordName = triadChordTypes.get(self.bottom).get(self.top)
            except:
                third = random.choice(allNotes)

                self.bottom, self.top = (
                    random.choice(intervalsList),
                    random.choice(intervalsList),
                )

            try:
                self.top = distOfNotes(self.third, fifth)
                chordName = triadChordTypes.get(self.bottom).get(self.top)
            except:
                fifth = random.choice(allNotes)

        self.chord.append(self.root)

        while len(self.chord) != 2:
            for i in range(len(whiteNotes)):
                thisGo = random.choice(theLists)
                if distOfNotes(self.root, thisGo[i]) == self.bottom:
                    self.third = thisGo[i]
                    self.chord.append(self.third)
        while len(self.chord) != 3:
            for i in range(len(whiteNotes)):
                thisGo = random.choice(theLists)
                if distOfNotes(self.third, thisGo[i]) == self.top:
                    self.fifth = thisGo[i]
                    self.chord.append(self.fifth)

        return f"{self.root} {chordName}"

    def identTriad(self, root, third, fifth):
        self.bottom = distOfNotes(root, third)
        self.top = distOfNotes(third, fifth)

        chordName = triadChordTypes.get(self.bottom).get(self.top)

        if chordName is None:
            chordName = self.createTriad(root, third, fifth)

        chord = f"{root.title()} {chordName}"

        return chord

    def identSev(self, root, third, fifth, seventh):

        self.bottom = distOfNotes(root, third)
        self.top = distOfNotes(third, fifth)
        self.seventh = distOfNotes(fifth, seventh)

        chordName = seventhChordTypes.get(self.bottom).get(self.top).get(self.seventh)

        if chordName is None:
            chordName = self.createSeventh(root, third, fifth, seventh)

        chord = f"{root.title()} {chordName}"

        return chord

    def createSeventh(self, root, third, fifth, seventh):
        self.root = root
        try:
            chordName = triadChordTypes.get(self.bottom).get(self.top)
        except:
            chordName = None

        while chordName is None:
            self.bottom, self.top, self.tiptop = (
                random.choice(intervalsList),
                random.choice(intervalsList),
                random.choice(intervalsList),
            )
            try:
                self.bottom = distOfNotes(self.root, third)
                chordName = triadChordTypes.get(self.bottom).get(self.top)
            except:
                third = random.choice(allNotes)

                self.bottom, self.top = (
                    random.choice(intervalsList),
                    random.choice(intervalsList),
                )

            try:
                self.top = distOfNotes(self.third, fifth)
                chordName = triadChordTypes.get(self.bottom).get(self.top)
            except:
                fifth = random.choice(allNotes)

            try:
                self.seventh = distOfNotes(self.fifth, seventh)
                chordName = triadChordTypes.get(self.bottom).get(self.seventh)
            except:
                seventh = random.choice(allNotes)

        self.chord.append(self.root)

        while len(self.chord) != 2:
            for i in range(len(whiteNotes)):
                thisGo = random.choice(theLists)
                if distOfNotes(self.root, thisGo[i]) == self.bottom:
                    self.third = thisGo[i]
                    self.chord.append(self.third)
        while len(self.chord) != 3:
            for i in range(len(whiteNotes)):
                thisGo = random.choice(theLists)
                if distOfNotes(self.third, thisGo[i]) == self.top:
                    self.fifth = thisGo[i]
                    self.chord.append(self.fifth)
        while len(self.chord) != 4:
            for i in range(len(whiteNotes)):
                thisGo = random.choice(theLists)
                if distOfNotes(self.fifth, thisGo[i]) == self.tiptop:
                    self.seventh = thisGo[i]
                    self.chord.append(self.seventh)

        return f"{self.root} {chordName}"


chordsDF = {"Chord": [], "Notes": []}


# def createDF():
#     # for i in range(3):
#     #     currNote = randomNote()
#     #     currChord = Chord(currNote)

#     #     currChordName, currChordNotes = currChord.createRandomTriad()

#     #     chordsDF[currChordName] = currChordNotes

#     for i in range(10000):

#         currNote = randomNote()
#         currChord = Chord(currNote)

#         chordName, chordNotes = currChord.identTriad("A", "C")
#         # dist = [50, 50]
#         chordsDF["Chord"].append(chordName)
#         chordsDF["Notes"].append(chordNotes)


# df = pd.DataFrame(createDF())
