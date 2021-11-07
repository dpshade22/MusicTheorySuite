import random
import pandas as pd

from chordSolve import Chord


class Key:
    def __init__(self, root, quality=None):
        self.root = root.title()
        self.quality = quality
        self.romans = {}
        self.allNotes = [
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
            "Cbb",
            "Cb",
            "C",
            "C#",
            "C##",
        ]
        self.sharps = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
            "C",
        ]
        self.flats = [
            "C",
            "Db",
            "D",
            "Eb",
            "E",
            "F",
            "Gb",
            "G",
            "Ab",
            "A",
            "Bb",
            "B",
            "C",
            "Db",
            "D",
            "Eb",
            "E",
            "F",
            "Gb",
            "G",
            "Ab",
            "A",
            "Bb",
            "B",
            "C",
        ]

        self.sharpMajKeys = ["D", "G", "A", "E", "B", "F#", "C#"]
        self.flatMajKeys = ["F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"]
        self.sharpMinKeys = ["E", "A", "E", "B", "F#", "C#"]
        self.flatMinKeys = ["C", "D", "F", "G", "A"]

        self.majorSkips = [2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1]
        self.minorSkips = [2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2]

    def getMajorScale(self):
        scale = []

        # Default cases
        if self.root == "F#":
            scale = ["F#", "G#", "A#", "B", "C#", "D#", "E#", "F#"]
            return scale
        elif self.root == "C#":
            scale = ["C#", "D#", "E#", "F#", "G#", "A#", "B#", "C#"]
            return scale
        elif self.root == "Gb":
            scale = ["Gb", "Ab", "Bb", "Cb", "Db", "Eb", "F", "Gb"]
            return scale
        elif self.root == "Cb":
            scale = ["Cb", "Db", "Eb", "Fb", "Gb", "Ab", "Bb", "Cb"]
            return scale
        elif self.root == "C":
            scale = ["C", "D", "E", "F", "G", "A", "B", "C"]
            return scale
        if self.root in self.sharpMajKeys:
            i = self.sharps.index(self.root)
            scaleLength = 0
            skipper = 0
            scale.append(self.sharps[i])

            while scaleLength < 7:
                skipper += self.majorSkips[scaleLength]
                nextNote = self.sharps[i + skipper]

                scale.append(nextNote)
                scaleLength += 1
            return scale
        if self.root in self.flatMajKeys:
            i = self.flats.index(self.root)
            scaleLength = 0
            skipper = 0
            scale.append(self.flats[i])

            while scaleLength < 7:
                skipper += self.majorSkips[scaleLength]
                nextNote = self.flats[i + skipper]

                scale.append(nextNote)
                scaleLength += 1
            return scale

    def getMinorScale(self):
        scale = []

        # Default cases
        if self.root == "F#":
            scale = ["F#", "G#", "A#", "B", "C#", "D#", "E#", "F#"]
            return scale
        elif self.root == "C#":
            scale = ["C#", "D#", "E#", "F#", "G#", "A#", "B#", "C#"]
            return scale
        elif self.root == "Gb":
            scale = ["Gb", "Ab", "Bb", "Cb", "Db", "Eb", "F", "Gb"]
            return scale
        elif self.root == "Cb":
            scale = ["Cb", "Db", "Eb", "Fb", "Gb", "Ab", "Bb", "Cb"]
            return scale
        elif self.root == "C":
            scale = ["C", "D", "Eb", "F", "G", "Ab", "Bb", "C"]

        if self.root in self.sharpMinKeys:
            i = self.sharps.index(self.root)
            scaleLength = 0
            skipper = 0
            scale.append(self.sharps[i])

            while scaleLength < 7:
                skipper += self.minorSkips[scaleLength]
                nextNote = self.sharps[i + skipper]

                scale.append(nextNote)
                scaleLength += 1
            return scale
        if self.root in self.flatMinKeys:
            i = self.flats.index(self.root)
            scaleLength = 0
            skipper = 0
            scale.append(self.flats[i])

            while scaleLength < 7:
                skipper += self.minorSkips[scaleLength]
                nextNote = self.flats[i + skipper]

                scale.append(nextNote)
                scaleLength += 1
            return scale

    def getRomans(self):
        romansList = []

        if self.quality == "major" and self.getMajorScale() is None:
            self.quality = "minor"
        elif self.quality == "minor" and self.getMinorScale() is None:
            self.quality = "major"

        if self.quality == "major":
            romansList = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
            scale = self.getMajorScale()

        elif self.quality == "minor":
            romansList = ["i", "ii°", "III", "iv", "v", "VI", "VII"]
            scale = self.getMinorScale()

        scale += scale[1:]

        for i in range(len(romansList)):

            root = 0
            third = 2
            fifth = 4
            seventh = 6

            bot = scale[root + i]
            mid = scale[third + i]
            top = scale[fifth + i]
            sev = scale[seventh + i]

            romanNumeral = romansList[i]

            sharpFive = romanNumeral + "#5"
            flatFive = romanNumeral + "b5"
            addNine = romanNumeral + "+9"

            alteredChords = [sharpFive, flatFive, addNine, "None"]
            altProb = [5, 5, 5, 150]
            alteredOrNo = random.choices(alteredChords, altProb)

            if alteredOrNo == [sharpFive]:
                topNdx = self.allNotes.index(top)
                top = self.allNotes[topNdx + 1]
                romanNumeral = sharpFive

            elif alteredOrNo == [flatFive]:
                topNdx = self.allNotes.index(top)
                top = self.allNotes[topNdx - 1]
                romanNumeral = flatFive

            if romansList[i] == "v":
                harmOrMel = ["V", "v"]
                five = random.choice(harmOrMel)

                if five == "V":
                    ndx = self.allNotes.index(mid)
                    mid = self.allNotes[ndx + 1]
                    romanNumeral = "V"

            if romansList[i] == "IV" or romansList[i] == "iv":
                options = ["IV", "vii°/V", "iv"]
                optionProb = [45, 5, 45]
                option = random.choices(options, optionProb)

                if romansList[i] == "IV" and option == ["vii°/V"]:
                    botNdx = self.allNotes.index(bot)
                    midNdx = self.allNotes.index(mid)
                    topNdx = self.allNotes.index(top)
                    bot = self.allNotes[botNdx + 1]
                    mid = self.allNotes[midNdx]
                    top = self.allNotes[topNdx]
                    romanNumeral = "vii°/V"

            if romansList[i] == "ii" or romansList[i] == "ii°":
                neo = ["bII", "V/V", "E"]
                prob = [3, 15, 95]

                neoOrNot = random.choices(neo, prob)

                if neoOrNot[0] == "bII":
                    botNdx = self.allNotes.index(bot)
                    topNdx = self.allNotes.index(top)
                    bot = self.allNotes[botNdx - 1]
                    top = self.allNotes[topNdx - 1]

                    romanNumeral = "bII"

                elif neoOrNot[0] == "V/V":
                    if romansList[i] == "ii":
                        ndx = self.allNotes.index(mid)
                        mid = self.allNotes[ndx + 1]
                        romanNumeral = "V/V"
                    else:
                        midNdx = self.allNotes.index(mid)
                        mid = self.allNotes[midNdx + 1]
                        topNdx = self.allNotes.index(top)
                        top = self.allNotes[topNdx + 1]

                        romanNumeral = "V/V"

            if random.random() < 0.1:
                if random.random() < 0.7:
                    sevNdx = self.allNotes.index(sev)
                    sev = self.allNotes[sevNdx + random.randint(-1, 1)]

                k = f"{self.root} {self.quality} 7th"

                romanNumeral += "7"
                value = [f"{bot} {mid} {top} {sev}", f"{romanNumeral}"]

            else:
                k = f"{self.root} {self.quality}"
                value = [f"{bot} {mid} {top}", romanNumeral]

            if self.romans.get(k) is None:
                self.romans[k] = [value]
            else:
                self.romans[k].append(value)

        return self.romans


sharpMajKeys = ["D", "C", "G", "A", "E", "B", "F#", "C#"]
flatMajKeys = ["C", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"]
sharpMinKeys = ["E", "A", "E", "B"]
flatMinKeys = ["C", "D", "F", "G", "A"]

keys = [sharpMajKeys, flatMajKeys, sharpMinKeys, flatMinKeys]


def generateDataset(num):
    df = {"Key": [], "Quality": [], "Notes": [], "Chord": [], "RomanNumeral": []}

    for i in range(num):
        keys = [
            sharpMajKeys,
            flatMajKeys,
            sharpMinKeys,
            flatMinKeys,
        ]
        choiceKeyList = random.choice(keys)
        choiceKey = random.choice(choiceKeyList)

        if i % 2 == 0:
            choiceQuality = "major"
        else:
            choiceQuality = "minor"

        tempKey = Key(choiceKey, choiceQuality)

        for i in range(len(tempKey.getRomans()[f"{tempKey.root} {tempKey.quality}"])):

            isSev = False

            if random.random() < 0.5:
                try:
                    notesStr = tempKey.getRomans()[
                        f"{tempKey.root} {tempKey.quality} 7th"
                    ][i][0]
                    isSev = True

                except:
                    notesStr = tempKey.getRomans()[f"{tempKey.root} {tempKey.quality}"][
                        i
                    ][0]
            else:
                notesStr = tempKey.getRomans()[f"{tempKey.root} {tempKey.quality}"][i][
                    0
                ]

            notesList = notesStr.split()

            if random.random() < 0.3:
                notesStr = notesStr.replace(" ", "")

            currChord = Chord(tempKey.root)

            try:
                if len(notesList) >= 4:
                    chordName = currChord.identSev(
                        notesList[0], notesList[1], notesList[2], notesList[3]
                    )

                else:
                    chordName = currChord.identTriad(
                        notesList[0], notesList[1], notesList[2]
                    )
            except:
                continue

            df["Key"].append(tempKey.root)
            df["Quality"].append(tempKey.quality)
            df["Notes"].append(notesStr)
            if isSev:
                df["RomanNumeral"].append(
                    tempKey.getRomans()[f"{tempKey.root} {tempKey.quality} 7th"][i][1]
                )
            else:
                df["RomanNumeral"].append(
                    tempKey.getRomans()[f"{tempKey.root} {tempKey.quality}"][i][1]
                )
            df["Chord"].append(chordName)

    return df


df = pd.DataFrame(generateDataset(10000))

df.to_csv("./DataAndModels/data1000", index=False)
