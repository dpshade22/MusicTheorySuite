from sklearn import svm, linear_model, preprocessing, neighbors
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score
from generateDataset import generateDataset

import streamlit as st
import sklearn
import random
import pandas as pd
import numpy as np


@st.cache
def loadData():
    df = pd.read_csv("data")
    return df


# df.head()
# Scikit Learn Classification
ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
## Column Transformation and Model
def getCategoircalDataTransformed(columns=[]):
    column_trans = make_column_transformer(
        (ohe, columns),
        remainder="passthrough",
    )
    return column_trans


def getModel():
    model = neighbors.KNeighborsClassifier(15)
    return model


## Declare Supervised Training Set

df = loadData()
X = df.drop(columns=["RomanNumeral"])
y = df["RomanNumeral"]

## Declare Training and Testing Sets
X_train, y_train, X_test, y_test = train_test_split(X, y, random_state=0, shuffle=True)
## Establish Pipeline to Run Encoding and Model Simultaneously


def getPipeline():

    pipe = make_pipeline(
        getCategoircalDataTransformed(["Key", "Quality", "Notes"]), getModel()
    )
    pipe.fit(X, y)
    pipe.score(X_train, X_test)

    return pipe
