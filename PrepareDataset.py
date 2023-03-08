#!/usr/bin/env python

"""Prepares the dataset prior to training.
This is necessary, because the format of the dataset is not as keras needs it.
"""
import os
import xml.etree.ElementTree as ET
import unicodedata
import random
from enum import Enum


class DataCategory(Enum):
    TRAINING = 1
    VALIDATION = 2
    TEST = 3

def stripControlChars(data: str) -> str:
    return ''.join(c for c in data if unicodedata.category(c) != 'Cc')

def main():

    # settings for splitting data
    # 1 - trainingSplit - validationSplit is testSplit
    random.seed(1337)
    trainingSplit = 0.8
    validationSplit = 0.1


    # find the data
    # (VS Code and bash use different root folder)
    if os.path.exists("unprocessedBlogs"):
        datasetPath = os.path.abspath("unprocessedBlogs")
        useDirectPaths = True
    elif os.path.exists("BlogClassification/unprocessedBlogs"):
        datasetPath = os.path.abspath("BlogClassification/unprocessedBlogs")
        useDirectPaths = False
    else:
        raise FileNotFoundError
    
    if useDirectPaths:
        targetPath = os.path.abspath("preparedDataset")
    else:
        targetPath = os.path.abspath("BlogClassification/preparedDataset")

    if not os.access(targetPath, os.W_OK):
        os.mkdir(targetPath)

    # Walk through data and prepare it.
    for path, directories, files in os.walk(datasetPath):

        amountOfFiles = len(files)
        print(amountOfFiles, "found in", path)
        filenumber = 1  # counter to determine progress

        for file in files:

            # extract info from filename: (id.gernder.age.industry.astroSign.xml)
            id, gender, age, industry, astrologicalSign = file.split(".")[0:-1]
            id = int(id)
            age = int(age)

            # determine label
            if age < 20:
                label = "10s"
            elif age < 30:
                label = "20s"
            else:
                label = "30s"

            # choosing whether post should be in training, validation or test data
            randomNumber = random.random()
            if randomNumber < trainingSplit:
                dataCategory = DataCategory.TRAINING
                dataCategoryString = "training_ds"
            elif randomNumber < trainingSplit + validationSplit:
                dataCategory = DataCategory.VALIDATION
                dataCategoryString = "validation_ds"
            else:
                dataCategory = DataCategory.TEST
                dataCategoryString = "test_ds"

            # creating file paths
            filepath = path + "/" + file  # path of original file
            idTargetFilePath = targetPath + "/" + dataCategoryString + "/" + label + "/" + str(id) + "/"

            # making sure the path to write to exists
            if not os.path.exists(targetPath + "/" + dataCategoryString):
                os.mkdir(targetPath + "/" + dataCategoryString)
            if not os.path.exists(targetPath + "/" + dataCategoryString + "/" + label):
                os.mkdir(targetPath + "/" + dataCategoryString + "/" + label)
            if not os.path.exists(idTargetFilePath):
                os.mkdir(idTargetFilePath)

            # filter characters which may break XML parsing
            with open(filepath, "r+", encoding="utf-8", errors="ignore") as f:

                content = f.read()

                # replace relevant xml tags with placeholders
                content = content.replace("<Blog>", "awuiegsnuirjsaa")
                content = content.replace("</Blog>", "asjdiofawjosijfeos")
                content = content.replace("<post>", "asdjkfaokeslrjaspolk")
                content = content.replace("</post>", "iaosfiuahjsahgnjnsvu")
                content = content.replace("<date>", "asfgoiaegsnvsjlaje")
                content = content.replace("</date>", "asdyboijwnisdjas")

                # eliminate all html/xml artefacts
                content = content.replace("<", "(")
                content = content.replace(">", ")")
                content = content.replace("&nbsp", " ")
                content = content.replace("&", "")
                # stip some strange characters
                content = stripControlChars(content)

                # put xml tags back
                content = content.replace("awuiegsnuirjsaa", "<Blog>")
                content = content.replace("asjdiofawjosijfeos", "</Blog>")
                content = content.replace("asdjkfaokeslrjaspolk", "<post>")
                content = content.replace("iaosfiuahjsahgnjnsvu", "</post>")
                content = content.replace("asfgoiaegsnvsjlaje", "<date>")
                content = content.replace("asdyboijwnisdjas", "</date>")

                # TODO: remove unused replace
                #content = content.replace("<>", "")
                #content = content.replace(" <img src=\"C:\Documents and Settings\Owner\My Documents\My Pictures\descrew.jpg\"", "")
                #content = content.replace("<-", "")
                #content = content.replace("->", "")
                #content = content.replace("(>", "")
                #content = content.replace("<)", "")
                #content = content.replace("<3", "love")
                #content = content.replace(">>", "")
                #content = content.replace("<<", "")
                #content = content.replace(" >< ", "")
                #content = content.replace(">i", "")
                #content = content.replace("i<", "")
                #content = content.replace(">.<", "")
                #content = content.replace(" :< ", " :( ")
                #content = content.replace(">_<", "")
                #content = content.replace(">^..^<", "")

                f.seek(0)
                f.truncate()
                f.write(content)   

            # Parse XML
            tree = ET.parse(filepath)
            root = tree.getroot()

            # iterate over blog posts and write to new file
            blogNumber = 1  # counter to name posts of blogger with ascending numbers
            for post in root.findall("./post"):

                postFilePath = idTargetFilePath + str(blogNumber) + ".txt"
                with open(postFilePath, "w") as f:
                    f.write(post.text)

                blogNumber += 1

            # Print progress
            progress = float(filenumber)/amountOfFiles * 100;
            progressString = str(filenumber) + "/" + str(amountOfFiles)
            progressString += " (" + str(int(progress)) + "%)"
            progressString += " " + filepath
            print(progressString)
            filenumber += 1
            




    return

if __name__ == "__main__":
    main()
