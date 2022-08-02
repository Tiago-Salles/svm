
from click import command
import pandas
import numpy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC, SVC
import seaborn


def main():
    file = pandas.read_csv("C:\\Users\\Houseasy - Frontend\\Documents\\Development\\svm_project\\commands/commands.csv")
    file.head(3)
    seaborn.countplot(y="Izzy", data=file)
    
main()