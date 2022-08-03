
from cgi import test
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
    x = []
    for data in range(file.shape[0]):
        x.append(file.iloc[data][1])
    y = numpy.array(file["Commands"])
    x_train, x_test, y_train, Y_test = train_test_split(x, y, test_size=0.3)
    model = Pipeline([
        ("vectorizer", CountVectorizer())
        ("tfidf", TfidfTransformer())
        ("clf", LinearSVC(C=0.2))
    ])
    model.fit(x_train, y_train)
main()