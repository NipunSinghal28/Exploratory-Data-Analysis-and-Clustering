import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from  sklearn.linear_model import  LogisticRegression
from  sklearn.neighbors import KNeighborsClassifier
from  sklearn.tree import DecisionTreeClassifier
from  sklearn.svm import SVC
from  sklearn.cluster import KMeans
from  sklearn.preprocessing import MinMaxScaler

sns.set(rc={'figure.figsize':(12,10)})
def Titanic_Survival():
# GEEKFORGEEKS PROJECT


    #loading the dataset
    data=pd.read_csv('titanic.csv')
#    print(data.head(10))
    '''
    
    **TYPES OF FEATURES**
    - **Categorical**  - sex, embarked
    - **Continuous**   - Age, Fare
    - **Discrete**     - SibSp, Parch
    - **Alphanumeric** - Cabin
    
    '''
#    print(data.info())
#    print(data.isnull().sum())
#    print(data.describe())

# NUMERICAL VALUE ANALYSIS
    heatmap = sns.heatmap(data[['Survived','SibSp','Parch','Age','Fare']].corr(), annot=True)
    plt.figure(figsize=(12,10))
#    plt.show()

# sibsp - Number of siblings/spouses aboard the Titanic
#    print(data['SibSp'].nunique())
#    print(data['SibSp'].unique())
    sns.catplot(x='SibSp',y='Survived', data=data, kind='bar').set_ylabels('Survival Probability')
#    plt.show()

# AGE PARAMETER
    age_visual = sns.FacetGrid(data, col = 'Survived')
    age_visual = age_visual.map(sns.distplot,'Age')
 #   plt.show()

# SEX PARAMETER
    age_plot=sns.barplot(x='Sex', y='Survived', data=data).set_ylabel('Survival probability')
#    plt.show()

# PCLASS PARAMETER
    pclass= sns.catplot(x='Pclass', y='Survived', data=data, kind='bar', hue='Sex')
    plt.show()


def main():
    Titanic_Survival()

main()