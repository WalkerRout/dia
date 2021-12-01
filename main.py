
# Import libraries used for constructing model
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics



def verifyColumn(col):

    '''
    Purpose: To verify whether or not a column should be checked for 0 values
    Parameters: A column to check
    Return values: A true or false value based on an argument provided
    '''

    return ((col == "Glucose") or (col == "BloodPressure") or (col == "SkinThickness") or (col == "Insulin") or (col == "BMI")) # Checks to see if col is equal to any of these



def showData(df):

    '''
    Purpose: Make a seaborn/matplotlib plot of the Pima First Nations dataframe
    Parameters: df (a dataframe to be pairplotted)
    Return values: N/A (not a pure function, intended side effects)
    '''

    #df.hist()
    #plt.show()
    sns.pairplot(data=df, diag_kind="kde", hue="Outcome") # visualize correlation between non-diabetic and diabetic, in which diabetic is orange and non-diabetic is blue
    plt.show()



def cleanZero(df):

    '''
    Purpose: Verify the values; every single datatype is numerical (vital for regression analysis)
    When dealing with this dataset, it is impossible to have 0 values for; Glucose, BloodPressure, SkinThickness, Insulin, and BMI
    Must clean up the data to rectify and 0 values. Several ways to do this, most low-effort one being removing all rows with 0 values. 
    Unfortunately, this will remove a substantial amount of data. Instead of removing the rows, I will replace all of the 0 values with NaN,
    find the mean, and then replace all of those NaN values with the found mean. This will make the data a little bit more consistent.
    Parameters: df (the dataframe to be cleaned)
    Return values: df (the cleaned dataframe)
    '''
    
    # Set 0 values to NaN so that they are not present when finding the mean
    for col in df.columns:
        if verifyColumn(col):
            df[col] = df[col].replace(0, np.nan)

    # Set NaN values with the found mean
    for col in df.columns:
        if verifyColumn(col):
            df[col] = df[col].fillna(df[col].mean())

    return df



def setupData(path):

    '''
    Purpose: This function serves to create the dataframe responsible for containing all of the cleaned data for use in regression analysis
    Parameters: path (path to the Pima dataset)
    Return values: The dataframe after all of the data has been cleaned and wrangled
    '''

    column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"] # creating names for the columns in the csv file
    df = pd.read_csv(path, names=column_names) # load the csv file into a pandas dataframe

    # Plot/show the dataframe (testing data, not for actual use)
    #showData(df)

    return cleanZero(df)



def logisticAnalysis(df):

    '''
    Purpose: Perform logistic regression on the dataframe passed as an argument
    Parameters: df (dataframe to perform logistic regression on)
    Return values: returns a dictionary containing the model score and a confusion matrix of actual values vs predicted values
    '''

    # Create sample values
    X = df.drop("Outcome", axis=1) # X values, drop the Outcome column since that is the result (axis=1 is for columns, axis=0 is for index)
    y = df[["Outcome"]] # y value, set to the Outcome column, since this is the actual conclusion of the patients symptoms

    # Create training and testing values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)

    # Create the logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train.squeeze()) # y_train is squeezed into a series
    
    # Predicted values given the test values
    y_predict = model.predict(X_test)
    
    '''
    # Compare the found values in a dataframe
    y_actual = y.iloc[X_test.index]
    y_actual = y_actual.squeeze()
    comparison = pd.DataFrame(data={"Actual Outcome" : y_actual, "Pred. Outcome" : list(y_predict)})
    '''

    # Model score and confusion matrix

    model_score = model.score(X_test, y_test)
    mat = metrics.confusion_matrix(y_test, y_predict) # First vector is false positive, second vector is false negative

    return {"score" : model_score, "confusion_matrix" : mat} # Return a dictionary of general values, can be modified in the future



if __name__ == "__main__":

    data = setupData("Resources/pima-first-nations-diabetes.csv") # Pass in path to Pima dataset as argument
    logA = logisticAnalysis(data) # Run regression analysis on data

    print(logA["score"]) # print the model score
    print(logA["confusion_matrix"]) # print the confusion matrix of the actual vs predicted values

