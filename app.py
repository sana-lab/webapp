import pandas as pd 
import streamlit as st
import sklearn
from sklearn.ensemble import RandomForestClassifier

st.title("Wine Quality Prediction Web Application")
st.markdown("""
This web app performs machine learning model and predicts wine quality by using wine features that you choose. 
The features default is set to be the mean of its dataset corresponding. 
* ** Python libraries that I have used:** pandas, streamlit, sklearn
""")


wine = pd.read_csv('C:/Users/mulis/Documents/Webapp/Data9.csv')


st.sidebar.header('User Input Features')
def user_input():
    fixed_acidity = st.sidebar.slider('fixed_acidity', 4.6, 16.0, 8.31)
    volatile_acidity = st.sidebar.slider('volatile_acidity', 0.12, 1.59, 0.52)
    citric_acid = st.sidebar.slider('citric_acid', 0.0, 1.1, 0.27)
    residual_sugar = st.sidebar.slider('residual_sugar', 0.9, 15.6, 2.5)
    chlorides = st.sidebar.slider('chlorides', 0.012, 0.612, 0.087467)
    free_sulfur_dioxide = st.sidebar.slider('free_sulfur_dioxide', 1.0, 73.0, 15.874)
    total_sulfur_dioxide = st.sidebar.slider('total_sulfur_dioxide', 6.0, 290.0, 46.677)
    density = st.sidebar.slider('density', 0.99, 1.004, 0.9967)
    pH = st.sidebar.slider('pH', 2.74, 4.02, 3.3111)
    sulphates = st.sidebar.slider('sulphates', 0.33, 2.1, 1.0)
    alcohol = st.sidebar.slider('alcohol', 8.4, 15.0, 10.0)

    data = {'fixed_acidity': fixed_acidity,
            'volatile_acidity': volatile_acidity,
            'citric_acid': citric_acid,
            'residual_sugar': residual_sugar,
            'chlorides': chlorides,
            'free_sulfur_dioxide': free_sulfur_dioxide,
            'total_sulfur_dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input()

st.subheader('User Input Parameters')
st.write(df)




from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = wine.drop('quality', axis = 1)
Y = wine['quality']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)



#Create a Gaussian Classifier
clf=RandomForestClassifier(random_state=0)

clf.fit(X, Y)

prediction=clf.predict(df)
predict_proba = clf.predict_proba(df)


st.subheader('Based on the features that you chose, the Quality Prediction is: ')
st.write(prediction)
st.markdown('3 being the lowest and 8 the highest')

pandadf = pd.DataFrame(data = predict_proba,
                       columns = ['3', '4', '5', '6', '7', '8'])

st.subheader('Prediction Probability')
st.write(pandadf)

st.markdown('***')
st.markdown('By Mulisana Gerveshi <3')
