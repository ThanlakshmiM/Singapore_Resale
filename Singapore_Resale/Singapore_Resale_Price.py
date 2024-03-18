#install & import the libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from PIL import Image 
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np


def concatenate_preprocess():
    df1=pd.read_csv(r"Singapore_Resale/ResaleFlatPricesBasedonApprovalDate19901999.csv")
    df2=pd.read_csv(r"Singapore_Resale/ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv")
    df3=pd.read_csv(r"Singapore_Resale/ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv")
    df4=pd.read_csv(r"Singapore_Resale/ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv")
    df5=pd.read_csv(r"Singapore_Resale/ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv")
    
    # Create a list of dataframes
    df_list = [df1, df2, df3, df4, df5]
    #920445 rows × 11 columns

    # Concatenate the dataframes
    df = pd.concat(df_list)
    # fill the mode of  majority null value columns
    # (i.e) remaining_lease = 7,09,050 null value 

    # remaining lease datatype of 'O' error dtype mixed column so extract to the numerical value of year 
    # Loop through the elements of the mixed column
    # to convert Years and months in 'remaining_lease' column into a 'float' variable
    df['Remaining_lease_float'] = (pd.to_numeric(df['remaining_lease'].str.extractall('(\d+)')[0])
                 .unstack().div([1, 12]).sum(axis=1)
                 .round(2))
    #Handling Null values
    columns_modes = df[['Remaining_lease_float']].mode().iloc[0]
    df[['Remaining_lease_float']] = df[['Remaining_lease_float']].fillna(columns_modes)
    
    df['flat_type'] = df['flat_type'].str.replace(r'MULTI-GENERATION','MULTI GENERATION',regex=True)
    # Converting the flat_model column to lowercase
    df['flat_model'] = df['flat_model'].str.lower()
    # Splitting month col
    month=df['month'].str.split('-',expand = True)
    # Creating the year values as column
    df.insert(loc = 1,
          column = 'selling_year',
          value = month[0])
    # Creating the month values as column
    df.insert(loc = 2,
          column = 'selling_month',
          value = month[1])    
    df['selling_month'] = df['selling_month'].astype('int64')
    df['selling_year']=df['selling_year'].astype('int64')
    #drop remaining lease and month
    df.drop(['month','remaining_lease'],axis = 1,inplace = True)
    return df


def preprocessing_numeric(df):
     
    #LabelEncoder using categarical column of fit transform
    le=LabelEncoder()
    df["town"]=le.fit_transform(df["town"])
    df["flat_type"]=le.fit_transform(df["flat_type"])
    df["storey_range"]=le.fit_transform(df["storey_range"])
    df['block']=le.fit_transform(df['block'])
    df['street_name']=le.fit_transform(df['street_name'])
    df['flat_model']=le.fit_transform(df['flat_model'])

    return df

def EDA_process(df):
     #EDA process
   # Treat the outliers of IQR method
   def treat_outliers(columns):
     for i in columns:
        percentile25 = df[i].quantile(0.25)
        percentile75 = df[i].quantile(0.75)
        iqr = percentile75 - percentile25      # IQR = Q3 - Q1
        upper_limit = percentile75 + 1.5 * iqr
        lower_limit = percentile25 - 1.5 * iqr
        df[i]= np.where(df[i] > upper_limit, upper_limit, np.where(df[i] < lower_limit, lower_limit, df[i]))
    
   treat_outliers(['floor_area_sqm','storey_range','resale_price','lease_commence_date','Remaining_lease_float'])
   
   def log_transformation(columns):
    for i in columns:
        df[i] = np.log(df[i])

   log_transformation(['Remaining_lease_float','resale_price'])


   return df

def algorithm(df):
     
     # Assuming your DataFrame is named 'df'
     features = df[['selling_year','selling_month','town', 'flat_type','block','street_name','storey_range','floor_area_sqm','flat_model','lease_commence_date','Remaining_lease_float']]
     target = df['resale_price']

     # Split the data into training and testing sets
     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.15)

     # Create and train a linear regression model
     model = DecisionTreeRegressor()
     #model=RandomForestRegressor()
     model.fit(X_train, y_train)
     # Make predictions on the test set
     y_pred = model.predict(X_test)

    # Evaluate the model
     mse = mean_squared_error(y_test, y_pred)
     rmse = np.sqrt(mse)
     r2 = r2_score(y_test, y_pred)
     mae = mean_absolute_error(y_test, y_pred)
     acc_rf= round(model.score(X_train, y_train) * 100, 2)

     metrics = {'R2': r2,
           'Mean Absolute Error': mae,
           'Mean Squared Error': mse,
           'Root Mean Squared Error': rmse,
           "Accuracy " : str(acc_rf)+"%" }
    #print(f"Mean Squared Error: {mse}")

       
     return model,metrics

     
def Numeric(value,select):
    con=concatenate_preprocess()
    town=con["town"].unique()
    flat_type=con["flat_type"].unique()
    storey_range=con["storey_range"].unique()
    block=con["block"].unique()
    street_name=con["street_name"].unique()
    flat_model=con["flat_model"].unique()
    numeric=preprocessing_numeric(con)
    if select == "town" :
        for i in town:
            if i == value:
                town_map=dict(zip(town,numeric["town"].unique()))
                return town_map[value]
    if select == "flat_type" :
        for i in flat_type:
            if i == value:
                rooms_map=dict(zip(flat_type,numeric["flat_type"].unique()))
                return rooms_map[value]
    if select == "storey_range" :
        for i in storey_range:
            if i == value:
                storey_map=dict(zip(storey_range,numeric["storey_range"].unique()))
                return storey_map[value]
    if select == "block" :
        for i in block:
            if i == value:
                block_map=dict(zip(block,numeric["block"].unique()))
                return block_map[value]
    if select == "street_name" :
        for i in street_name:
            if i == value:
                street_name_map=dict(zip(street_name,numeric["street_name"].unique()))
                return street_name_map[value]
    if select == "flat_model" :
        for i in flat_model:
            if i == value:
                flat_model_map=dict(zip(flat_model,numeric["flat_model"].unique()))
                return flat_model_map[value]

#page congiguration
st.set_page_config(page_title= "Singapore House Price Prediction",
                   page_icon= 'random',
                   layout= "wide",)
st.markdown("<h1 style='text-align: center; color: white;'> Singapore Resale Flat Price Pridiction </h1>", unsafe_allow_html=True)


#application background
def app_bg():
    st.markdown(f""" <style>.stApp {{
                        background: url("https://cdn.wallpapersafari.com/7/90/BFUQb1.jpg");
                        background-size: cover}}
                     </style>""",unsafe_allow_html=True)
app_bg()



#Creating option menu in the menu bar
selected = option_menu(None,['Home','Metrics',"Predict"],
                        icons=["house","at","toggles"],
                        default_index=0,
                        orientation="horizontal")

if selected == 'Home':
   col1,col2=st.columns(2,gap='medium')

   col1.markdown("### :blue[Domain] : 👉 Real Estate")
   col1.markdown("### :blue[Skills take away From This Project] :👉 Data Wrangling, EDA, Model Building, Model Deployment")
   col1.markdown("### :blue[Overview] : 👉Singapore  Resale Flat Prices Predicting, then build regression ML models and performance based on model accuracy and RMSE in Python .")
   flat=Image.open(r'Singapore_Resale/1_N0YOUGrSXw9ILS9nCczDjg.jpg')
   col2.image(flat)
   


if selected == 'Metrics': 
    con=concatenate_preprocess()
    df1=preprocessing_numeric(con)
    df=EDA_process(df1)
    model,score=algorithm(df)
    st.write("Metrics",score)

if selected == 'Predict': 
  with st.form('my_form'):     
     con=concatenate_preprocess()
     col1,col2=st.columns(2)
     selling_year=col1.selectbox('selling_year',sorted(con['selling_year'].unique()),key=1)
     selling_month=col1.selectbox('selling_month',sorted(con['selling_month'].unique()),key=2)
     town=col1.selectbox('Town',sorted(con['town'].unique()),key=3)
     flat_type=col2.selectbox('number_of_rooms',sorted(con['flat_type'].unique()),key=4)
     block=col2.selectbox('block',sorted(con['block'].unique()),key=5)
     street_name=col2.selectbox('street_name',sorted(con['street_name'].unique()),key=6)
     storey_range=col1.selectbox('storey_range',sorted(con['storey_range'].unique()),key=7)
     lease_commence_date=col2.selectbox('lease_commence_dated',sorted(con['lease_commence_date'].unique()),key=8)
     floor_area_sqm=col1.selectbox('floor_area_sqm',sorted(con['floor_area_sqm'].unique()),key=9)
     flat_model=col1.selectbox('flat_model',sorted(con['flat_model'].unique()),key=10)
     remaining_lease=col2.selectbox('remaining_lease',sorted(con['Remaining_lease_float'].unique()),key=11)
#st.button("Predict")
     if st.form_submit_button("Predict"):
        df1=preprocessing_numeric(con)
        df=EDA_process(df1)
        model,score=algorithm(df)
        # Now, you can use the trained model to make predictions on new data
        # For example, if you have a new data point with 'floor_area_sqm_new' and 'lease_commence_date_new'
        #new_data_point = [[2024,3,26,5,1748,572,1.386294,4.955827,1.098612,1987,4.553877]] 
        new_data_point=np.array([[selling_year,selling_month,Numeric(town,'town'),Numeric(flat_type,'flat_type'),Numeric(block,'block'),Numeric(street_name,'street_name'),Numeric(storey_range,'storey_range'),floor_area_sqm,Numeric(flat_model,'flat_model'),lease_commence_date,remaining_lease]])
        st.write(new_data_point)
        predicted_price = model.predict(new_data_point)
       # print(f"Predicted Resale Price for the new data point: {predicted_price[0]}")
    
        st.write('Flat Resale Price : ',predicted_price[0])
       
