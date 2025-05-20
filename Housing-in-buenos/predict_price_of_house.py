import warnings
import wqet_grader
warnings.simplefilter(action="ignore", category=FutureWarning)

# Import libraries here
import pandas as pd
#import matplotlib to plot 
import matplotlib.pyplot as plt
import plotly.express as px
from glob import glob
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.impute import SimpleImputer
from category_encoders import OneHotEncoder

# Build your `wrangle` function
def wrangle(filePath):
    df = pd.read_csv(filePath)
    # select on apartment in property type
    mask_proptype = df["property_type"] == "apartment"
    
    #subset our data frame to hold only places in Distrito Federal
    mask_place = df["place_with_parent_names"].str.contains("Distrito Federal")
    
    #subset our data frame to hold only places in distrito federal that cost less than 100000
    mask_price = df["price_aprox_usd"]  < 100_000
    
    
    df = df[ mask_proptype & mask_place & mask_price]
    
    high, low = df["surface_covered_in_m2"].quantile([0.9, 0.1])
    mask_outl = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_outl]
    
    #Create separate "lat" and "lon" columns, split and rop old lat-lon column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace =True)
    
    #Create a borough column that hold all 16 borough, and drop column that held the infrmaton as it is not useful anymore
    df["borough"] = df["place_with_parent_names"].str.split("|", expand=True)[1]
    df.drop(columns="place_with_parent_names", inplace =True)
    
    #drop column that has more than 50% missing values
    df.drop(columns=[ 
        "surface_total_in_m2", 
        "price_usd_per_m2", 
        "floor",
        "rooms",
        "expenses"
    ],
            inplace=True)
    
    #remove low and high cardinality, as they have no unique or too many unique value that wont help our model
    df.drop(columns=[
        "operation",
        "property_type", 
        "currency", 
        "properati_url"
    ], inplace=True)
    
    #drop leaky data, that give cheating info about price to our model 
    df.drop(columns=[
        "price_aprox_local_currency",
        "price", 
        "price_per_m2"
    ], 
                 inplace=True)
    return df

# Use this cell to test your wrangle function and explore the data
df = wrangle("data/mexico-city-real-estate-1.csv")
df.head()

files = glob("data/mexico-city-real-estate-*.csv")
files

frames = [wrangle(file) for file in files] 
type(frames)
df = pd.concat(frames, ignore_index=True)
print(df.info())
df.head()

# Build histogram
plt.hist(df["price_aprox_usd"])


# Label axes
plt.xlabel("Price [$]")

# Add title
plt.title("Count")

# Don't delete the code below 👇
plt.savefig("images/2-5-4.png", dpi=150)

# Build scatter plot
plt.scatter(df["surface_covered_in_m2"], df["price_aprox_usd"])

# Label axes
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")

# Add title
plt.title("Mexico City: Price vs. Area")

# Don't delete the code below 👇
plt.savefig("images/2-5-5.png", dpi=150)

# Plot Mapbox location and price
fig = px.scatter_mapbox(
    df,
    lat = "lat",
    lon = "lon",
    color= "price_aprox_usd",
    mapbox_style="open-street-map",
    zoom=10
)
fig.show()

# Split data into feature matrix `X_train` and target vector `y_train`.
#surface_covered_in_m2	lat	lon	borough
target= "price_aprox_usd"
feature= ["surface_covered_in_m2", "lat", "lon", "borough"]
X_train = df[feature]
y_train = df[target]

y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred_baseline)
print("Mean apt price:", y_mean)
print("Baseline MAE:", baseline_mae)

# Build Model
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    Ridge()
)
# Fit model
model.fit(X_train, y_train)

X_test = pd.read_csv("data/mexico-city-test-features.csv")
print(X_test.info())
X_test.head()

y_test_pred = pd.Series(model.predict(X_test))
y_test_pred.head()

#Communicate Result
# Step 1: Get the coefficient values from the trained Ridge model
# These tell us how much each feature affects the predicted house price
coefficients = model.named_steps["ridge"].coef_

# Step 2: Get the actual names of the features (like 'rooms', 'neighbourhood_Palermo', etc.)
# The OneHotEncoder converts categorical features to numbers, and this gives us their names
features = model.named_steps["onehotencoder"].get_feature_names_out()

# Step 3: Combine the feature names and coefficients into a single labeled list (Series)
# Each feature name is now matched to its influence (coefficient) on price
feat_imp = pd.Series(data=coefficients, index=features)

# Step 4: Sort the features by the absolute size of their impact (regardless of + or - sign)
# This shows which features have the most and least influence on the price, from least to most
feat_imp = feat_imp.reindex(feat_imp.abs().sort_values().index)

# Now, feat_imp shows the sorted list of features and how much each one influences price
feat_imp
