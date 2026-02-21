#Problem 1: YouTube Video Performance Predictor

# 1. Import Required Libraries

import numpy as np                  # For numerical operations
import pandas as pd                 # For data handling
import matplotlib.pyplot as plt     # For visualization
from sklearn.linear_model import LinearRegression  # ML model

# 2. Load the Dataset
data = {
    'ctr': [3.2, 5.8, 2.1, 7.4, 4.5, 6.2, 1.8, 5.1, 3.9, 8.5,
            4.2, 2.8, 6.8, 3.5, 7.1, 2.4, 5.5, 4.8, 6.5, 3.1,
            7.8, 2.6, 5.3, 4.1, 6.9, 3.7, 8.1, 2.2, 5.9, 4.6],

    'total_views': [12000, 28000, 8500, 42000, 19000, 33000, 7000, 24000, 16000, 51000,
                    18000, 11000, 38000, 14500, 44000, 9500, 26000, 21000, 35000, 13000,
                    47000, 10000, 25000, 17500, 40000, 15000, 49000, 8000, 31000, 20000]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# 3. Explore the Data
print(df.head())        # Display first 5 rows
print(df.describe())    # Statistical summary

# 4. Visualize the Data
plt.scatter(df['ctr'], df['total_views'])
plt.xlabel('Click Through Rate (CTR %)')
plt.ylabel('Total Views after 30 days')
plt.title('Relationship between CTR and YouTube Views')
plt.show()

# 5. Prepare Feature and Target
X = df[['ctr']]          # Independent variable (Feature)
y = df['total_views']   # Dependent variable (Target)

# 6. Train Linear Regression Model
model = LinearRegression()  # Create model
model.fit(X, y)             # Train model

# 7. Display Model Equation
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# 8. Predict Views for 8% CTR
predicted_views = model.predict([[8]])
print("Predicted views for 8% CTR:", int(predicted_views[0]))

##########    Problem 2: Food Delivery Time Predictor   ############

# 1. Import Required Libraries
import numpy as np                  # For numerical operations
import pandas as pd                 # For data handling
import matplotlib.pyplot as plt     # For visualization
from sklearn.linear_model import LinearRegression  # ML model

# 2. Load the Dataset
data = {
    'distance_km': [2.5, 6.0, 1.2, 8.5, 3.8, 5.2, 1.8, 7.0, 4.5, 9.2,
                    2.0, 6.5, 3.2, 7.8, 4.0, 5.8, 1.5, 8.0, 3.5, 6.8,
                    2.2, 5.5, 4.2, 9.0, 2.8, 7.2, 3.0, 6.2, 4.8, 8.2],

    'prep_time': [10, 20, 8, 25, 12, 18, 7, 22, 15, 28,
                  9, 19, 11, 24, 14, 17, 6, 26, 13, 21,
                  10, 16, 14, 27, 11, 23, 12, 18, 15, 25],

    'delivery_time': [18, 38, 12, 52, 24, 34, 14, 45, 29, 58,
                       15, 40, 21, 50, 27, 35, 11, 54, 23, 43,
                       17, 32, 26, 56, 19, 47, 20, 37, 30, 53]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# 3. Explore the Data
print(df.head())        # Show first 5 rows
print(df.describe())    # Statistical summary

# 4. Visualize Relationships
# Distance vs Delivery Time
plt.scatter(df['distance_km'], df['delivery_time'])
plt.xlabel('Distance (km)')
plt.ylabel('Delivery Time (min)')
plt.title('Distance vs Delivery Time')
plt.show()

# Prep Time vs Delivery Time
plt.scatter(df['prep_time'], df['delivery_time'])
plt.xlabel('Preparation Time (min)')
plt.ylabel('Delivery Time (min)')
plt.title('Preparation Time vs Delivery Time')
plt.show()

# 5. Prepare Features and Target
X = df[['distance_km', 'prep_time']]   # Independent variables
y = df['delivery_time']               # Dependent variable

# 6. Train Multiple Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# 7. Model Coefficients
print("Coefficient for Distance (km):", model.coef_[0])
print("Coefficient for Prep Time (min):", model.coef_[1])
print("Intercept:", model.intercept_)

# 8. Prediction for Given Input

# Given: distance = 7 km, prep time = 15 min
predicted_time = model.predict([[7, 15]])

print("Expected delivery time (minutes):", round(predicted_time[0], 2))


########## Problem 3: Laptop Price Predictor ##########

# 1. Import Required Libraries
import numpy as np                  # Numerical operations
import pandas as pd                 # Data handling
import matplotlib.pyplot as plt     # Visualization
from sklearn.linear_model import LinearRegression  # ML model
from sklearn.metrics import r2_score               # Model accuracy

# 2. Load the Dataset
data = {
    'ram': [4,8,4,16,8,8,4,16,8,16,4,8,4,16,8,8,4,16,8,16,
            4,8,4,16,8,8,4,16,8,16],
    
    'storage': [256,512,128,512,256,512,256,1024,256,512,128,512,
                256,1024,256,512,128,512,256,1024,256,512,128,512,
                256,512,256,1024,256,512],
    
    'processor': [2.1,2.8,1.8,3.2,2.4,3.0,2.0,3.5,2.6,3.0,1.6,2.8,
                  2.2,3.4,2.5,2.9,1.9,3.1,2.3,3.6,2.0,2.7,1.7,3.3,
                  2.4,3.0,2.1,3.5,2.6,3.2],
    
    'price': [28000,45000,22000,72000,38000,52000,26000,95000,42000,68000,
              20000,48000,29000,88000,40000,50000,23000,70000,36000,98000,
              25000,46000,21000,75000,39000,53000,27000,92000,43000,73000]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# 3. Explore the Data
print(df.head())        # First 5 rows
print(df.describe())    # Summary statistics

# 4. Visualize Relationships

# RAM vs Price
plt.scatter(df['ram'], df['price'])
plt.xlabel('RAM (GB)')
plt.ylabel('Price (INR)')
plt.title('RAM vs Laptop Price')
plt.show()

# Storage vs Price
plt.scatter(df['storage'], df['price'])
plt.xlabel('Storage (GB)')
plt.ylabel('Price (INR)')
plt.title('Storage vs Laptop Price')
plt.show()

# Processor vs Price
plt.scatter(df['processor'], df['price'])
plt.xlabel('Processor Speed (GHz)')
plt.ylabel('Price (INR)')
plt.title('Processor Speed vs Laptop Price')
plt.show()

# 5. Prepare Features and Target

X = df[['ram', 'storage', 'processor']]  # Independent variables
y = df['price']                          # Dependent variable

# 6. Train Multiple Linear Regression Model

model = LinearRegression()
model.fit(X, y)

# 7. Model Coefficients
print("Coefficient for RAM:", model.coef_[0])
print("Coefficient for Storage:", model.coef_[1])
print("Coefficient for Processor:", model.coef_[2])
print("Intercept:", model.intercept_)

# 8. Model Accuracy (R² Score)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print("R² Score:", r2)

# 9. Prediction for Meera's Laptop

# Input: 16GB RAM, 512GB Storage, 3.2 GHz Processor
predicted_price = model.predict([[16, 512, 3.2]])

print("Predicted fair price (INR):", int(predicted_price[0]))

# 10. Bonus: Check if Laptop is Overpriced

# Given laptop: 8GB RAM, 512GB Storage, 2.8 GHz, Price = 55,000
predicted_price_2 = model.predict([[8, 512, 2.8]])

print("Predicted price for comparison laptop:", int(predicted_price_2[0]))
