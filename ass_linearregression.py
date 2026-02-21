#-----------------------------------------------------

##########    Example 1: Study Hours vs Exam Marks   ############

# Step 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 2: Create the dataset
data = {
    'study_hours': [2, 4, 6, 8],
    'exam_marks': [35, 50, 65, 80]
}

df = pd.DataFrame(data)

# Step 3: Explore the data
print("Dataset:")
print(df)
print("\nBasic Statistics:")
print(df.describe())

# Step 4: Visualize the relationship
plt.scatter(df['study_hours'], df['exam_marks'])
plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.title("Study Hours vs Exam Marks")
plt.show()

# Step 5: Prepare data for training
X = df[['study_hours']]   # Feature (2D)
y = df['exam_marks']      # Target

# Step 6: Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 7: Model parameters
print("\nModel Equation:")
print(f"Marks = {model.coef_[0]:.2f} * Study Hours + {model.intercept_:.2f}")

# Step 8: Make predictions on new examples
new_hours = [[5], [7], [10]]
predictions = model.predict(new_hours)

# Step 9: Display predictions
for hours, marks in zip(new_hours, predictions):
    print(f"Study Hours: {hours[0]} → Predicted Marks: {marks:.1f}")

# Step 10: Visualize regression line
plt.scatter(df['study_hours'], df['exam_marks'], label="Actual Data")
plt.plot(df['study_hours'], model.predict(X), color='red', label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.title("Linear Regression: Study Hours vs Exam Marks")
plt.legend()
plt.show()
#-----------------------------------------------------


##########    Example 2: Gym Attendance vs Weight Loss   ############

# Step 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 2: Create the dataset
data = {
    'gym_days': [5, 10, 15, 20],
    'weight_loss_kg': [0.8, 1.6, 2.4, 3.2]
}

df = pd.DataFrame(data)

# Step 3: Explore the data
print("Dataset:")
print(df)
print("\nBasic Statistics:")
print(df.describe())

# Step 4: Visualize the relationship
plt.scatter(df['gym_days'], df['weight_loss_kg'])
plt.xlabel("Gym Days per Month")
plt.ylabel("Weight Loss (kg)")
plt.title("Gym Days vs Weight Loss")
plt.show()

# Step 5: Prepare data for training
X = df[['gym_days']]        # Feature (2D)
y = df['weight_loss_kg']    # Target

# Step 6: Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 7: Model parameters
print("\nModel Equation:")
print(f"Weight Loss = {model.coef_[0]:.2f} * Gym Days + {model.intercept_:.2f}")

# Step 8: Make predictions on new data
new_days = [[12], [18], [25]]
predictions = model.predict(new_days)

# Step 9: Display predictions
for days, loss in zip(new_days, predictions):
    print(f"Gym Days: {days[0]} → Predicted Weight Loss: {loss:.2f} kg")

# Step 10: Visualize regression line
plt.scatter(df['gym_days'], df['weight_loss_kg'], label="Actual Data")
plt.plot(df['gym_days'], model.predict(X), color='red', label="Regression Line")
plt.xlabel("Gym Days per Month")
plt.ylabel("Weight Loss (kg)")
plt.title("Linear Regression: Gym Days vs Weight Loss")
plt.legend()
plt.show()
