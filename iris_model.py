import tkinter as tk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Define the function for making predictions
def classify_iris():
    try:
        sepal_length = float(entry1.get())
        sepal_width = float(entry2.get())
        petal_length = float(entry3.get())
        petal_width = float(entry4.get())
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        result_text.set(f"Prediction: {iris.target_names[prediction][0]}")
    except ValueError:
        result_text.set("Invalid input. Please enter valid numbers.")

# Build the GUI
app = tk.Tk()
app.title("Iris Flower Classifier")

# Input labels and entry fields
tk.Label(app, text="Sepal Length (cm)").pack()
entry1 = tk.Entry(app)
entry1.pack()

tk.Label(app, text="Sepal Width (cm)").pack()
entry2 = tk.Entry(app)
entry2.pack()

tk.Label(app, text="Petal Length (cm)").pack()
entry3 = tk.Entry(app)
entry3.pack()

tk.Label(app, text="Petal Width (cm)").pack()
entry4 = tk.Entry(app)
entry4.pack()

# Button to classify the Iris
result_text = tk.StringVar()
tk.Button(app, text="Classify", command=classify_iris).pack()
tk.Label(app, textvariable=result_text).pack()

# Display the accuracy of the model
tk.Label(app, text=f"Model Accuracy: {accuracy:.2f}").pack()

# Run the app
app.mainloop()
