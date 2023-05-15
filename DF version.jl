using CSV
using DataFrames
using TextAnalysis

# Load the training data
train_df = CSV.read("training_data.csv")

# Create a Bag-of-Words representation of the text data
textdata = train_df.text
vectorizer = BowVectorizer(textdata)
X_train = fit_transform!(vectorizer, textdata)

# Train a Naive Bayes classifier
y_train = train_df.labels
nb_classifier = NaiveBayesClassifier(labels(y_train), X_train)

# Load the test data
test_df = CSV.read("test_data.csv")

# Transform the test data into the same Bag-of-Words representation
textdata_test = test_df.text
X_test = transform(vectorizer, textdata_test)

# Make predictions on the test data
y_pred = predict(nb_classifier, X_test)

# Output the predicted labels
test_df.labels_predicted = y_pred
CSV.write("predictions.csv", test_df)
