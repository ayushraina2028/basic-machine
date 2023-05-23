from flask import Flask, render_template, request
import pandas as pd
import numpy as np
app = Flask(__name__)

# Reading dataset in global scope
df = pd.read_csv("winequalityN.csv")

# This is the home page
@app.route("/")
def home():
    return render_template("home.html")

# This is the page where we will load the dataset
@app.route("/load", methods = ["GET", "POST"])
def load():
    if request.method == "POST" and "submit_button" in request.form:
        return render_template("results.html")

# This is the page where we will check the shape of the dataset
@app.route("/shape", methods = ["GET", "POST"])
def shape():
    if request.method == "POST" and "check_shape" in request.form:
        rc = df.shape
        shape = f"Rows: {rc[0]}, Columns: {rc[1]}"
        return render_template("results2.html", shape = shape)

# This is the page where we will check the column names of the dataset
@app.route("/column_names", methods = ["GET", "POST"])
def column_names_fun():
    if request.method == "POST" and "check_column_names" in request.form:
        col_names = df.columns
        arr = list(col_names)
        return render_template("columnr.html", colnames = arr)

# This is the page where we will check the missing values of the dataset
@app.route("/missing_values", methods = ["GET", "POST"])
def miss_val():
    if request.method == "POST" and "missing_values" in request.form:
        values = list(df.isnull().sum())
        miss_values = []
        arr = list(df.columns)
        for i in range(len(values)):
            if values[i] != 0:
                miss_values.append((arr[i], values[i])) # to get a list of tuples of feature and number of missing values
            
        return render_template("results4.html", missing_values = miss_values)

# This is the page where we will handle the missing values of the dataset
@app.route("/handle_miss_value", methods = ["GET", "POST"])
def handle_mis():
    df = pd.read_csv("winequalityN.csv")
    if request.method == "POST" and "handling_values" in request.form:
        # case 1 : when there are too many rows in dataset as compared to columns
        # we will proceed by deleting them
        
        rc = df.shape
        if rc[0] - df.dropna().shape[0] < 0.05*rc[0]:  # if the number of rows with missing values is less than 5% of total rows
            df = df.dropna()
            return render_template("results5.html", more_rows = "Missing Values have been dropped by dropping rows")
        else:
            # Here we will replace the missing values with the mean if outliers are not present
            # We dont need to find outliers as we will just see difference between mean and median
            values = list(df.isnull().sum())
            arr = list(df.columns)
            for i in range(len(values)):
                if values[i] != 0 and (arr[i].dtype == "float64" or arr[i].dtype == "int64"):
                    if df[arr[i]].mean() - df[arr[i]].median() < 2:
                        df[arr[i]] = df[arr[i]].fillna(df[arr[i]].mean())
                        return render_template("results5.html", no_outliers = "Missing Values have been replaced with mean because no outliers were present")
                    else:
                        df[arr[i]] = df[arr[i]].fillna(df[arr[i]].median())
                        return render_template("results5.html", outliers = "Missing Values have been replaced with median because outliers were present")
   
    return render_template("results5.html", no_missing_values = "No Missing Values were present")

# here we will count how many categorical features are present
@app.route("/categorical_feature", methods = ["GET", "POST"])
def handle():
        if request.method == "POST" and "cat_feature" in request.form:
            cat_arr = df.select_dtypes(include=['object']).columns.tolist() # list of Categorical Features
            if len(cat_arr) == 0:
                return render_template("results6.html", x = "No Categorical Features were present")
            else:
                return render_template("results6.html", x = f"{len(cat_arr)} Categorical Features were present and these are {cat_arr}")

# Now we will do OrdinalEncoding on the categorical features
@app.route("/ordinal_encoding", methods = ["GET", "POST"])
def ord_enc():
    if request.method == "POST" and "encode" in request.form:
        from sklearn.preprocessing import OrdinalEncoder
        encoder = OrdinalEncoder()
        cat_arr = df.select_dtypes(include=['object']).columns.tolist()
        for i in cat_arr:
            df[i] = encoder.fit_transform(df[[i]]).astype(int)
        return render_template("results7.html", oe = "Ordinal Encoding has been done on the Categorical Features")


# And we are not adding any king of upsampling and downsampling techniques due to lack of time
# Here we are making an assumption that last feature is the target feature

# This Sets Dataset target variable as Wine Quality
@app.route("/set_target", methods = ["GET", "POST"])
def taregt():
    if request.method == "POST" and "target" in request.form:
        global X, y
        # Independent Features
        X = df.drop(df.columns[-1], axis = 1)

        # Target Feature
        y = df[df.columns[-1]]

        return render_template("results8.html", target = "Target Feature has been set as Wine Quality")

# Now Feature Selection, checking if there are any two independent features which are highly correlated
# we will take only one of them and drop the other one, we keep threshold as 0.8
@app.route("/feature_selection", methods = ["GET", "POST"])
def feature_select():
    
    if request.method == "POST" and "f_select" in request.form:
        X = df.drop(df.columns[-1], axis = 1)
        dataframe = X
        threshold = 0.8
        
        # Correlation matrix
        corr_matrix = dataframe.corr().abs() # absolute value of correlation matrix 
        
        # Creating a set to hold the correlated features
        corr_features = set()
        
        # Looping through Each Feature
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                
                # Check if the correlation between two features is greater than threshold
                if corr_matrix.iloc[i, j] >= threshold:
                    colname = corr_matrix.columns[i] 
                    corr_features.add(colname)  # we need only 1 feature out of the two highly correlated features
        
        # Dropping the correlated features
        X = X.drop(labels = corr_features, axis = 1)
        X = X.dropna()
        return render_template("results9.html", feature = "Feature Selection has been done")
        
# Here we will import standard scaler 
@app.route("/standard_scaler", methods = ["GET", "POST"])
def scaler():
    global scaler
    if request.method == "POST" and "std_scaler" in request.form:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        return render_template("results10.html", scaler = "Standard Scaler has been imported")


# Here we will split the dataset into train and test
# Make sure to first set the target feature and then split the dataset
@app.route("/train_test_split", methods = ["GET", "POST"])
def train_test():
    global X_test, X_train
    if request.method == "POST" and "split" in request.form:
        global X_train, X_test, y_train, y_test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        trained_size = X_train.shape
        test_size = X_test.shape
        
    return render_template("results11.html", trained = "Dataset has been split into train and test", trained_2 = "Now you can train the model"
                           , trained_3 = f"Train Size is {trained_size} and Test Size is {test_size}")
    
# now we will do standard scaling on the dataset
@app.route("/scaling", methods = ["GET", "POST"])  
def transform():
    global X_train_scaled, X_test_scaled
    if request.method == "POST" and "scale" in request.form:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=np.nanmean(X_train_scaled))
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=np.nanmean(X_test_scaled))
        
    return render_template("results12.html", scaled = "Standard Scaling has been done on the dataset")


#1 Linear Regression Training
@app.route("/linear_regression", methods = ["GET", "POST"])
def lin_reg():
    global regressor
    if request.method == "POST" and "linear" in request.form:
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train_scaled, y_train)
    return render_template("results13.html", linear_reg = "Linear Regression Model has been trained")
        
#2 Ridge Regression Training
@app.route("/ridge_regression", methods = ["GET", "POST"])   
def ridge_reg():
    global ridge_regressor
    if request.method == "POST" and "ridge" in request.form:
        from sklearn.linear_model import Ridge
        ridge_regressor = Ridge()
        ridge_regressor.fit(X_train_scaled, y_train)
    return render_template("results14.html", ridge_reg = "Ridge Regression Model has been trained")     
        
#3 Lasso Regression Training
@app.route("/lasso_regression", methods = ["GET", "POST"])
def lasso_reg():
    global lasso_regressor
    if request.method == "POST" and "lasso" in request.form:
        from sklearn.linear_model import Lasso
        lasso_regressor = Lasso()
        lasso_regressor.fit(X_train_scaled, y_train)
    return render_template("results15.html", lasso_reg = "Lasso Regression Model has been trained")

#4 Elastic Net Regression Training
@app.route("/elastic_net_regression", methods = ["GET", "POST"])
def elastic_reg():
    global elastic_regressor
    if request.method == "POST" and "elastic" in request.form:
        from sklearn.linear_model import ElasticNet
        elastic_regressor = ElasticNet()
        elastic_regressor.fit(X_train_scaled, y_train)
    return render_template("results16.html", elastic_reg = "Elastic Net Regression Model has been trained")

#5 Logistic Regression Training
@app.route("/logistic_regression", methods = ["GET", "POST"])
def log_reg():
    global log_regressor
    if request.method == "POST" and "logistic" in request.form:
        from sklearn.linear_model import LogisticRegression
        log_regressor = LogisticRegression()
        log_regressor.fit(X_train_scaled, y_train)
    return render_template("results17.html", log_reg = "Logistic Regression Model has been trained")

#6 Decision Tree Classifier Training
@app.route("/decision_tree_classifier", methods = ["GET", "POST"])
def dec_tree():
    global dec_tree_classifier
    if request.method == "POST" and "decision" in request.form:
        from sklearn.tree import DecisionTreeClassifier
        dec_tree_classifier = DecisionTreeClassifier()
        dec_tree_classifier.fit(X_train_scaled, y_train)
    return render_template("results18.html", decs_tree = "Decision Tree Classifier Model has been trained")

#7 Decision Tree Regressor Training
@app.route("/decision_tree_regressor", methods = ["GET", "POST"])
def dec_tree_reg():
    global dec_tree_regressor
    if request.method == "POST" and "decision_regressor" in request.form:
        from sklearn.tree import DecisionTreeRegressor
        dec_tree_regressor = DecisionTreeRegressor()
        dec_tree_regressor.fit(X_train_scaled, y_train)
    return render_template("results19.html", decs_tree_reg = "Decision Tree Regressor Model has been trained")

#8 Support Vector Classifier Training
@app.route("/support_vector_classifier", methods = ["GET", "POST"])
def svc():
    global svc_classifier
    if request.method == "POST" and "support" in request.form:
        from sklearn.svm import SVC
        svc_classifier = SVC()
        svc_classifier.fit(X_train_scaled, y_train)
    return render_template("results20.html", svc = "Support Vector Classifier Model has been trained")

#9 Support Vector Regressor Training
@app.route("/support_vector_regressor", methods = ["GET", "POST"])
def svr():
    global svr_regressor
    if request.method == "POST" and "support_regressor" in request.form:
        from sklearn.svm import SVR
        svr_regressor = SVR()
        svr_regressor.fit(X_train_scaled, y_train)
    return render_template("results21.html", svr = "Support Vector Regressor Model has been trained")

#10 Naive Bayes Classifier Training
@app.route("/naive_bayes_classifier", methods = ["GET", "POST"])
def naive():
    global naive_classifier
    if request.method == "POST" and "naive" in request.form:
        from sklearn.naive_bayes import GaussianNB
        naive_classifier = GaussianNB()
        naive_classifier.fit(X_train_scaled, y_train)
    return render_template("results22.html", naive = "Naive Bayes Classifier Model has been trained")

#11 Random Forest Classifier Training
@app.route("/random_forest_classifier", methods = ["GET", "POST"])
def random():
    global random_classifier
    if request.method == "POST" and "random" in request.form:
        from sklearn.ensemble import RandomForestClassifier
        random_classifier = RandomForestClassifier()
        random_classifier.fit(X_train_scaled, y_train)
    return render_template("results23.html", random = "Random Forest Classifier Model has been trained")

#12 AdaBoost Classifier Training
@app.route("/adaboost_classifier", methods = ["GET", "POST"])
def adaboost():
    global adaboost_classifier
    if request.method == "POST" and "adaboost" in request.form:
        from sklearn.ensemble import AdaBoostClassifier
        adaboost_classifier = AdaBoostClassifier()
        adaboost_classifier.fit(X_train_scaled, y_train)
    return render_template("results24.html", adaboost = "AdaBoost Classifier Model has been trained")

# All Predictions at one place
@app.route("/prediction", methods = ["GET", "POST"])
def predict():
    global y_pred_linear, y_pred_lasso, y_pred_adaboost, y_pred_dec_tree, y_pred_dec_tree_reg, y_pred_elastic, y_pred_log, y_pred_random, y_pred_ridge,y_pred_support, y_pred_support_reg, y_pred_naive
    
    

    if request.method == "POST" and "predict" in request.form:
        
        #1 Linear Regression Prediction
        y_pred_linear = regressor.predict(X_test_scaled)
        
        #2 Ridge Regression Prediction
        y_pred_ridge = ridge_regressor.predict(X_test_scaled)
        
        #3 Lasso Regression Prediction
        y_pred_lasso = lasso_regressor.predict(X_test_scaled)
        
        #4 Elastic Net Regression Prediction
        y_pred_elastic = elastic_regressor.predict(X_test_scaled)
        
        #5 Logistic Regression Prediction
        y_pred_log = log_regressor.predict(X_test_scaled)
        
        #6 Decision Tree Classifier Prediction
        y_pred_dec_tree = dec_tree_classifier.predict(X_test_scaled)
        
        #7 Decision Tree Regressor Prediction
        y_pred_dec_tree_reg = dec_tree_regressor.predict(X_test_scaled)
        
        #8 Support Vector Classifier Prediction
        y_pred_support = svc_classifier.predict(X_test_scaled)
        
        #9 Support Vector Regressor Prediction
        y_pred_support_reg = svr_regressor.predict(X_test_scaled)
        
        #10 Naive Bayes Classifier Prediction
        y_pred_naive = naive_classifier.predict(X_test_scaled)
        
        #11 Random Forest Classifier Prediction
        y_pred_random = random_classifier.predict(X_test_scaled)
        
        #12 AdaBoost Classifier Prediction
        y_pred_adaboost = adaboost_classifier.predict(X_test_scaled)
        
        return render_template("results25.html",predict = "All Predictions have been done")

# All Accuracy Scores at one place
@app.route("/accuracy", methods = ["GET", "POST"])
def accuracy():
    
    if request.method == "POST" and "accuracy" in request.form:
        
        #1 Linear Regression Accuracy
        from sklearn.metrics import r2_score
        r2_linear = r2_score(y_test, y_pred_linear)
        
        #2 Ridge Regression Accuracy
        r2_ridge = r2_score(y_test, y_pred_ridge)
        
        #3 Lasso Regression Accuracy
        r2_lasso = r2_score(y_test, y_pred_lasso)
        
        #4 Elastic Net Regression Accuracy
        r2_elastic = r2_score(y_test, y_pred_elastic)
        
        #5 Logistic Regression Accuracy
        from sklearn.metrics import accuracy_score
        accuracy_log = accuracy_score(y_test, y_pred_log)
        
        #6 Decision Tree Classifier Accuracy
        accuracy_dec_tree = accuracy_score(y_test, y_pred_dec_tree)
        
        #7 Decision Tree Regressor Accuracy
        from sklearn.metrics import r2_score
        r2_dec_tree_reg = r2_score(y_test, y_pred_dec_tree_reg)
        
        #8 Support Vector Classifier Accuracy
        accuracy_support = accuracy_score(y_test, y_pred_support)
        
        #9 Support Vector Regressor Accuracy
        from sklearn.metrics import r2_score
        r2_support_reg = r2_score(y_test, y_pred_support_reg)
        
        #10 Naive Bayes Classifier Accuracy
        accuracy_naive = accuracy_score(y_test, y_pred_naive)
        
        #11 Random Forest Classifier Accuracy
        accuracy_random = accuracy_score(y_test, y_pred_random)
        
        #12 AdaBoost Classifier Accuracy
        accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
        
        max_accuracy = max(accuracy_log, accuracy_dec_tree, accuracy_support, accuracy_naive, accuracy_random, accuracy_adaboost)
        return render_template("results26.html",linea_reg = r2_linear, ridg_reg = r2_ridge, lass_reg = r2_lasso, elasti_reg = r2_elastic, lo_reg = accuracy_log, de_tree = accuracy_dec_tree, de_tree_reg = r2_dec_tree_reg, suppor_vec = accuracy_support, suppor_vec_reg = r2_support_reg, naiv = accuracy_naive, rando = accuracy_random, adaboos = accuracy_adaboost, x = max_accuracy)      

# Hyperparameter Tuning
# Find Best parameters that will fit all 12 models at once
# find parameters using RandomizedSearchCV
    

    
        
if __name__=="__main__":
    app.run(host="0.0.0.0")
