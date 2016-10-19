import tempfile
import urllib

global df_train
global df_test
def get_files():
    """
    """
    train_file = tempfile.NamedTemporaryFile()
    test_file = tempfile.NamedTemporaryFile()
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)
    return train_file, test_file

import pandas as pd

def get_data(train_file, test_file):
    COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num","marital_status", "occupation", "relationship", "race", "gender","capital_gain", "capital_loss", "hours_per_week", "native_country","income_bracket"]
    df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
    df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
    LABEL_COLUMN = "label"
    df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    return df_train, df_test

import tensorflow as tf
def input_fn(df):
    """
    """
    LABEL_COLUMN = "label"
    CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "gender", "native_country"]
    CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    #Create a directory mapping from each continous feature column name (k) to the values of that column stored in a Tensor
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    #Create a directory mapping from each categorical feature column name (k) to the values of that column stored in a tf.SparseTensor
    categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)], values=df[k].values, shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}
    #Merge the two dictionaries into one
    feature_cols = continuous_cols.copy()
    feature_cols.update(categorical_cols)
    # feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    #Convert the label column into a constant tensor
    labels = tf.constant(df[LABEL_COLUMN].values)
    #Returnthe feature columns and the labels
    return feature_cols, labels

def train_input_fn():
    return input_fn(df_train)

def test_input_fn():
    return input_fn(df_test)

def get_model():
    #convert categorical data to sparse tensors
    gender = tf.contrib.layers.sparse_column_with_keys(column_name='gender', keys=['male','female'])
    education = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='education', hash_bucket_size=1000)
    race = tf.contrib.layers.sparse_column_with_keys(column_name='race', keys=["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"])
    marital_status = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='marital_status', hash_bucket_size=1000)
    relationship = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='relationship', hash_bucket_size=1000)
    workclass = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='workclass', hash_bucket_size=1000)
    occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
    native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)
    #put real values in tensors
    age = tf.contrib.layers.real_valued_column('age')
    education_num = tf.contrib.layers.real_valued_column('education_num')
    capital_gain = tf.contrib.layers.real_valued_column('capital_gain')
    capital_loss = tf.contrib.layers.real_valued_column('capital_loss')
    hours_per_week = tf.contrib.layers.real_valued_column('hours_per_week')
    #bucketized real feature
    age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    #column intersection
    education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))
    age_buckets_x_race_x_occupation = tf.contrib.layers.crossed_column([age_buckets,race, occupation], hash_bucket_size=int(1e6))
    model_dir = tempfile.mkdtemp()
    m = tf.contrib.learn.LinearClassifier(feature_columns=[gender, native_country, education, occupation, workclass, marital_status, race, age_buckets,education_x_occupation, age_buckets_x_race_x_occupation], model_dir=model_dir)
    return m

def train_model(m):
    m.fit(input_fn=train_input_fn, steps=200)
    return m

def eval_model(m):
    results = m.evaluate(input_fn=test_input_fn, steps=1)
    for key in sorted(results):
        print ("%s: %s" % (key, results[key]))

#NOTE: I am not using a main function as there is a problem with passing df_train and df_test to the train and eval model functions
train_file, test_file = get_files()
print("files downloaded")
df_train, df_test = get_data(train_file, test_file)

print("dataframe made")
# print(df_train)
model = get_model()
print("model made")
model = train_model(model)
eval_model(model)
