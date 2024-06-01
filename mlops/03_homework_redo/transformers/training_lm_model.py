from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import mlflow
import pickle

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test



EXPERIMENT_NAME = "03_homework_lr"

# mlflow.set_tracking_uri("http://localhost:8080")
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri("http://host.docker.internal:8080")

mlflow.set_experiment(EXPERIMENT_NAME)
# mlflow.sklearn.autolog()

vectorizer_path = "dict_vectorizer.pkl"

@transformer
def transform(df, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, I str, etc.)
    """
    # Specify your transformation logic here

    categorical = ['PULocationID', 'DOLocationID']
    # numerical = ['trip_distance']   

    columns_with_id = [col for col in df.columns if 'LocationID' in col]

    df[columns_with_id] = df[columns_with_id].astype(str)

    train_dicts = df[categorical].to_dict(orient='records')

    # mlflow.autolog()

    with mlflow.start_run():

        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts)

        target = 'duration'
        y_train = df[target].values

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        mlflow.sklearn.log_model(lr, artifact_path="models")
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")


        y_pred = lr.predict(X_train)

          # Save the DictVectorizer using pickle and log it as an artifact
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(dv, f)

        mlflow.log_artifact(vectorizer_path)


    #mean_squared_error(y_train, y_pred, squared=False)

    print(lr.intercept_)

    #data = df
    #print(intercept_)

    return dv, lr


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'