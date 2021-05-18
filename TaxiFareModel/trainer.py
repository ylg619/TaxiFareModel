from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from TaxiFareModel.encoder import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data
from sklearn.model_selection import train_test_split


class Trainer():

    def __init__(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc',
                               TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")
        self.pipeline = Pipeline([('preproc', preproc_pipe),
                                  ('linear_model', LinearRegression())])
        return self

    def train(self):
        '''returns a trained pipelined model'''
        self.pipeline.fit(self.X_train, self.y_train)
        return self

    def run(self):
        '''returns a trained pipelined model'''
        self.set_pipeline().train()
        return self

    def evaluate(self,X_test, y_test):
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

if __name__ == '__main__':
    df = get_data()

    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    train = Trainer()
    # build pipeline
    pipeline = train.set_pipeline()

    # train the pipeline
    train.train(X_train, y_train, pipeline)

    # evaluate the pipeline
    rmse = train.evaluate(X_val, y_val, pipeline)
