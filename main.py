import pandas as pd
from means import make_dept_weekly_mean, predict_from_means


class TrainData(object):
    def __init__(self, train_file, features_file, stores_file):
        self.train = pd.read_csv(train_file)
        self.features = pd.read_csv(features_file)
        self.stores = pd.read_csv(stores_file)

        self.x = None
        self.y = None

        self.means = None

    def concat_clean_dfs(self):
        temp = self.train.merge(self.stores, on="Store").merge(self.features, on=["Store", "Date"])
        temp.drop("IsHoliday_y", axis=1, inplace=True)
        temp.drop("Size", axis=1, inplace=True)
        temp.drop("Temperature", axis=1, inplace=True)
        temp.drop("Type", axis=1, inplace=True)
        temp.fillna(0, inplace=True)

        temp["Date"] = pd.to_datetime(temp.Date)
        temp["week"] = temp.Date.dt.week

        self.train = temp

    def make_training_dfs(self):
        self.means = make_dept_weekly_mean(self.train)


class TestData(object):
    def __init__(self, test_file, train_data):
        self.test = pd.read_csv(test_file)
        self.TrainData = train_data

    def make_test_key(self):
        self.test["key"] = self.test.Store.map(str) + '_' + self.test.Dept.map(str) + '_' + self.test.Date.map(str)

    def add_means(self):
        self.test = predict_from_means(self.TrainData.means, self.test)

    def make_submission_file(self, columns):
        self.test["Weekly_Sales"] = self.test[columns].mean(axis=1)
        self.test[["key", 'Weekly_Sales']].to_csv("submission.csv")


if __name__ == "__main__":
    training = TrainData("train.csv", "features.csv", "stores.csv")
    test = TestData("test.csv", training)
    test.make_test_key()
    test.add_means()
    test.make_submission_file(["Weekly_Sales_means"])




