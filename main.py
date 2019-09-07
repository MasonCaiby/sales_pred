import pandas as pd
from median import make_dept_weekly_median, make_dept_monthly_median, predict_from_median
from prev_year import make_last_record, add_prev_year


class TrainData(object):
    def __init__(self, train_file, features_file, stores_file):
        self.train = pd.read_csv(train_file)
        self.features = pd.read_csv(features_file)
        self.stores = pd.read_csv(stores_file)

        self.week_means = None
        self.month_means = None
        self.prev_year = None

        self.concat_clean_dfs()
        self.make_training_dfs()

    def concat_clean_dfs(self):
        temp = self.train.merge(self.stores, on="Store").merge(self.features, on=["Store", "Date"])
        temp.drop("IsHoliday_y", axis=1, inplace=True)
        temp.drop("Size", axis=1, inplace=True)
        temp.drop("Temperature", axis=1, inplace=True)
        temp.drop("Type", axis=1, inplace=True)
        temp.fillna(0, inplace=True)

        temp["Date"] = pd.to_datetime(temp.Date)
        temp["year"] = temp.Date.dt.year
        temp["week"] = temp.Date.dt.week
        temp["month"] = temp.Date.dt.month

        self.train = temp

    def make_training_dfs(self):
        self.week_means = make_dept_weekly_median(self.train)
        self.prev_year = make_last_record(self.train)
        self.month_means = make_dept_monthly_median(self.prev_year)

    def add_features(self):
        self.train = predict_from_median(self.week_means, self.train, "week")
        self.train = predict_from_median(self.month_means, self.train, "month")
        self.train = add_prev_year(self.prev_year, self.train)


class TestData(object):
    def __init__(self, test_file, train_data):
        self.test = pd.read_csv(test_file)
        self.TrainData = train_data

        self.make_test()

    def make_test(self):
        self.test.Date = pd.to_datetime(self.test.Date)
        self.test["week"] = self.test.Date.dt.week
        self.test["month"] = self.test.Date.dt.month
        self.test["year"] = self.test.Date.dt.year
        self.test.Date = self.test["Date"].dt.date

        self.test["Weekly_Sales"] = 0
        self.test["Id"] = self.test.Store.map(str) + '_' + self.test.Dept.map(str) + '_' + self.test.Date.map(str)

    def feature_engineering(self):
        self.test = predict_from_median(self.TrainData.week_means, self.test, "week")
        self.test = predict_from_median(self.TrainData.month_means, self.test, "month")
        self.test = add_prev_year(self.TrainData.prev_year, self.test)


    def make_submission_file(self, columns):
        self.test["Weekly_Sales"] = self.test[columns].mean(axis=1)
        self.test["Weekly_Sales"].fillna(0, inplace=True)
        self.test[["Id", 'Weekly_Sales']].to_csv("submission.csv", index=False)


if __name__ == "__main__":
    training = TrainData("train.csv", "features.csv", "stores.csv")
    training.add_features()
    test = TestData("test.csv", training)
    test.feature_engineering()
    test.make_submission_file(["Weekly_Sales_month_means"])




