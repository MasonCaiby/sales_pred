""" Just means return a score of 3462.43912     3339.46038"""

def make_dept_weekly_mean(train_df):
    """takes a full training df and returns a df with the store, dept, week, and avg weekly sales"""

    gb = train_df.groupby(["Store", "Dept", "week"]).mean()["Weekly_Sales"]
    return gb


def predict_from_means(means_gb, test_df):
    return test_df.join(means_gb, on=["Store", "Dept", "week"], rsuffix="_means")

