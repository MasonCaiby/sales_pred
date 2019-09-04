def make_dept_weekly_mean(train_df):
    """takes a full training df and returns a df with the store, dept, week, and avg weekly sales"""

    gb = train_df.groupby(["Store", "Dept", "week", "IsHoliday_x"]).mean()["Weekly_Sales"]
    return gb


def predict_from_means(means_gb, test_df):
    return test_df.join(means_gb, on=["Store", "Dept", "week"], rsuffix="_means")

