""" Just means return a score of 3462.43912     3339.46038"""

def make_dept_weekly_median(train_df):
    """takes a full training df and returns a df with the store, dept, week, and avg weekly sales"""

    gb = train_df.groupby(["Store", "Dept", "week"]).median()["Weekly_Sales"]
    return gb

def make_dept_monthly_median(train_df):
    """takes a full training df and returns a df with the store, dept, week, and avg weekly sales"""

    gb = train_df.groupby(["Store", "Dept", "month"]).median()["Weekly_Sales"]
    return gb


def predict_from_median(means_gb, target_df, time):
    return target_df.join(means_gb, on=["Store", "Dept", time], rsuffix= "_"+time+"_means")

