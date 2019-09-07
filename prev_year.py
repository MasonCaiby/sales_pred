# Predict the week's sales based on the previous year


def make_last_record(train_df):
    gb_idx = train_df.groupby(["Store", "Dept", "week"])['year'] \
                 .transform(max) == train_df['year']
    prev_year = train_df[gb_idx][["Store", "Dept", "week", 'Weekly_Sales', "month"]]
    prev_year = prev_year.groupby(["Store", "Dept", "week"]).mean()
    return prev_year


def add_prev_year(prev_year_gb, target_df):
    return target_df.join(prev_year_gb, on=["Store", "Dept", "week"], rsuffix="_pyear")

