def apply_cwola_regions(
    df,
    sr_low,
    sr_high,
    sb_left_low,
    sb_left_high,
    sb_right_low,
    sb_right_high,
):
    sr_mask = (df["mjj"] >= sr_low) & (df["mjj"] <= sr_high)
    sb_mask = (
        ((df["mjj"] >= sb_left_low) & (df["mjj"] <= sb_left_high)) |
        ((df["mjj"] >= sb_right_low) & (df["mjj"] <= sb_right_high))
    )

    df_sr = df[sr_mask].copy()
    df_sb = df[sb_mask].copy()

    df_sr["cwola_target"] = 1
    df_sb["cwola_target"] = 0

    return df_sr, df_sb
