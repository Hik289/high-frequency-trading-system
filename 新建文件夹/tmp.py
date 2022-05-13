if item_1 == "hlr":
        df_1 = dt_data["day"]["high"] / dt_data["day"]["low"] - 1
    elif item_1 == "ocr":
        df_1 = dt_data["day"]["open"] / dt_data["day"]["close"] - 1
    else:
        df_1 = dt_data["day"][item_1].pct_change(1)