"""从合并后的用电和天气数据中按 Unix 秒时间戳提取时间段。"""

from pathlib import Path

import pandas as pd


def slice_power_weather(
    data_path="data/power_weather.csv",
    output_path="data/slice_power_weather.csv",
    start_unix=None,
    end_unix=None,
):
    """
    在已有的 power_weather.csv 上按 Unix 时间戳进行时间切片。

    参数
    ----
    data_path : str
        原始合并数据文件路径（默认 data/power_weather.csv）
    output_path : str
        切片后保存的文件路径
    start_unix : int | float | None
        起始 Unix 时间戳（单位为秒）。为 None 则不限制下界。
    end_unix : int | float | None
        结束 Unix 时间戳（单位为秒）。为 None 则不限制上界。
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")

    df = pd.read_csv(data_path)

    if "Datetime" not in df.columns:
        raise ValueError("数据中没有 'Datetime' 列，无法按时间切片。")

    df["Datetime"] = pd.to_datetime(df["Datetime"])

    start_time = pd.to_datetime(start_unix, unit="s") if start_unix is not None else None
    end_time = pd.to_datetime(end_unix, unit="s") if end_unix is not None else None

    mask = pd.Series(True, index=df.index)
    if start_time is not None:
        mask &= df["Datetime"] >= start_time
    if end_time is not None:
        mask &= df["Datetime"] <= end_time

    sliced_df = df.loc[mask].reset_index(drop=True)

    print("原始数据时间范围:", df["Datetime"].min(), "~", df["Datetime"].max())
    if start_time is not None or end_time is not None:
        print("切片条件(时间):", start_time, "~", end_time)
    print("切片后时间范围:", sliced_df["Datetime"].min(), "~", sliced_df["Datetime"].max())
    print("切片后数据形状:", sliced_df.shape)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sliced_df.to_csv(output_path, index=False)
    print(f"✅ 已保存切片结果到: {output_path}")


if __name__ == "__main__":
    # 默认提取 2016-07-01 至 2016-07-07（含边界）。
    start_unix = int(pd.Timestamp("2016-07-01 00:00:00").timestamp())
    end_unix = int(pd.Timestamp("2016-07-07 23:59:00").timestamp())

    slice_power_weather(
        start_unix=start_unix,
        end_unix=end_unix,
    )
