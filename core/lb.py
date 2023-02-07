# 长桥数据源
import time
from decimal import Decimal

import pandas as pd
import cufflinks as cf
import matplotlib.pyplot as plt
from longbridge.openapi import Config, TradeContext, QuoteContext, Period, AdjustType, Candlestick


cf.set_config_file(offline=True)

"""
输入正股的涨跌幅，来估算衍生品的收益

Machine Learning 主流程
1. 数据源准备
    数据源需要包含多个特征 X，需要有一个标签 y
2. 数据源处理
    将数据源分为训练集和测试集，一般为 9:1
3. 训练模型
    选择合适的模型，如 KNN、决策树、向量积
    将训练集数据带入模型进行拟合训练
    训练完成之后，可以将分别测试模型在训练集和测试集的准确率
    如果训练集分数很高，测试集分数很低，则为过拟合
    如果训练集分数很低，测试集分数很高，则为欠拟合
4. 选择合适的模型参数
"""


def timer(func):
    def inner(*args, **kw):
        start = time.time()
        print(f"执行函数: {func.__name__}")
        res = func(*args, **kw)
        print(f"执行时间: {round(time.time() - start, 4)} s")
        print(f"执行结果: {res}")
        return res
    return inner


class LB(object):
    def __init__(self):
        self.config = Config.from_env()
        self.quote = QuoteContext(self.config)
        self.trade = TradeContext(self.config)
        # 显示所有列
        pd.set_option('display.max_columns', None)
        # 显示所有行
        # pd.set_option('display.max_rows', None)

    def k_line(self, symbol: str = "700.HK", period: Period = Period.Day, count: int = 100,
               adjust_type: AdjustType = AdjustType.NoAdjust) -> pd.DataFrame:
        """获取标的的 k 线"""
        items = self.quote.candlesticks(symbol=symbol, period=period, count=count, adjust_type=adjust_type)
        keys = list(filter(lambda i: "_" not in i, dir(Candlestick)))
        data = {}
        for item in items:
            for key in keys:
                x = getattr(item, key)
                if isinstance(x, Decimal):
                    x = float(x)
                if data.get(key):
                    data[key].append(x)
                else:
                    data[key] = [x]
        df = pd.DataFrame(data)
        df["close_diff"] = df["close"].diff()
        df["last_close"] = df["close"] - df["close_diff"]
        df["open_diff"] = df["open"] - df["last_close"]
        df["open_rate"] = df["open_diff"] / df["last_close"]
        df["close_rate"] = df["close_diff"] / df["last_close"]
        df["high_rate"] = df["high"] / df["last_close"] - 1
        df["low_rate"] = df["low"] / df["last_close"] - 1
        # 转为百分比
        for column in ["open_rate", "close_rate", "high_rate", "low_rate"]:
            df[column] = df[column].fillna(0).astype(float).map("{:.2%}".format)
        # 列名排序
        df = df[["timestamp", "last_close", "open", "open_diff", "open_rate",
                 "close", "close_diff", "close_rate", "high", "high_rate", "low", "low_rate", "volume"]]
        return df

    @staticmethod
    def show_k_line(df: pd.DataFrame):
        qf = cf.QuantFig(df, title="TX Price", name="TX")
        qf.add_volume()
        qf.add_sma(periods=5, column="close", color="red")
        qf.add_ema(periods=5, color="green")
        qf.iplot()

    def intraday(self, symbol: str = "700.HK"):
        resp = self.quote.intraday(symbol)
        return resp

    def get_diff(self, symbol: str = "700.HK", diff: float = 0.02):
        data = self.intraday(symbol)
        close_price = data[-1].price
        price = close_price * Decimal(1 + diff)
        for item in data:
            if abs(price - item.price) < 0.2:
                return item
        raise Exception(f"未找到这个范围之内的数据")

    def get_wt_price(self, m_symbol: str = "700.HK", s_symbols: list = ("50600.HK", "28138.HK", "50132.HK")):
        lst = []
        timestamp = self.get_diff(m_symbol).timestamp
        for symbol in s_symbols:
            items = self.intraday(symbol)
            last_price = items[-1].price
            for item in items:
                if item.timestamp == timestamp:
                    lst.append({
                        "symbol": symbol,
                        "price": item.price,
                        "timestamp": item.timestamp,
                        "rate": round(item.price / last_price - 1, 2)
                    })
                    break
        print(lst)
        return lst

    @staticmethod
    def show(data: pd.DataFrame):
        # 设置画布的尺寸为10*5
        plt.figure(figsize=(10, 5))
        # 使用折线图绘制出每天的收盘价
        data['close'].plot(linewidth=2, color='k', grid=True)
        # 如果当天股价上涨，标出卖出信号，用倒三角表示
        # plt.scatter(data['close'].loc[data.signal == 1].index,
        #             data['close'][data.signal == 1],
        #             marker='v', s=80, c='g')
        # # 如果当天股价下跌给出买入信号，用正三角表示
        # plt.scatter(data['close'].loc[data.signal == 0].index,
        #             data['close'][data.signal == 0],
        #             marker='^', s=80, c='r')
        # 将图像进行展示
        plt.show()

    def method(self, data: pd.DataFrame, m):
        """
        策略，策略决定什么时候买入，什么时候卖出，买入、卖出的仓位是多少
        :param data:
        :param m:
        :return:
            bs: -1 卖出 0 不操作 1 买入
            change_quantity: 交易数量
            quantity: 持仓数量
        """
        ...

    @staticmethod
    def backtest(data: pd.DataFrame, init_cash: int = 100000):
        """
        回测
        :param data:
        :param init_cash: 初始资金
        :return:
            stock_value: 持仓市值 = price * quantity
            cash_value: 现金 = 上日现金余额 - price * change_quantity
            total_value: 总资产 = 持仓市值 + 现金
        """
        data["stock_value"] = data["close"] * data["quantity"]          # 当前股票市值
        data["cash_value"] = init_cash - (data["close"] * data["change_quantity"]).cumsum(skipna=True)   # 当前现金
        data["total_value"] = data["stock_value"] + data["cash_value"]  # 当前总资产
        return data


@timer
def ut():
    lb = LB()
    lb.k_line()
    # lb.show_k_line(data)
    # lb.show(data)
    # lb.intraday()
    # lb.get_diff(0.02)
    # lb.intraday("50600.HK")
    # lb.get_wt_price()


if __name__ == '__main__':
    ut()
