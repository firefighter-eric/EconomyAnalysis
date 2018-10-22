from data_process import *
import statsmodels.api as sm
import itertools
import matplotlib.pyplot as plt


class DataConfig:
    filename = "outdata201803_hz_monthly_1.csv"
    index_col = 'Dates'
    tag = 'CPI'
    time_step = 10


class ParaSearch:
    def __init__(self, data, pdq_range):
        self.data = data
        self.pdq, self.pdqs, self.n_search = self.initial_pdqs(pdq_range)
        self.results = self.train()
        self.i_min_aic = self.min_aic(self.results)
        self.result_best = self.results[self.i_min_aic]

    def initial_pdqs(self, pdq_range):
        p = d = q = range(pdq_range[0], pdq_range[1])
        pdq = tuple(itertools.product(p, d, q))
        pdqs = [(x[0], x[1], x[2], 12) for x in pdq]
        n_search = len(pdq) * len(pdqs)
        return pdq, pdqs, n_search

    def train(self):
        results = list()
        for pdq in self.pdq:
            for pdqs in self.pdqs:
                mod = sm.tsa.statespace.SARIMAX(self.data,
                                                order=pdq,
                                                seasonal_order=pdqs,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results.append(mod.fit())
                # plt.plot(results[i].fittedvalues, 'r')
        for i in range(self.n_search):
            print('{}\tpdq:{}\tpdqs:{}\taic:{}'.format(i,
                                                       self.pdq[i // len(self.pdq)],
                                                       self.pdqs[i % len(self.pdqs)],
                                                       results[i].aic))
        return results

    def min_aic(self, result):
        i_m = 0
        m = result[0].aic
        for i in range(len(result)):
            if m > result[i].aic:
                m = result[i].aic
                i_m = i
        return i_m


class ParaCal:
    def __init__(self, data, p_max, d_max, q_max):
        self.data_diff = self.diff(data, d_max)
        self.fig_acf = self.acf(self.data_diff[2], q_max)
        self.fig_pacf = self.pacf(self.data_diff[2], p_max)

    def diff(self, data, d_max):
        data_diff = [data]
        for i in range(d_max):
            data_diff.append(np.diff(data_diff[-1]))

        fig_diff = plt.figure()
        ax_diff = list()
        for i in range(d_max + 1):
            ax_diff.append(fig_diff.add_subplot(3, 4, i + 1))
            ax_diff[i].plot(data_diff[i])

        return data_diff

    def acf(self, data, q_max):
        fig_acf = sm.graphics.tsa.plot_acf(data, lags=q_max)
        return fig_acf

    def pacf(self, data, p_max):
        fig_pacf = sm.graphics.tsa.plot_pacf(data, lags=p_max)
        return fig_pacf


class Prediction:
    def __init__(self, data, steps):
        self.data = data
        self.steps = steps
        self.result = self.best_model()
        self.predict(self.result, steps)

    def best_model(self):
        mod = sm.tsa.statespace.SARIMAX(self.data,
                                        order=(7, 2, 1),
                                        # seasonal_order=(1, 1, 1, 12),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

        result = mod.fit()
        return result

    def predict(self, result, steps):
        pred_uc = result.get_forecast(steps=steps)
        pred_ci = pred_uc.conf_int()
        # ax = y.plot(label='observed', figsize=(20, 15))
        ax1.plot(range(N_TRAIN, N_TRAIN + steps), pred_uc.predicted_mean, 'b')
        # ax.fill_between(pred_ci.index,
        #                 pred_ci.iloc[:, 0],
        #                 pred_ci.iloc[:, 1], color='k', alpha=.25)
        # ax.set_xlabel('Date')
        # ax.set_ylabel('CO2 Levels')
        #
        # plt.legend()
        # plt.show()


dc = DataConfig()
DataFile = Data(dc.filename, dc.index_col, dc.tag, dc.time_step)
f1 = plt.figure()
ax1 = f1.add_subplot(1, 1, 1)
ax1.plot(DataFile.data.raw, 'g')

# TrainResults = ParaSearch(DataFile.data.raw, (0, 2))
# plt.plot(TrainResults.result_best.fittedvalues, 'r')

data_log = np.log(DataFile.data.raw)
pc = ParaCal(DataFile.data.raw, 10, 10, 40)

N_TRAIN = 380
PredictResult = Prediction(DataFile.data.raw[0:N_TRAIN], 20)
# data_log_diff = data_log - data_log.shift()
