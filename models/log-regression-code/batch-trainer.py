import numpy as np
import pandas as pd
import pickle
import os

# Loads the stock data of each company into a CSV
def load_stock_data(filePath):
    df = pd.read_csv(filePath)  # Reads stock data and stores into dataframe
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

class LogisticRegression:
    def __init__(self, learningRate=0.01, numOfIterations=999):
        self.learningRate = learningRate
        self.numOfIterations = numOfIterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))

    def fit(self, X, y):
        numOfSamples, numOfFeatures = X.shape
        self.weights = np.zeros(numOfFeatures)
        self.bias = 0

        # Gradient descent
        for _ in range(self.numOfIterations):
            linearModel = np.dot(X, self.weights) + self.bias
            yPred = self._sigmoid(linearModel)

            dw = (1 / numOfSamples) * np.dot(X.T, (yPred - y))
            db = (1 / numOfSamples) * np.sum(yPred - y)

            # update step (this was missing)
            self.weights -= self.learningRate * dw
            self.bias -= self.learningRate * db

    def predict(self, X):
        linearModel = np.dot(X, self.weights) + self.bias
        yPred = self._sigmoid(linearModel)
        return np.array([1 if i > 0.5 else 0 for i in yPred])

if __name__ == "__main__":
    tickers = ["A","AAPL","ABBV","ABNB","ABT","ACGL","ACN","ADBE","ADI","ADM",
               "ADP","ADSK","AEE","AEP","AES","AFL","AIG","AIZ","AJG","AKAM",
               "ALB","ALGN","ALL","ALLE","AMAT","AMCR","AMD","AME","AMGN","AMP",
               "AMT","AMZN","ANET","ANSS","AON","AOS","APA","APD","APH","APO","APTV",
               "ARE","ATO","AVB","AVGO","AVY","AWK","AXON","AXP","AZO","BA","BAC","BALL",
               "BAX","BBY","BDX","BEN","BG","BIIB","BK","BKNG","BKR","BLDR","BLK","BMY",
               "BR","BRO","BSX","BX","BXP","C","CAG","CAH","CARR","CAT","CB","CBOE","CBRE",
               "CCI","CCL","CDNS","CDW","CEG","CF","CFG","CHD","CHRW","CHTR","CI","CINF",
               "CL","CLX","CMCSA","CME","CMG","CMI","CMS","CNC","CNP","COF","COO","COP",
               "COR","COST","CPAY","CPB","CPRT","CPT","CRL","CRM","CRWD","CSCO","CSGP","CSX",
               "CTAS","CTRA","CTSH","CTVA","CVS","CVX","CZR","D","DAL","DASH","DAY","DD","DE",
               "DECK","DELL","DFS","DG","DGX","DHI","DHR","DIS","DLR","DLTR","DOC","DOV","DOW",
               "DPZ","DRI","DTE","DUK","DVA","DVN","DXCM","EA","EBAY","ECL","ED","EFX","EG","EIX",
               "EL","ELV","EMN","ENPH","EOG","EPAM","EQIX","EQR","EQT","ERIE","ES","ESS","ETN",
               "ETR","EVRG","EW","EXC","EXE","EXPD","EXPE","EXR","F","FANG","FAST","FCX","FDS",
               "FDX","FE","FFIV","FI","FICO","FIS","FITB","FOX","FOXA","FRT","FSLR","FTNT","FTV",
               "GD","GDDY","GE","GEN","GEV","GILD","GIS","GL","GLW","GM","GNRC","GOOG","GOOGL","GPC",
               "GPN","GWW","HAL","HAS","HBAN","HCA","HD","HES","HIG","HII","HLT","HOLX","HON","HPE",
               "HPQ","HRL","HSIC","HST","HSY","HUBB","HUM","HWM","IBM","ICE","IDXX","IEX","IFF","INCY",
               "INTC","INTU","INVH","IP","IPG","IQV","IR","IRM","ISRG","IT","ITW","IVZ","J","JBHT","JBL",
               "JCI","JKHY","JNJ","JNPR","JPM","K","KDP","KEY","KEYS","KHC","KIM","KKR","KLAC","KMB","KMI",
               "KMX","KO","KR","KVUE","L","LDOS","LEN","LH","LHX","LII","LIN","LKQ","LLY","LMT","LNT","LOW",
               "LOW","LRCX","LULU","LUV","LVS","LW","LYB","LYV","MA","MAA","MAR","MAS","MCD","MCHP","MCK",
               "MCO","MDLZ","MDT","MET","META","MGM","MHK","MKC","MKTX","MLM","MMC","MMM","MNST","MO","MOH",
               "MOS","MPC","MPWR","MRK","MRNA","MS","MSCI","MSFT","MSI","MTB","MTCH","MTD","MU","NCLH","NDAQ",
               "NDSN","NEE","NEM","NFLX","NI","NKE","NOC","NOW","NRG","NSC","NTAP","NTRS","NUE","NVDA","NVR",
               "NWS","NWSA","NXPI","O","ODFL","OKE","OMC","ON","ORCL","ORLY","OTIS","OXY","PANW","PARA","PAYC",
               "PAYX","PCAR","PCG","PEG","PEP","PFE","PG","PGR","PH","PHM","PKG","PLD","PLTR","PM","PNC","PNR",
               "PNW","PODD","POOL","PPG","PPL","PRU","PSA","PSX","PTC","PWR","PYPL","QCOM","RCL","REG","REGN",
               "RF","RJF","RL","RMD","ROK","ROL","ROP","ROST","RSG","RTX","RVTY","SBAC","SBUX","SCHW","SHW","SJM",
               "SJM","SLB","SMCI","SNA","SNPS","SO","SOLV","SPG","SPGI","SRE","STE","STLD","STT","STX","STZ","SW",
               "SWK","SWKS","SYF","SYK","SYY","T","TAP","TDG","TDY","TECH","TEL","TER","TFC","TGT","TJX","TKO","TMO",
               "TMUS","TPL","TPR","TRGP","TRMB","TROW","TRV","TSCO","TSLA","TSN","TT","TTWO","TYL","UAL","UBER","UDR",
               "UHS","ULTA","UNH","UNP","UPS","URI","USB","V","VICI","VLO","VLTO","VMC","VRSK","VRSN","VST","VTR","VTRS",
               "VZ","WAB","WAT","WBA","WBD","WDAY","WDC","WDC","WEC","WELL","WFC","WM","WMB","WMT","WRB","WSM","WST","WTW",
               "WY","WYNN","XEL","XOM","XYL","YUM","ZBH","ZBRA","ZTS"]
    
    data_folder = "add-own-directory-to-.csv-files"
    model_folder = "add-own-directory-to-save-model-files"

    os.makedirs(model_folder, exist_ok=True)

    for ticker in tickers:
        print(f"Training model for {ticker}...")
        file_path = os.path.join(data_folder, f"{ticker}.csv")
        df = load_stock_data(file_path)

        features = ["Close", "High", "Low", "Open", "Volume", "Daily Return(%)", "Monthly Volatility", "Monthly Momentum"]
        X = df[features].values

        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        y = df["Target"].dropna().values
        X = X[:len(y)]

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = LogisticRegression(learningRate=0.001, numOfIterations=9999)
        model.fit(X_train, y_train)

        model_file = os.path.join(model_folder, f"logistic_{ticker}_model.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print(f"{ticker} Accuracy: {accuracy:.2f}")
