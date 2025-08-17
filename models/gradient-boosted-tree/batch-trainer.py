import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# --- FUNCTIONS ---
def load_stock_data(filePath):
    df = pd.read_csv(filePath)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

# --- CONFIG ---
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

dataFolder = "/content/drive/MyDrive/Stock.Predictor-AI/data/csv_with_features/"
modelFolder = "/content/drive/MyDrive/StockPredictor/models_gbt_regressor/"
os.makedirs(modelFolder, exist_ok=True)

features = ["Close", "High", "Low", "Open", "Volume", "Daily Return(%)", 
            "Monthly Volatility", "Monthly Momentum"]

gbtParameters = dict(
    n_estimators=400,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    max_features=None,
    random_state=42
)

# --- TRAIN LOOP ---
for ticker in tickers:
    filePath = os.path.join(dataFolder, f"{ticker}.csv")
    if not os.path.exists(filePath):
        print(f"[Skipping..] Missing CSV for {ticker}")
        continue

    df = load_stock_data(filePath)

    # Target: next day's close
    df["Target"] = df["Close"].shift(-1)
    df = df.dropna()

    X = df[features].values
    y = df["Target"].values

    if len(y) < 200:
        print(f"Not enough rows, skipping {ticker}")
        continue

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train regressor
    model = GradientBoostingRegressor(**gbtParameters)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{ticker} → Test RMSE: {rmse:.4f}")

    # --- Predict next day ---
    latest_features = df[features].iloc[-1].values.reshape(1, -1)
    next_close = model.predict(latest_features)[0]
    today_close = df["Close"].iloc[-1]
    trend = "Up" if next_close > today_close else "Down"
    confidence = abs(next_close - today_close) / today_close

    print(f"{ticker} → Predicted Next Close: {next_close:.2f}, Trend: {trend}, Confidence: {confidence:.2%}")

    # Save model
    modelPath = os.path.join(modelFolder, f"gbt_regressor_{ticker}.pkl")
    with open(modelPath, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved: {modelPath}\n")




