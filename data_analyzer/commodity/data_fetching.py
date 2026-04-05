import pandas as pd
import yfinance as yf

# commodities

essential_commodities_electronics = {
    "Copper-Wiring and Circuitry" : "HG=F",
    "Silver-Batteries & Contacts" : "SI=F",
    "Gold-High-end Chips/Connectors": "GC=F",
    "Palladium-Semi-conductors/Storage":"PA=F",
    "Platinum-Precision Components" : "PL=F",
    "Natural Gas-Manufacturing Power":"NG=F",
}

essential_commodities_Food_and_Grocery = {
    "Wheat-Cereal & Flour" : "ZW=F",
    "Soybeans-Cooking Oil & Processed Foods":"ZS=F",
    "Sugar-Snacks & Baking":"SB=F",
    "Coffee-Beverages (Coffee)":"KC=F",
    "Cocoa-Chocolates/Confectionery":"CC=F",
}

essential_commodities_Clothing_Textiles = {
    "Cotton-Natural Fabrics (T-shirts/Denim)":"CT=F",
    "Crude Oil-Synthetic Fabrics/Shipping":"CL=F",
    "Lumber-Packaging (Online Delivery Boxes)":"LBS=F"
}

def get_data(ticker,name,period="12mo",interval = "1d"):
    data = yf.download(tickers=ticker,period=period,interval=interval)[["Close"]]
    data.index = pd.to_datetime(data.index)
    data.columns = [name]
    data = data.ffill()
    return data

