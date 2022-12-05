def cleanData_aug(Europe, embedding_arr,countryDat,mig_stock):
    europe = Europe[~(Europe["Origin Country"]=="LIE")]
    europe = europe[~(europe["Destination Country"]=="LIE")]
    europe = europe[["year","Origin Country","# Migrants","Destination Country"]]

    for embDist in embedding_arr:
        europe = pd.merge(europe,embDist,
                   left_on = ["Origin Country", "Destination Country"],
                   right_on = ["orig","dest"],
                   how = "left").drop(columns = ["orig", "dest"])

    countryDat = countryDat[["country", "year", "population", "gdp"]]

    europe3 = pd.merge(europe,countryDat.add_suffix(" (origin)"), left_on = ["Origin Country","year"], right_on = ["country (origin)", "year (origin)"],
                   how = "left").drop(columns = ["year (origin)","country (origin)"])

    europe4 = pd.merge(europe3,countryDat.add_suffix(" (dest)"), left_on = ["Destination Country", "year"], right_on = ["country (dest)", "year (dest)"],
                       how = "left").drop(columns = ["year (dest)", "country (dest)"])

    
    europe5 = pd.merge(europe4,mig_stock, left_on = ["Origin Country", "Destination Country", "year"], 
                       right_on = ["orig", "dest", "year"], how = "inner")
    
    europe_f = europe5.dropna()

    return europe_f
  
  
eurocountries = set(Europe["Origin Country"])-set("LIE")
europe_stock = migrant_stock[migrant_stock["orig"].apply(lambda x: x in eurocountries) 
                             &migrant_stock["dest"].apply(lambda x: x in eurocountries)]
for i in range(2001,2005):
    europe_stock[i] = europe_stock["2000"]+(i-2000)*(europe_stock["2005"] - europe_stock["2000"])/5
for i in range(2006,2010):
    europe_stock[i] = europe_stock["2005"]+(i-2005)*(europe_stock["2010"] - europe_stock["2005"])/5
    
europe_stock.rename(columns ={"2005":"2k05"},inplace = True)
europe_stock[2005] = (europe_stock["2010"] - europe_stock["2k05"])/5
    
euro_stock = europe_stock[[i for i in range(2001,2010)]]

euro_stock = europe_stock.melt(id_vars = ["dest", "orig"],value_vars = [i for i in range(2001,2010)])
euro_stock["year"] = euro_stock["variable"].apply(lambda x: int(x))
euro_stock.drop(columns = ["variable"], inplace = True)
euro_stock.rename(columns = {"value": "stock"})
