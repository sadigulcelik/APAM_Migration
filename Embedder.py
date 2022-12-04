import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def setupEmb(data_year,migrant_stock,country_data,eur_imm,onlyEuro=False):
    mig = migrant_stock[["dest", "orig", str(data_year)]].sort_values([str(data_year)],ascending = False).reset_index().drop(columns = ["index"])

    pop = country_data[country_data["year"]==data_year]
    pop_filt = pop[pop["population"] == pop["population"]]
    countries = set(mig["dest"][0:]).union(set(mig["orig"][0:])).intersection(set(pop_filt["country"]))
    if(onlyEuro):
        countries = countries.intersection(set(eur_imm["Origin Country"]))
    return mig,pop_filt, np.array(list(countries))

def embed(mig, pop,countries,data_year = 2000,tightness = 0.06,epsilon = 0.01,mindist = 0.04,expfac = 0.1):


    countries = np.array(list(countries))
    n = len(countries)
    countryMap = {}
    for i in range(0,len(countries)):
        countryMap[countries[i]] = i

    conn= np.zeros((len(countries),len(countries)))
    for i in range(0,len(mig)):
        entry = mig.iloc[i]
        c1 = entry["dest"];
        c2 = entry["orig"];

        if(c1 not in countries):
            continue
        if(c2 not in countries):
            continue

        i1 = countryMap[c1]
        i2 = countryMap[c2]
        # print(entry["orig"])

        # print(country_data[(country_data["country"]==entry["orig"]) &(country_data["year"]==2019)])
        div = np.min(pop[(pop["country"]==entry["orig"])]["population"].values[0])

        if(not div==div):
            print(entry["orig"])
        # div = np.sqrt(pop[pop["country"]==entry["dest"]]["2020"].values[0] * pop[pop["country"]==entry["orig"]]["2020"].values[0])
        conn[i1,i2] += entry[str(data_year)]/div
        conn[i2,i1] += entry[str(data_year)]/div
    # conn/=np.max(conn)

    conn*=100

    norm = np.linalg.norm(conn)/(len(countries)**2)
    conn2 = np.zeros((n,n))
    for i in range(0,len(conn)):
        for j in range(0,len(conn)):
            # cand = norm/ conn[i,j]
            # if(cand <1):
            conn2[i,j] = np.exp(-expfac*conn[i,j]/norm)
            # else:
            #     conn[i,j] = 1
    nearest = {}

    num_near = 15
    for i1 in range(0,len(conn2)):
        point = conn2[i1]
        indices = np.argsort(point)
        for i in range(0,num_near):
            i2 = indices[i]
            nearest[i1,i2]=point[i2]+mindist


    import cvxpy as cp
    N=len(conn2)
    D = np.ones((N,N))*(-1/(N))+np.identity(N)



    X = cp.Variable((N,N), symmetric=True)
    constraints = [X >> 0] # positive semi definite constraint
    constraints += [
        (((X)[k[0],k[0]]+(X)[k[1],k[1]]-(X)[k[0],k[1]]-(X)[k[1],k[0]]) - (nearest[k]**2))**2<=tightness for k in nearest # distance constraints
    ]

    prob = cp.Problem(cp.Maximize(cp.trace((D @ X))),
                          constraints)


    prob.solve(eps=epsilon)

    A,s,B = np.linalg.svd(X.value)
    rs = np.zeros((N,N))
    for i in range(N):
        rs[i,i] = np.sqrt(s[i])

    emb2 = A@ rs
    distances = []
    for i in range(0,len(countries)):
        for j in range(0,len(countries)):
            c1 = countries[i]
            c2 = countries[j]
            distances.append((c1,c2,np.linalg.norm(emb2[i] - emb2[j])))
    embDist = pd.DataFrame(distances)
    embDist = embDist.rename(columns = {0:"orig", 1: "dest", 2: "dist"})
    return embDist,A,s,B

def cleanData(Europe, embedding_arr,countryDat):
    europe = Europe[~(Europe["Origin Country"]=="LIE")]
    europe = europe[~(europe["Destination Country"]=="LIE")]
    europe = europe[["year","Origin Country","# Migrants","Destination Country"]]

    countryDat = countryDat[["country", "year", "population", "gdp"]]

    europe3 = pd.merge(europe,countryDat.add_suffix(" (origin)"), left_on = ["Origin Country","year"], right_on = ["country (origin)", "year (origin)"],
                   how = "left").drop(columns = ["year (origin)","country (origin)"])

    europe4 = pd.merge(europe3,countryDat.add_suffix(" (dest)"), left_on = ["Destination Country", "year"], right_on = ["country (dest)", "year (dest)"],
                       how = "left").drop(columns = ["year (dest)", "country (dest)"])

    for embDist in embedding_arr:
        if("year" in embDist):
            print("year")
            europe4 = pd.merge(europe4,embDist,
                       left_on = ["Origin Country", "Destination Country","year"],
                       right_on = ["orig","dest","year"],
                       how = "left").drop(columns = ["orig", "dest"])
        else:
            europe4 = pd.merge(europe4,embDist,
                       left_on = ["Origin Country", "Destination Country"],
                       right_on = ["orig","dest"],
                       how = "left").drop(columns = ["orig", "dest"])

    europe_f = europe4.dropna()

    return europe_f

def time_average(df):
    return pd.DataFrame(df.groupby(["Origin Country", "Destination Country"]).agg(np.average)).reset_index()

def Gravity_data(df,dist_measure):
    return np.vstack([np.log(df["population (origin)"].values),
                   np.log(df["gdp (origin)"].values/df["population (origin)"].values),
                   np.log(df["population (dest)"].values),
                   np.log(df["gdp (dest)"].values/df["population (dest)"].values),
                   np.log(df[dist_measure].values)
                   , np.ones(len(df))]).T

def Gravity_data_aug(df,dist_measure):
    return np.vstack([np.log(df["population (origin)"].values),
                   np.log(df["gdp (origin)"].values/df["population (origin)"].values),
                   np.log(df["population (dest)"].values),
                   np.log(df["gdp (dest)"].values/df["population (dest)"].values),
                   np.log(df["value"].values),
                   np.log(df[dist_measure].values),
                   np.ones(len(df))]).T



def draw_emb(A,s,countries,country_data, title=""):
    N = len(countries)
    rs = np.zeros((N,N))
    for i in range(N):
        rs[i,i] = np.sqrt(s[i])
    emb2 = A[:,0:2]@ rs[0:2,:]





    plt.figure(figsize = (8,8))
    plt.title(title, size = 20)


    for i in range(0,len(countries)):
        col =mcolors.hsv_to_rgb((i/len(countries),0.6,1))
        msize = country_data[(country_data["country"] == countries[i]) & (country_data["year"] == 2000)]["population"].values[0]
        plt.scatter(emb2[i,0], emb2[i,1],color = col,s= 20+np.sqrt(msize/1000))

        plt.annotate(countries[i], (emb2[i,0],emb2[i,1]))
