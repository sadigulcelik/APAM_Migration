# metropolis hastings

targets = []
migrations = []
predictions = []
diffarr = []

deviations = []

numiter = 0;
counter = 0;
for c1 in countries:
    print(c1,end = " ")
    for c2 in countries:
        if(c1==c2):
            continue
            
        if(c1 == "LIE" or c1 == "MLT"):
            continue
        if(c2 == "LIE" or c2 == "MLT"):
            continue
            
        numiter+=1;
        target = comb[(comb["orig"]==c1) & (comb["dest"]==c2)]["value"].values[0]
        base = comb[(comb["orig"]==c1) & (comb["dest"]==c2)]["vo"].values[0]
        truth = [(euro_test["orig"] ==c1) & (euro_test["dest"]==c2)][0].values
        indices = np.where(truth)
        
        
        pred = pred_sel[indices]
        cur = pred.copy()
        p1 = 1
        for j in range(0,len(pred)):
            p1*=gamma.pdf(pred[j]-cur[j],a, loc = loc, scale = scale)
        cur_dif = np.log(base+np.sum(np.exp(cur))) - np.log(target+1)
        p2 = gamma.pdf(cur_dif, aS,loc = locS,scale = scaleS)
        cur_prob = p1*p2
        arr = []
        
        
        while(len(arr)<50):
            norm = np.random.normal(size = 2)
            new = cur + stdofhop*norm
            p1 = 1
            for j in range(0,len(pred)):
                p1*=gamma.pdf(pred[j]-new[j],a, loc = loc, scale = scale)
            new_dif = np.log(base+np.sum(np.exp(new))) - np.log(target+1)
            p2 = gamma.pdf(new_dif, aS,loc = locS,scale = scaleS)
            
            new_prob = p1*p2
            
            alpha = new_prob / cur_prob
            

            if(np.random.random()<= alpha):
                arr.append(new)
                diffarr.append(new_dif)
                cur = new
                cur_prob = new_prob
        predictions.append(np.mean(arr,axis = 0))
        deviations.append(np.std(arr,axis = 0))
        targets.append(target)
        migrations.append(euro_test[(euro_test["orig"] ==c1) & (euro_test["dest"]==c2)]["# Migrants"].values)


preds = np.array(predictions)
logpreds = np.log(np.sum(np.exp(preds),axis = 1))
logtargets = np.log(np.array(targets)+1)
migrations = np.array(migrations)
migflat = (migrations.flatten())
predsflat = preds.flatten()

