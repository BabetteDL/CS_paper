import pandas as pd
import numpy as np
import statistics
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GridSearchCV

data = pd.read_excel(r'C:/Users/delee/Documents/Babette studie/Master/Blok 2/Computer Science/Data-paper.xlsx')

#data cleaning to make data comparabile
data['DVI Inputs'] = data['DVI Inputs'].replace(['No'], '0')
data['Width'] = data['Width'].str.replace('-1/18','.1')
data['Width'] = data['Width'].str.replace('-1/8','.1')
data['Width'] = data['Width'].str.replace('-9/64','.1')
data['Width'] = data['Width'].str.replace('-5/16','.3')
data['Width'] = data['Width'].str.replace('-19/64','.3')
data['Width'] = data['Width'].str.replace('-1/4','.3')
data['Width'] = data['Width'].str.replace('-3/8','.4')
data['Width'] = data['Width'].str.replace('-1/2','.5')
data['Width'] = data['Width'].str.replace('-5/8','.6')
data['Width'] = data['Width'].str.replace('-19/32','.6')
data['Width'] = data['Width'].str.replace('-39/64','.6')
data['Width'] = data['Width'].str.replace('-47/64','.7')
data['Width'] = data['Width'].str.replace('-49/64','.8')
data['Width'] = data['Width'].str.replace('-3/4','.8')
data['Width'] = data['Width'].str.replace('-7/8','.9')
data['Width'] = data['Width'].str.replace('-9/10','.9')
data['Width'] = data['Width'].str.replace('-55/64','.9')
data['Width'] = data['Width'].str.replace(' inches','"')
data['Width'] = data['Width'].str.replace('.0"','"')
data['TV Type'] = data['TV Type'].str.replace(' Flat-Panel', '')
data['Screen Size (Measured Diagonally)']=data['Screen Size (Measured Diagonally)'].str.replace('-1/32','')
data['Screen Size (Measured Diagonally)']=data['Screen Size (Measured Diagonally)'].str.replace('-1/25','')
data['Screen Size (Measured Diagonally)']=data['Screen Size (Measured Diagonally)'].str.replace('-1/8','.1')
data['Screen Size (Measured Diagonally)']=data['Screen Size (Measured Diagonally)'].str.replace('-1/5','.2')
data['Screen Size (Measured Diagonally)']=data['Screen Size (Measured Diagonally)'].str.replace('-1/2','.5')
data['Screen Size (Measured Diagonally)']=data['Screen Size (Measured Diagonally)'].str.replace('-27/50','.5')
data['Screen Size (Measured Diagonally)']=data['Screen Size (Measured Diagonally)'].str.replace('-3/4','.8')
data['Screen Size (Measured Diagonally)']=data['Screen Size (Measured Diagonally)'].str.replace('-7/8','.9')
data['Screen Size (Measured Diagonally)']=data['Screen Size (Measured Diagonally)'].str.replace('-9/10','.9')
data['Screen Size (Measured Diagonally)']=data['Screen Size (Measured Diagonally)'].str.replace('-5/8','.6')
data['Screen Size (Measured Diagonally)']=data['Screen Size (Measured Diagonally)'].replace(['64-1/2"'],'64.5"')
data['Mount Bracket/VESA Pattern']=data['Mount Bracket/VESA Pattern'].str.replace('mm','')
data['Mount Bracket/VESA Pattern']=data['Mount Bracket/VESA Pattern'].replace('[15.7" x 15.7"]','400 x 400')
data['Mount Bracket/VESA Pattern']=data['Mount Bracket/VESA Pattern'].replace('[23.6" x 15.7"]','600 x 400')
data['PC Inputs'] = data['PC Inputs'].replace(['No'], '0')
data['Weight']=data['Weight'].str.replace('lbs.', 'lb')
data['Weight']=data['Weight'].str.replace('lbs', 'lb')
data['Weight']=data['Weight'].str.replace('pounds', 'lb')
data['Brightness']=data['Brightness'].str.replace('m2', '')
data['Brightness']=data['Brightness'].str.replace('[cd/m²Â]', '')
data['Brightness']=data['Brightness'].str.replace('[Nit]', '')
data['Maximum Resolution']=data['Maximum Resolution'].replace(['1,024 x 768 (Native)'], '1024 x 768')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-1/8','.1')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-3/32','.1')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-5/32','.16')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-15/64','.23')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-5/16','.3')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-19/64','.3')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-1/4','.25')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-11/32','.34')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-3/8','.4')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-25/64','.4')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-7/16','.44')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-29/64','.45')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-1/2','.50')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-5/8','.6')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-9/16','.56')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-11/16','.7')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-45/64','.7')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-47/64','.73')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-3/4','.75')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-49/64','.75')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-7/8','.9')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-59/64','.90')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-9/10','.90')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-31/32','.95')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-61/64','.95')
data['Product Height (with stand)'] = data['Product Height (with stand)'].str.replace('-63/64','.98')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-1/32','')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-1/8','.1')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-11/64','.2')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-13/64','.2')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-9/32','.3')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-21/64','.3')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-1/4','.3')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-3/8','.4')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('3/8','.4')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-1/2','.5')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-5/8','.6')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-37/64','.6')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-2/3','.7')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-11/16','.7')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-45/64','.7')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-3/4','.8')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-13/16','.8')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-27/32','.8')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-7/8','.9')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-9/10','.9')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-29/32','.9')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('-15/16','.95')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].str.replace('.0"','"')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].replace(['HDTV: 1.3"; Blu-ray player: 7.5"'],'1.3"')
data['Product Depth (without stand)'] = data['Product Depth (without stand)'].replace(['1.6" (panel); 2.5" (speaker)'],'1.6"')
data['Warranty Terms - Labor'] = data['Warranty Terms - Labor'].replace(['1 year limited'], '1 year')
data['Warranty Terms - Labor'] = data['Warranty Terms - Labor'].replace(['1 year limited; 2 years limited: panel'], '1 year limited; 2 years: panel')
data['Warranty Terms - Labor'] = data['Warranty Terms - Labor'].replace(['2 years limited'], '2 years')
data['Warranty Terms - Labor'] = data['Warranty Terms - Labor'].replace(['90 days limited'], '90 days')
data['Warranty Terms - Parts'] = data['Warranty Terms - Parts'].replace(['1 year limited', '1Year'], '1 year')
data['Warranty Terms - Parts'] = data['Warranty Terms - Parts'].replace(['1 year limited; 2 years limited: panel', '1 year limited; 2 year: panel'], '1 year limited; 2 years: panel')
data['Warranty Terms - Parts'] = data['Warranty Terms - Parts'].replace(['2 Year','2 years limited', '2Year'], '2 years')
data['Warranty Terms - Parts'] = data['Warranty Terms - Parts'].replace(['3 Year', '3Year'], '3 years limited')
data['Warranty Terms - Parts'] = data['Warranty Terms - Parts'].replace(['90 Day', '90 days limited', '90Day'], '90 days')
data['Simulated Surround'] = data['Simulated Surround'].replace(['SRS TruSurround XT, SRS WOW'], 'SRS TruSurround XT, SRSWOW')
data['Simulated Surround'] = data['Simulated Surround'].replace(['Yes (Surround Effect: Cinema; Sports; Music; Game)'], 'Surround Effect: Cinema, Game, Music, Sports')
data['Simulated Surround'] = data['Simulated Surround'].replace(['SRS TSXT', 'SRS TruSound XT'], 'SRS TruSurround XT')
data['Simulated Surround'] = data['Simulated Surround'].replace(['SRS TrueSurround HD'], 'SRS TruSurround HD')
data['Simulated Surround'] = data['Simulated Surround'].replace(['Infinite Surround System', 'Infinite surround', 'Infinite Sound'], 'Infinite Surround')
data['Brand'] = data['Brand'].replace(['JVC TV'], 'JVC')
data['Brand'] = data['Brand'].replace(['LG Electronics'], 'LG')
data['Brand'] = data['Brand'].replace(['Pansonic'], 'Panasonic')
data['Brand'] = data['Brand'].replace(['SuperSonic'], 'Supersonic')
data['Brand'] = data['Brand'].replace(['TOSHIBA'], 'Toshiba')
data['Brand'] = data['Brand'].replace(['VIZIO'], 'Vizio')
data['title'] = data['title'].str.replace('SuperSonic', 'Supersonic')
data['title'] = data['title'].str.replace('TOSHIBA', 'Toshiba')
data['title'] = data['title'].str.replace('VIZIO', 'Vizio')
data['title'] = data['title'].str.replace(' inches', '"')
data['title'] = data['title'].str.replace('-Inch', '"')
data['title'] = data['title'].str.replace(' -', '')
data['title'] = data['title'].str.replace(' /', '')
#delete sites from titles since products on the same site can't be duplicates
data['title'] = data['title'].str.replace(' Best Buy', '')
data['title'] = data['title'].str.replace('Newegg.com ', '')
data['title'] = data['title'].str.replace(' Newegg.com', '')
data['title'] = data['title'].str.replace(' TheNerds.net', '')

#extract information from product titles to data and create binary vector representations
Brand = pd.get_dummies(data["Brand"])
for b in range(len(data)):
    for item in list(Brand):
        if item in data.loc[b, 'title']:
            Brand.loc[b,item]=1
            data.loc[b, 'Brand']=item

Screen_size = pd.get_dummies(data["Screen Size Class"])
flag = False
for b in range(len(data)):
    for item1 in list(Screen_size):
        if item1 in data.loc[b, 'title']:
            Screen_size.loc[b,item1]=1
            data.loc[b, "Screen Size Class"] = item1
            flag = True
            break #make sure I don't extract the diagonal screen size
    else:
            continue

Res_dum = pd.get_dummies(data["Vertical Resolution"])
for b in range(len(data)):
    for ite in list(Res_dum):
        if ite in data.loc[b, 'title']:
            Res_dum.loc[b,ite]=1
            data.loc[b, "Vertical Resolution"]=ite
        
SRR_dum = pd.get_dummies(data["Screen Refresh Rate"])
for b in range(len(data)):
    for it in list(SRR_dum):
        if it in data.loc[b, 'title']:
            SRR_dum.loc[b,it]=1
            data.loc[b, "Screen Refresh Rate"] = it

Type = pd.get_dummies(data["TV Type"])
for b in range(len(data)):
    for items in list(Type):
        if items in data.loc[b, 'title']:
            Type.loc[b,items]=1
            data.loc[b, "TV Type"] = items

datatoexcel = pd.ExcelWriter(r'C:/Users/delee/Documents/Babette studie/Master/Blok 2/Computer Science/Data_check.xlsx')
data.to_excel(datatoexcel)
datatoexcel.save()

bin_data = pd.concat([Brand, Screen_size, Res_dum, SRR_dum, Type], axis=1)

#minhashing
bin_data = bin_data.transpose()
n=100 #number of permutations
m = np.zeros((n, len(data)), dtype=int) #signature matrix
for p in range(n):
    perm = np.random.permutation(len(bin_data)).reshape(1,len(bin_data))
    for i in range(len(data)):
        for x in range(len(bin_data)):
            ind = int(np.where(perm == x)[1])
            if bin_data.iat[ind,i]==1:
                m[p,i]=x
                break
            
#locality sensitive hashing
bb=20
rr=5
#t=(1/bb)^(1/rr) #0.55
bands=np.split(m,bb)
hashm=np.zeros((bb,len(data)), dtype=np.int64) #hash matrix
for a in range(bb):
    band=bands[a]
    for c in range(len(data)):
        lists = band[:,c].tolist()
        hashm[a,c]= ''.join(str(item) for item in lists)

#determine candidate pairs based on hash matrix
candidates = pd.DataFrame(0, index=data.index, columns=data.index)
for q in range(len(data)):
    for r in range(len(data)):
        if r>q:  
            if data.iloc[q,2] != data.iloc[r,2]: #products on the same site cannot be duplicates
                if data.iloc[q,55] != data.iloc[r,55]: #products with different brands cannot be duplicates
                    continue
                else:
                    if data.iloc[q,24] != data.iloc[r,24]: #different vertical resoltion
                        continue
                    else:
                        if data.iloc[q,30] != data.iloc[r,30]: #different screen refresh rate
                            continue
                        else:        
                            for o in range(bb):
                                for s in range(bb):
                                    if hashm[o,r]==hashm[s,q]:
                                        candidates.iat[q,r]=1
                                        
#function to compute jaccard similarity
def jaccard_similarity(a, b):
    # convert to set
    a = set(a)
    b = set(b)
    j = float(len(a.intersection(b))) / len(a.union(b))
    return j

#extract model words from data
mw = data[['Brand', "Vertical Resolution","Screen Refresh Rate","TV Type",
           "Aspect Ratio", "Maximum Resolution", "Screen Size Class"]].copy()
#extract model words from title
titles=pd.DataFrame(index=data.index)
for d in range(24):
    titles[d] = data["title"].str.split(" ").str[d]

#compute jaccard similarity of model words for candidate pairs
dis_sim = pd.DataFrame(1, index=data.index, columns=data.index)
for u in range(len(data)):
    for v in range(len(data)):
        if v>u:
            if candidates.iat[u,v]==1:
                dis_sim.iat[u,v]=1-(jaccard_similarity(mw.iloc[u], mw.iloc[v])+
                jaccard_similarity(titles.iloc[u], titles.iloc[v]))/2

#compare all data
dis_sim_all = pd.DataFrame(1, index=data.index, columns=data.index)
for u in range(len(data)):
    for v in range(len(data)):
        if v>u:
            dis_sim_all.iat[u,v]=1-(jaccard_similarity(mw.iloc[u], mw.iloc[v])+
            jaccard_similarity(titles.iloc[u], titles.iloc[v]))/2

# splitting data in training and test set
# for bootstrapping, use random_state=7,42,59,130,200
train, test = train_test_split(data, test_size=0.37, random_state=7)
train["label"] = train["modelID"].astype("category")
train["label"] #check how many clusters

dis_sim_train, dis_sim_test = train_test_split(dis_sim, test_size=0.37, random_state=7)
dis_sim_train = dis_sim_train.drop(dis_sim_test.index, axis = 1)
dis_sim_train = pd.DataFrame(dis_sim_train, columns=dis_sim_train.index)
dis_sim_test = dis_sim_test.drop(dis_sim_train.index, axis = 1)
dis_sim_test = pd.DataFrame(dis_sim_test, columns=dis_sim_test.index)

# Create random grid
param = {'distance_threshold': (0.25, 0.3, 0.35, 0.40, 0.45)}
clustering_model = AgglomerativeClustering(n_clusters = None,
                                           affinity='precomputed',
                                           compute_full_tree=True,
                                           linkage="single")
grid_search = GridSearchCV(clustering_model, param, n_jobs=-1,
                           verbose=1, scoring='f1')
grid_search.fit(dis_sim)
best_param = grid_search.best_estimator_.get_params()

#clustering
clustering_model = AgglomerativeClustering(n_clusters = None,
                                           affinity='precomputed',
                                           compute_full_tree=True,
                                           linkage="single",
                                           distance_threshold=0.39)
clustering_model.fit(dis_sim_train)
clustering_model.n_clusters_

#performance evaluation to determine distance threshold
train["Cluster"] = clustering_model.labels_
found_dup_train = 0
fn_train = 0
tot_dup_train = 0
tot_comp_train = 0
candidates_train = 0
tn_train = 0
fp_train = 0
for g in range(len(train)):
    for h in range(len(train)):
        if h > g:
            if train.iloc[g,4] == train.iloc[h,4]:
                tot_dup_train += 1
                if train.iloc[g,64] == train.iloc[h,64]:
                    found_dup_train += 1
                else:
                    fn_train += 1
            if dis_sim_train.iat[g,h] != 1:
                tot_comp_train += 1
                if train.iloc[g,4] == train.iloc[h,4]:
                    candidates_train += 1
            if train.iloc[g,4] != train.iloc[h,4]:
                if train.iloc[g,64] != train.iloc[h,64]:
                    tn_train += 1
                else:
                    fp_train += 1
pair_quality_train = found_dup_train/tot_comp_train
pair_complete_train = found_dup_train/tot_dup_train
precision = found_dup_train/(found_dup_train+fp_train)
recall = found_dup_train/(found_dup_train+fn_train)
dat1_train = [precision, recall]
f1 = statistics.harmonic_mean(dat1_train)


#performance evaluation for test data
clustering_model.fit(dis_sim_test)
clustering_model.n_clusters_
test["Cluster"] = clustering_model.labels_
found_dup = 0
fn = 0
tot_dup = 0
tot_comp = 0
candidates = 0
tn = 0
fp = 0
for g in range(len(test)):
    for h in range(len(test)):
        if h > g:
            if test.iloc[g,4] == test.iloc[h,4]:
                tot_dup += 1
                if test.iloc[g,63] == test.iloc[h,63]:
                    found_dup += 1
                else:
                    fn += 1
            if dis_sim_test.iat[g,h] != 1:
                tot_comp += 1
                if test.iloc[g,4] == test.iloc[h,4]:
                    candidates += 1
            if test.iloc[g,4] != test.iloc[h,4]:
                if test.iloc[g,63] != test.iloc[h,63]:
                    tn += 1
                else:
                    fp += 1
pair_quality = found_dup/tot_comp
pair_complete = found_dup/tot_dup
precision_test = found_dup/(found_dup+fp)
recall_test = found_dup/(found_dup+fn)
dat1 = [precision_test, recall_test]
f1_test = statistics.harmonic_mean(dat1)

per_comp = tot_dup/((len(test)*len(test)-601)/2)
per_found_candidates = candidates/tot_comp

#performance evaluation using all data
clustering_model = AgglomerativeClustering(n_clusters = None,
                                           affinity='precomputed',
                                           compute_full_tree=True,
                                           linkage="single",
                                           distance_threshold=0.24)
clustering_model.fit(dis_sim_all)
clustering_model.n_clusters_

#performance evaluation to determine distance threshold
data["Cluster"] = clustering_model.labels_
found_dup_train = 0
fn_train = 0
tot_dup_train = 0
tot_comp_train = 0
candidates_train = 0
tn_train = 0
fp_train = 0
for g in range(len(data)):
    for h in range(len(data)):
        if h > g:
            if data.iloc[g,4] == data.iloc[h,4]:
                tot_dup_train += 1
                if data.iloc[g,63] == data.iloc[h,63]:
                    found_dup_train += 1
                else:
                    fn_train += 1
            if dis_sim_all.iat[g,h] != 1:
                tot_comp_train += 1
                if data.iloc[g,4] == data.iloc[h,4]:
                    candidates_train += 1
            if data.iloc[g,4] != data.iloc[h,4]:
                if data.iloc[g,63] != data.iloc[h,63]:
                    tn_train += 1
                else:
                    fp_train += 1
pair_quality_train = found_dup_train/tot_comp_train
pair_complete_train = found_dup_train/tot_dup_train
precision = found_dup_train/(found_dup_train+fp_train)
recall = found_dup_train/(found_dup_train+fn_train)
dat1_train = [precision, recall]
dat2_train = [pair_quality_train, pair_complete_train]
f1 = statistics.harmonic_mean(dat1_train)
f2 = statistics.harmonic_mean(dat2_train)