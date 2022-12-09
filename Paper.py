import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from sklearn.cluster import AgglomerativeClustering

data = pd.read_excel(r'C:/Users/delee/Documents/Babette studie/Master/Blok 2/Computer Science/Data-paper.xlsx')

#data cleaning
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

datatoexcel = pd.ExcelWriter(r'C:/Users/delee/Documents/Babette studie/Master/Blok 2/Computer Science/Data_check.xlsx')
data.to_excel(datatoexcel)
datatoexcel.save()

# get more information from product titles
Brand = pd.get_dummies(data["Brand"])
for b in range(1624):
    for item in list(Brand):
        if item in data.loc[b, 'title']:
            Brand.loc[b,item]=1

Res_dum = pd.get_dummies(data["Vertical Resolution"])
for b in range(1624):
    for ite in list(Res_dum):
        if ite in data.loc[b, 'title']:
            Res_dum.loc[b,ite]=1
        
SRR_dum = pd.get_dummies(data["Screen Refresh Rate"])
for b in range(1624):
    for it in list(SRR_dum):
        if it in data.loc[b, 'title']:
            SRR_dum.loc[b,it]=1

#binary vector representations
DVI_dum = pd.get_dummies(data["DVI Inputs"])
Energy = pd.get_dummies(data["ENERGY STAR Certified"])
Sleep_dum = pd.get_dummies(data["Sleep Timer"])
USB = pd.get_dummies(data["USB Port"])
Type = pd.get_dummies(data["TV Type"])
Chip_dum = pd.get_dummies(data["V-Chip"])
Video_dum = pd.get_dummies(data["Component Video Inputs"])
PC_dum = pd.get_dummies(data["PC Inputs"])
As_dum = pd.get_dummies(data["Aspect Ratio"])
SL_dum = pd.get_dummies(data["Sound Leveler"])
Med_dum = pd.get_dummies(data["Media Card Slot"])
Comp_dum = pd.get_dummies(data["Composite Inputs"])
Bright_dum = pd.get_dummies(data["Brightness"])
Aud_dum = pd.get_dummies(data["Audio Outputs"])
Hdmi_dum = pd.get_dummies(data["HDMI Inputs"])
MaxR_dum = pd.get_dummies(data["Maximum Resolution"])
WTL_dum = pd.get_dummies(data["Warranty Terms - Labor"])
WTP_dum = pd.get_dummies(data["Warranty Terms - Parts"])
SimS_dum = pd.get_dummies(data["Simulated Surround"])
USBIN_dum = pd.get_dummies(data["USB Input"])
Speak_dum = pd.get_dummies(data["Speakers"])
bin_data = pd.concat([Brand,DVI_dum,Energy,Sleep_dum,USB,Type,Chip_dum,Video_dum,
                     PC_dum,As_dum,SL_dum,Med_dum,Comp_dum,Bright_dum,Aud_dum,
                     Res_dum,Hdmi_dum,MaxR_dum,SRR_dum,WTL_dum, WTP_dum,SimS_dum,
                     USBIN_dum, Speak_dum], axis=1)

# splitting data in training and test set
# for bootstrapping, use random_state=7,32,59,130,200
train, test = train_test_split(bin_data, test_size=0.37, random_state=200)

#minhashing
bin_train = train.transpose()
n=100
m = np.zeros((n, 1023), dtype=int)
for p in range(n):
    perm = np.random.permutation(225).reshape(1,225)
    for i in range(1023):
        for x in range(225):
            ind = int(np.where(perm == x)[1])
            if bin_train.iat[ind,i]==1:
                m[p,i]=x
                break
            
#lsh with b=25 and r=4
t=0.447 #(1/b)^(1/r)
bands=np.split(m,25)
hashm=np.zeros((25,1023), dtype=np.int64)
for a in range(25):
    band=bands[a]
    for c in range(1023):
        lists = band[:,c].tolist()
        hashm[a,c]= ''.join(str(item) for item in lists)

#determine candidate pairs compute similarity
candidates = pd.DataFrame(0, index=range(1023), columns=range(1023))
candidates.columns =[train.index]
candidates.index = [train.index]
for q in range(1023):
    for r in range(1023):
        if r>q:  
            for o in range(25):
                for s in range(25):
                    if hashm[o,r]==hashm[s,q]:
                        candidates.iat[q,r]=1 

#compute jaccard similarity for candidate pairs
sim = pd.DataFrame(0, index=train.index, columns=train.index)
for q in range(1023):
    for r in range(1023):
        if r>q:
            for u in candidates.index:
                for v in candidates.index:
                    if candidates.iat[q,r]==1:
                        sim.iat[q,r]=jaccard_score(bin_train[u[0]], bin_train[v[0]])

#clustering
clustering_model = AgglomerativeClustering(linkage="single")
clustering_model.fit(sim)