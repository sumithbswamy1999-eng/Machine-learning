#Fetching the data 
#scikit-Learn provides many helper function to download popular dataset .MNIST is one of them 
#The Script runs on python 3.5
import sys #loads the built in sys -system specific parameter and function module which contain information about python interpreter
assert sys.version_info >=(3,5)#sys.version_info :Returns a tuple like object containing the major,minor and micro version numbers of python and assert ...>(3,5)Compares the current python version tuple against(3,5)
#Scikit_learn  >=0.20 is required
import sklearn 
assert sklearn.__version__>="0.20"
#function merges the OECD'S life satisfaction data and IMF'S GDP per capita data
file="/Users/sumithbswamy/Downloads/oecd_bli_2015.csv"
file1="/Users/sumithbswamy/Downloads/gdp_per_capita.csv"
import pandas as pd 

def prepare_country_stats(oecd_bli,gdp_per_capita):
    oecd_bli=oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]#Keeps the only rows where the "INEQUALITY" column has the value "TOT"(which stands for total)
    oecd_bli=oecd_bli.pivot(index="Country",columns="Indicator",values="Value")#each country has exactly one row and each distinct indicator becomes its own columns
    gdp_per_capita.rename(columns={"2015":"GDP per capita"},inplace=True)
    gdp_per_capita.set_index("Country",inplace=True)
    full_country_stats=pd.merge(left=oecd_bli,right=gdp_per_capita,left_index=True,right_index=True)
    full_country_stats.sort_values(by="GDP per capita",inplace=True)  
    remove_indices=[0,1,6,8,33,34,35]
    #keep_indices=list(set(range(36)-set(remove_indices)))    
    return full_country_stats[["GDP per capita","Life satisfaction"]]#.iloc[[keep_indices]]
import os 
#The code in the book expect the data files to be located in the current directory
datapath=os.path.join("datasets", "lifesat", "")
#To Plot pretty figure directly
import matplotlib as  mpl 
#mpl.rc("axes",label=14)
#mpl.rc("xtick",label=12)
#mpl.rc("ytick",label=12)
#Download the data 
import urllib.request
Download_Root="https://raw.githubusercontent.com/ageron/handson-ml2/master/"
os.makedirs(datapath,exist_ok=True)
for filename in ("oecd_bli_2015.csv","gdp_per_capita.csv"):
    print("Downloading",filename)
    url=Download_Root+"datasets/lifesat/"+filename
    urllib.request.urlretrieve(url,datapath+filename)
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import sklearn.linear_model 
#Load the data
oecd_bli=pd.read_csv(datapath+ "oecd_bli_2015.csv",thousands=",")
gdp_per_capita=pd.read_csv(datapath+"gdp_per_capita.csv",thousands=",",delimiter="\t",encoding="latin1",na_values="n/a")
#Prepare the data
country_stats=prepare_country_stats(oecd_bli,gdp_per_capita)
X=np.c_[country_stats["GDP per capita"]]
Y=np.c_[country_stats["Life satisfaction"]]
#Visualize the data 
country_stats.plot(kind="scatter",x="GDP per capita",y="Life satisfaction")
#Select a linear model 
model=sklearn.linear_model.LinearRegression()
#Train the model 
model.fit(X,Y)  
#Make a predication for Cyprus
X_new=[[22587]]#Cyprus GDP per capita 
print(model.predict(X_new))

plt.show()

    
