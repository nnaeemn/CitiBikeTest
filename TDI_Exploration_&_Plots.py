#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:46:03 2019

@author: naeemnowrouzi
"""
import os
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn import preprocessing
from matplotlib import pyplot as pl
sns.set(color_codes=True)
#%matplotlib inline

os.getcwd()
os.chdir('/Users/naeemnowrouzi/Desktop/TDIChallenge')
#os.listdir()

# Prepare for Merge
data_0 = pd.DataFrame(pd.read_csv('./all_inventors.csv', dtype={'application_number': np.int64}))
data_1 = pd.DataFrame(pd.read_csv('./application_data.csv', dtype={'application_number' : np.str}))
#data_2 = pd.DataFrame(pd.read_csv('./correspondence_address.csv'))

#data_0.application_number.nunique()

data_11 = data_1[~data_1.application_number.str.startswith('PCT')]
data_11.application_number = pd.to_numeric(data_11.application_number)
#data_11.dtypes
#data_0.dtypes

# Merge
merged_data = pd.merge(data_0, data_11, on='application_number')
merged_data = pd.DataFrame(merged_data)
# Convert date-time from object to datettime type.
merged_data.filing_date = merged_data.filing_date.astype(str)
merged_data.filing_date = pd.to_datetime(merged_data.filing_date)
merged_data.shape



# plot cummulative number of inventors in 5 states

 #filing_dates = pd.DatetimeIndex(merged_data.filing_date)
 #merged_data.groupby('inventor_region_code').size()

state_totals_cummul = {'New York' : data_0.inventor_region_code[data_0.inventor_region_code == 'NY'].shape[0],
          'New Jersey' : data_0.inventor_region_code[data_0.inventor_region_code == 'NJ'].shape[0],
          'California' : data_0.inventor_region_code[data_0.inventor_region_code == 'CA'].shape[0], 
          'Massachusetts' : data_0.inventor_region_code[data_0.inventor_region_code == 'MA'].shape[0],
          'Texas' : data_0.inventor_region_code[data_0.inventor_region_code == 'TX'].shape[0],
          'Florida' : data_0.inventor_region_code[data_0.inventor_region_code == 'FL'].shape[0]
          }
X = np.arange(len(state_totals_cummul))
pl.figure(figsize=(20,10))
pl.bar(X, state_totals_cummul.values(), align='center', width=0.5, alpha=0.5)
pl.xticks(X, state_totals_cummul.keys())
pl.xlabel('State')
pl.ylabel('Total Number of Patenr Applications (1931-2018)')
ymax = max(state_totals_cummul.values()) + 500000
pl.ylim(0, ymax)
pl.title('Total Number of Patent Applicants (Inventors) from 1931 - 2018')
pl.show()

# Plot NY and CA number of applications through time. 
merged_data_CA = merged_data[merged_data.inventor_region_code == 'CA']
merged_data_NY = merged_data[merged_data.inventor_region_code == 'NY']
merged_data_NY.filing_date = merged_data_NY.filing_date.astype(np.datetime64)

merged_data_NY.patent_number =  merged_data_NY.patent_number.astype(str)
issued_patents_NY = merged_data_NY[merged_data_NY.patent_number != 'None']
#issued_patents_NY.shape 

d = merged_data_NY.groupby('filing_date').size()

d
pl.figure(figsize=(20,10))
pl.title('Daily Number of Filed Patent Applications vs Number of Issued Patents for Inventors from NY')
d.plot(label = 'Patent Applications Filed')
issued_patents_NY_0 = issued_patents_NY.groupby(['filing_date']).size()
issued_patents_NY_0.plot(label = 'Issued Patents')
pl.legend(loc='upper left')
plt.show()


merged_data_CA.patent_number =  merged_data_CA.patent_number.astype(str)
issued_patents_CA = merged_data_CA[merged_data_CA.patent_number != 'None']

C = merged_data_CA.groupby('filing_date').size()
pl.figure(figsize=(20,10))
pl.title('Daily Number of Filed Patent Applications vs Number of Issued Patents for Inventors from CA')
C.plot(label = 'Patent Applications Filed')
issued_patents_CA_0 = issued_patents_CA.groupby(['filing_date']).size()
issued_patents_CA_0.plot(label = 'Issued Patents')
pl.legend(loc='upper left')
plt.show()


# Plotting percentages of application involving n Co-inventors.
number_of_collaborators = pd.DataFrame({'num_inventors' : merged_data.groupby('application_number').size()}).reset_index()

number_of_collaborators.application_number.nunique()
df= pd.DataFrame({'num_inventors_count' : number_of_collaborators.groupby('num_inventors').size()}).reset_index()
df['percentage'] = (df.num_inventors_count/number_of_collaborators.application_number.nunique())*100
df = df[df.num_inventors < 50]
y_labels= np.arange(50)+1
pl.figure(figsize=(20,10))
pl.xticks(y_labels)
pl.xlim(right = 11)
pl.title('Percentage of Patent Applications Having One or More Co-inventors')
pl.xlabel('Number of Inventors per Patent')
pl.ylabel('Percentage of All Patent Applications')
pl.bar(df.num_inventors,df.percentage, align='center', color='green', alpha=0.6, width=0.5)



# Invention classes

# Load data on USPC Class Descriptions
uspc_class_descriptions = pd.read_csv('./uspc_class_descriptions.csv')
uspc_class_descriptions.uspc_class = uspc_class_descriptions.uspc_class.astype(str) 

data_11.uspc_class = data_11.uspc_class.astype(str)

#  The general class descriptions for classes 514 and 424 are of the same category, as confirmed below from the uspc classes 
#  data downloaded separately, so we replace all instances of one with the other. 
uspc_class_descriptions[uspc_class_descriptions.uspc_class == '514']
uspc_class_descriptions[uspc_class_descriptions.uspc_class == '424']

data_11[data_11.uspc_class == '514'].shape
data_11[data_11.uspc_class == '424'].shape
uspc_class_descriptions.columns

# Plotting invention classes for all of the aplocations
invention_title_data = data_11[data_11.uspc_class != 'nan']
invention_title_data['uspc_class'] = invention_title_data['uspc_class'].replace('424','514')


#a=merged_data[merged_data.inventor_rank == 3]
#a.uspc_class

uspc_num_classes = pd.DataFrame({'num_inventions' : invention_title_data.groupby('uspc_class').size()}).reset_index()
uspc_num_classes = pd.merge(uspc_num_classes, uspc_class_descriptions, on='uspc_class', how='left')


from bokeh.plotting import figure, output_file, show

output_file('TDI.html')

p = figure(x_range=uspc_num_classes.uspc_class,plot_width=400, plot_height=400,
           title=None, toolbar_location="above")

p.vbar(x=uspc_num_classes.uspc_class, top=uspc_num_classes.num_inventions, width=0.9)

p.xgrid.grid_line_color = None
p.y_range.start = 0

show(p)











#pl.figure(figsize=(20,10))
#plot = pl.bar(uspc_num_classes.uspc_class, uspc_num_classes.num_inventions, align='center', color='red', alpha=0.5, width=1)
#pl.xticks(rotation=90)
#pl.xlabel('Invention Class')
#pl.ylabel('Number of Inventions')
#pl.title('Number of Inventions for the 100 Most Common Invention Classes')
#mplcursors.cursor(hover=True)
#pl.show()






#########################





# Plotting invention classes for applicants from NY


# Plotting invention classes for applicatns from CA 


#np.random.seed(1)
#x = np.random.rand(15)
#y = np.random.rand(15)
#names = np.array(list("ABCDEFGHIJKLMNO"))
#c = np.random.randint(1,5,size=15)
#norm = plt.Normalize(1,4)
#cmap = plt.cm.RdYlGn

#fig,ax = pl.subplots()
#sc = plt.bar(uspc_num_classes.uspc_class, uspc_num_classes.num_inventions, align='center')

#annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
#                    bbox=dict(boxstyle="round", fc="w"),
 #                   arrowprops=dict(arrowstyle="->"))
#annot.set_visible(False)

#def update_annot(ind):

 #   pos = sc.get_offsets()[ind["ind"][0]]
  #  annot.xy = pos
   # text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
    #                       " ".join([names[n] for n in ind["ind"]]))
    #annot.set_text(text)
    #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    #annot.get_bbox_patch().set_alpha(0.4)


#def hover(event):
 #   vis = annot.get_visible()
  #  if event.inaxes == ax:
   #     cont, ind = sc.contains(event)
    #    if cont:
     #       update_annot(ind)
      #      annot.set_visible(True)
       #     fig.canvas.draw_idle()
     #   else:
      #      if vis:
       #         annot.set_visible(False)
        #        fig.canvas.draw_idle()

#fig.canvas.mpl_connect("motion_notify_event", hover)

#plt.show()

data_11.dtypes
data_11.filing_location.isnull().sum()





















##################################################################################################################################################
ds = pd.DataFrame({'A':[1,2,3], 'B' :[4,5,6], 'C':[7,8,9]})
df = pd.DataFrame(np.random.randint(0,10,(5,5)))

ds[0:2,:]

ds.loc[1,1]
ds.iloc[1,1]
