import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

# destFolderName = 'C:/Users/gavangol/OneDrive - Intel Corporation/Desktop/STIS/Data example clean light.xlsx'

# destFolderName = 'C:/Users/gavangol/OneDrive - Intel Corporation/Desktop/STIS/Data example clean.xlsx'
# data = pd.read_excel(destFolderName)
destFolderName = 'C:/Users/gavangol/OneDrive - Intel Corporation/Desktop/STIS/Data.csv'
data = pd.read_csv(destFolderName)


# #df= data.Machine
# #labels=df.drop_duplicates(subset=['Machine'])
# #labels=['Machine1','Machine2']
# #feature_list = list(features.columns)
# ########################################################Box plot###################################################
# Machine1= data[data['Machine']=='Machine1']
# Machine1=Machine1[['Process duration']].values.flatten()
#
# Machine2= data[data['Machine']=='Machine2']
# Machine2=Machine2[['Process duration']].values.flatten()
#
# Machine3= data[data['Machine']=='Machine3']
# Machine3=Machine3[['Process duration']].values.flatten()
#
# # plt.boxplot([Machine1,Machine2],patch_artist=True,notch=True,labels=['Machine1','Machine2'])
#
# bplot= plt.boxplot([Machine1,Machine2,Machine3],patch_artist=True,labels=['Machine1','Machine2','Machine3'])
#
# plt.xlabel('Machine')
# plt.ylabel('Process duration')
# plt.title('Machine vs Process duration')
#
#
# colors = ['pink', 'lightblue', 'lightgreen']
# for bplot in (bplot, bplot):
#     for patch, color in zip(bplot['boxes'], colors):
#        patch.set_facecolor(color)
#
# plt.show()
# #
# ########################################################   Scatter plot   ###################################################
#
# pt= data[data['Machine']=='Machine1']
# data=data[data['PROCESSED_WAFER_COUNT'] > 24]
# pt=data[['Process duration']].values.flatten()
#
# overlap=data[['OVERLAP_SECONDS']].values.flatten()
#
# partiality=data[['PARTIALITY_SCORE']].values.flatten()
#
# colors= partiality
#
# plt.scatter(overlap,pt, s=50, c=partiality, cmap='Greens', edgecolor='black')
# plt.xlabel('OVERLAP_SECONDS')
# plt.ylabel('Process duration')
# plt.title('Overlap vs Process duration (full lots)')
# cbar=plt.colorbar()
# cbar.set_label('partiality')
#
# plt.show()

#
# #######################################################   Coreleation metrics   ###################################################
#

data=data[['Process duration','PARTIALITY_SCORE','OVERLAP_SECONDS','PROCESSED_WAFER_COUNT','Wafer STDV']]
corrMatrix=data.corr()
sn.heatmap(corrMatrix, cmap="BuGn", annot=True)
plt.title('Correlation Matrix')
plt.show()


#
# scaterMatrix= pd.plotting.scatter_matrix(data, alpha=0.2)
# # plt.title('Scater Matrix')
#
# plt.show()
