# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:19:59 2024

@author: Paul
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Load the Excel file
df = pd.read_pickle('results_3.pkl')
#df_full = pd.read_excel('results_full.xlsx')
#df = pd.read_pickle('results.pkl')
# Plot the data
df=df.astype(complex)
plt.figure(figsize=(10, 6))
df=df.drop(df['<0|0>'].idxmax())
df=df.drop(df['<2|2>'].idxmax())

#plt.plot(df['V'], df['<0|0>'], label='<0|0>', marker='o', linestyle='None')
#plt.plot(df['V'], df['<1|1>'], label='<1|1>', marker='x', linestyle='None')
plt.plot(df['V'], df['<2|2>'], label='<2|2>', marker='s', linestyle='None')
#plt.axvline(x=-0.5, color='red', linestyle='--', label='x = -0.5')

plt.xlabel('V')
plt.ylabel('Expectation Values')
plt.title('Expectation Values as a function of V')
plt.legend()
#plt.yticks(np.linspace(df[['<0|0>', '<1|1>', '<2|2>']].min().min(), df[['<0|0>', '<1|1>','<2|2>']].max().max(), num=10))

plt.grid(True)

plt.show()

