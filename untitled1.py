# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:19:59 2024

@author: Paul
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
<<<<<<< HEAD


# # Load the Excel file
df = pd.read_excel('results.xlsx')
df_full = pd.read_excel('results_full.xlsx')
#df = pd.read_pickle('results.pkl')
# Plot the data
df=df.astype(complex)
plt.figure(figsize=(10, 6))
df=df.drop(df['<0|0>'].idxmax())
df=df.drop(df['<2|2>'].idxmax())

plt.plot(df['V'], df['<0|0>'], label='<0|0>', marker='o')
plt.plot(df['V'], df['<1|1>'], label='<1|1>', marker='x')
plt.plot(df['V'], df['<2|2>'], label='<2|2>', marker='s')
plt.axvline(x=-0.5, color='red', linestyle='--', label='x = -0.5')

plt.xlabel('V')
plt.ylabel('Expectation Values')
plt.title('Expectation Values as a function of V')
plt.legend()
#plt.yticks(np.linspace(df[['<0|0>', '<1|1>', '<2|2>']].min().min(), df[['<0|0>', '<1|1>','<2|2>']].max().max(), num=10))

plt.grid(True)

plt.show()

=======
#first value is V, then 00 then 22
data_set=[[-1/2,0,1],
    [-1/4,-1,2],
    [-1/8,-3,4],
    [-1/16,-7,8],
    [-1/32,-15,16],
    #[-1/2500,-1249,1250],
    #[1/250,126,-125],
    [1/25,26,-25],
    [1/2,2,-1],
    [2,1.25,-0.25],
    [20,1.025,-0.025],
    [7.825,1.0638977635782747,-0.06389776357827476]
    ]


# Extracting values for plotting
y_values = [item[0] for item in data_set]
x1_values = [item[1] for item in data_set]
x2_values = [item[2] for item in data_set]

red_point_x = 0
red_point_y = -1/2
# Plot for x1 values
plt.figure(figsize=(10, 6))
plt.scatter(y_values, x1_values, label='<0|0>(V)', marker='o',linestyle='-')
plt.scatter(red_point_x, red_point_y, color='red', label='(-1/2, 0)', marker='o')

# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('<0|0>(V)')
plt.ylabel('V')
plt.title('Log-Log Plot of y versus x1')

plt.legend()

plt.grid(True, which="both", ls="--")
plt.show()

# # Plot for x2 values
# plt.figure(figsize=(10, 6))
# plt.scatter(x2_values, y_values, label='x2(y)', marker='x')
# #plt.xscale('log')
# #plt.yscale('log')
# plt.xlabel('x2')
# plt.ylabel('y')
# plt.title('Log-Log Plot of y versus x2')
# plt.legend()
# plt.grid(True, which="both", ls="--")
# plt.show()
>>>>>>> 7f695764ea33e04819c3e04db9dc76af61a1e227
