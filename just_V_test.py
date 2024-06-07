import numpy as np

# Beispiel-Array mit Werten (z.B. Messwerte über die Zeit)
werte = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Anzahl der letzten Werte, die integriert werden sollen
n = 5


letzte_werte = werte[-2:]
print


T_n =  n  

# Integration der letzten n Werte
integral = np.trapz(letzte_werte, dx=T_n)

# Mittelwert über die Zeit T_n
mittelwert = integral / T_n


new=(letzte_werte-mittelwert)**2
integral = np.trapz(new, dx=T_n)/ T_n


print(f"Der Mittelwert über die letzten {n} Werte ist: {mittelwert}")
print(f"mit varianz{integral}")
