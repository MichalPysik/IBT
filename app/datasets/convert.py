#  36499 93565

import pandas as pd
import numpy as np

df = pd.read_csv('MiniBooNE_PID_clean.txt', delim_whitespace=True, header=None)

classes = [1] * 36499 + [0] * 93565


df['class'] = classes

df = df.iloc[np.random.permutation(len(df))]

df.to_csv('MiniBooNE_particle.csv', index=False)