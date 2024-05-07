import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("training/tip_boxes_640_n22/results.csv")
df = df.rename(columns=lambda x: x.strip())
print(df.columns)

cols = [c for c in df.columns if "train" in c or "val" in c]

names = {c.split('/')[1] for c in cols}

grps = [[f"val/{n}",f"train/{n}"] for n in names]

df.plot(x="epoch", y = cols, subplots=grps)

plt.show()