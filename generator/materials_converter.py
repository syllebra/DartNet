import yaml
import json
import pandas as pd
from io import StringIO
import os

if __name__ == "__main__":

    src = "3D/materials/Fe.yml"
    with open(src) as f:
        input = yaml.safe_load(f)

    comments = "# From https://refractiveindex.info/\n"
    for k,v in input.items():
        if (k != "DATA"):
            comments += f"# {k}:{v}\n"

    print(json.dumps(input,indent=4))
    data = StringIO(input["DATA"][0]["data"])

    tmp = os.path.splitext(src)[0]
    feta = open(f'{tmp}.eta.spd', "w")
    feta.write(comments)
    feta.write("#\n# wavelength(nm) eta\n")

    fk = open(f'{tmp}.k.spd', "w")
    fk.write(comments)
    fk.write("#\n# wavelength(nm) k\n")

    df = pd.read_csv(data, sep=" ")
    print(df)

    for i,r in df.iterrows():
        feta.write(f'{r.iloc[0]*1000} {r.iloc[1]}\n')
        fk.write(f'{r.iloc[0]*1000} {r.iloc[2]}\n')

    feta.close()
    fk.close()


    
