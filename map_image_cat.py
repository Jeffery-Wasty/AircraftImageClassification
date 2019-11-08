import pandas as pd

# Returns a defect type from a given file nae


def return_defect(file):
    data = pd.read_csv("defect_map.csv")
    defect_map = data.loc[:, {"Filename", "Defect Type"}]
    return (data.loc[data['Filename'] == file, 'Defect Type'].item())
