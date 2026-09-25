import pandas as pd 


data = pd.read_csv("./data/gapminder-lex.csv")

nordics = ["Denmark", "Sweden", "Norway", "Finland","Iceland"]
nordics_selection = data["name"].isin(nordics)
pd_nordics_allyears = data[nordics_selection]

#print(pd_nordics_allyears)
years = [str(year) for year in range(1950,2006,5)]
cols = ["name"] + years
pd_nordics = pd_nordics_allyears[cols]
#print(pd_nordics)

pd_nordics.to_csv("./data/nordics-lex-1950-2005.csv", index=False)