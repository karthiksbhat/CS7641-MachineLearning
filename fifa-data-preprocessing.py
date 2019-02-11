import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

orig_dataset = pd.read_csv("~/Documents/Datasets/ML/fifa_data.csv")

labels = {
	"LS": 1,
	"ST": 1,
	"RS": 1,
	"LW": 1,
	"LF": 1,
	"CF": 1,
	"RF": 1,
	"RW": 1, 
	"LAM": 2,
	"CAM": 2,
	"RAM": 2,
	"LM": 2,
	"LCM": 2,
	"CM": 2,
	"RCM": 2,
	"RM": 2,
	"LDM": 2,
	"CDM": 2,
	"RDM": 2,
	"LB": 3,
	"LCB": 3,
	"CB": 3,
	"RCB": 3,
	"RB": 3,
	"LWB": 3,
	"RWB": 3,
	"GK": 4
}

orig_dataset = orig_dataset.drop(['Photo', 'Flag', 'Club Logo', 'Work Rate', 'Body Type', 'Real Face', 'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until'], axis=1)
orig_dataset["position_label"] = orig_dataset["Position"].map(labels)
label_distribution = orig_dataset["position_label"].value_counts()
print_label = label_distribution.cumsum()
print_label.plot()
plt.show()
orig_dataset.replace('', np.nan, inplace=True)
orig_dataset.fillna(0, inplace=True)

train_dataset, test_dataset = train_test_split(orig_dataset, test_size=0.3, random_state=1)

train_dataset.to_csv("~/Documents/Datasets/ML/fifa-processed-train.csv", encoding='utf-8')
test_dataset.to_csv("~/Documents/Datasets/ML/fifa-processed-test.csv", encoding='utf-8')

# orig_dataset.to_csv("~/Documents/Datasets/ML/fifa-data-processed.csv", encoding='utf-8')