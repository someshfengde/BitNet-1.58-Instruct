#%%
import pickle
fp = open("old_outputs.pickle", "rb")
old_data = pickle.load(fp)

print(len(old_data['label_ids']))
print(len(old_data["label_ids"][0]))
print(old_data["label_ids"][0])

# %%
