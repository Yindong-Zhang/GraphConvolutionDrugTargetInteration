from chembl_webresource_client.new_client import new_client
import pandas as pd
ids = []
with open('../../data/kiba-origin/compoundIDs', 'r') as f:
    for line in f:
        ids.append(line.strip())
molecule = new_client.molecule
nums = len(ids)
batchsize = 100
num_batch = nums // batchsize + 1
records = []
for i in range(num_batch):
    records.extend(molecule.get(ids[batchsize * i : min(batchsize * (i + 1), nums)]))

smiles = []
chemBL_ids = []
for r in records:
    structure =  r['molecule_structures']
    if(structure):
        smiles.append(r['molecule_structures']['canonical_smiles'])
        chemBL_ids.append(r['molecule_chembl_id'])
df = pd.DataFrame({'chEMBL_id': chemBL_ids, 'smiles': smiles})
df.to_csv('../../data/kiba-origin/kiba-smiles.csv')