pepbdb

# All data

- pdb gets screened protein data and peptide data by tiqv.ipynb (isolation)
- pepbdb The peptide_pepbdb and pocket_pepbdb are obtained through tiqv-Copy.ipynb

- Let's say we've intercepted the protein pocket and the peptide. From which we will extract the amino acids of the protein pocket, the 4 main atomic coordinates, the fasta file and the protbert model input file.Here fasta is used for the clustering of MMseq2. The specific clustering code is also given in ipynb.**Extract amino acids、 coordinates、fasta and protbert.ipynb**
- Next, we can obtain the input file of ESM-IF1 pre-training model and the semi-finished product file of GVP**GVP semi-finished product input and ESM-IF1 pre-training model input.ipynb**
- Grouping data and construction of negative examples **Grouping and constructing negative examples.ipynb**
- After the above we can ProtGVP final input data**ProtGVP_input.ipynb**
- Depending on the final input data of ProtGVP, we can easily get the input data of se3-transformer.**se3-transformer_input.ipynb**
- Depending on the final input data of ProtGVP and the output of the ESM-IF1 pre-training model, we can obtain the input of the ESM-IF1 model that we need.**ESM-IF1/esm-if1_input.ipynb**
