# conda environment: venv (has geneformer installed)
import datetime
import pickle
import json
from geneformer import Classifier

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)}{current_date.month:02d}{current_date.day:02d}_{current_date.hour:02d}{current_date.minute:02d}"

output_prefix = "gene_classification"
output_dir = f"/home/golem/scratch/chans/lincs/output/gene_classification/{datestamp}"

gene_token_dict = json.load(open("/home/golem/scratch/chans/lincs/data/gene_token_dict.json", "r"))
# print(gene_token_dict)

cc = Classifier(classifier="gene",
                gene_class_dict = gene_token_dict,
                max_ncells = 10_000,
                freeze_layers = 4,
                num_crossval_splits = 5,
                forward_batch_size=200,
                nproc=16)

# cc.prepare_data(input_data_file="/path/to/gc-30M_sample50k.dataset",
#                 output_directory=output_dir,
#                 output_prefix=output_prefix)

# all_metrics = cc.validate(model_directory="/path/to/Geneformer",
#                           prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled.dataset",
#                           id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
#                           output_directory=output_dir,
#                           output_prefix=output_prefix)