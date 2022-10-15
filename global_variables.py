#! /usr/bin/env python 3

__author__ = "Michael Ripa"
__email__ = "michaelripa@cmail.carleton.ca"
__version__ = "1.0.0"
__date__ = "August 26th 2022"

''' Contains global variables used throughout the MAEPy pipeline. Update with respect to your local environment and project needs.'''

#Path to graduates csv or parquet file that will be searched for duplicates
GRADUATES_PATH = 'data/psrs_graduates.parquet'

#Key to be used for finding 'gold standard pairs' within the DUPLICATE_GRADUATES_PATH (default is PRI)
match_col = 'PRI'

#Unique identifier with respect to the dataset. Ensured to be kept in the outputted dataset
unique_identifier = 'GRDT_ID'

# Default columns to be used for training classifier and embedding graduates
default_cols = ['GVN_NAME', 'SRNM', 'PDOB', 'postal_code', 'accuracy', 'place_name', 'state_code', 'lat-long','EE_CMP','LANG_CTZN_CMP','ENABLED_IND','EMPT_TYP_CMP','EDUC_CMP','RESM_CMP','PSEA_IND','EE_EMPT_IND','TMPRY_RELOC_IND']

#This is an optional preprocessing setting that specifies which pair of columns tends to be transposed. In context of PSRS Graduates, the first and last name often are transposed, making it important to consider in the pipeline.
transposed_pair = ['GVN_NAME','SRNM']

#This allows for easier modification on which distance functions are used on what column. See similarity.py and distance.py for information about the specific implementations of the code and for the code name (again, feel free to modify & add more wherever you please).
default_comp_functions = {'GVN_NAME':'EDIT_DISTANCE', 'SRNM':'EDIT_DISTANCE', 'PDOB':'EDIT_DISTANCE', 'postal_code': 'EQUALITY', 'accuracy': 'EQUALITY', 'place_name' :'EQUALITY','state_code': 'EQUALITY', 'lat-long': 'EUCLIDEAN','EE_CMP': 'EQUALITY' ,'LANG_CTZN_CMP': 'EQUALITY' ,'ENABLED_IND': 'EQUALITY', 'EMPT_TYP_CMP': 'EQUALITY' ,'EDUC_CMP': 'EQUALITY' ,'RESM_CMP': 'EQUALITY' ,'PSEA_IND': 'EQUALITY' ,'EE_EMPT_IND' : 'EQUALITY' ,'TMPRY_RELOC_IND': 'EQUALITY' }

#Columns that will be preprocessed into datetime objects, allowing for numerical date comparisons. Currently not being utilized in the pipeline, but left in for future use-cases (default distance function computes time delta)
DATETIME_COLS = 'CDT'

# If using pygeocode, this is an important variable to ensure is set correctly. Cooresponds to the column name of the dataset containing postalcodes.
postal_code = 'PSTCD'

#Names designated to the columns of cluster and duplicate labels created within the pipeline
CLUSTER_COL = 'CLUSTER_LABELS'
DUPLICATE_COL = 'DUPLICATE_LABELS'

#Where outputted data is stored
OUTPUT_PATH = 'output/'
output_df_name = 'psrs_graduates_with_dups.parquet'

