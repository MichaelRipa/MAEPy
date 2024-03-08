# MAEPy: Machine Learning Deduplication Pipeline

MAEPy is an experimental deduplication pipeline developed during a 4-month Co-op placement with the Public Service Commission of Canada (PSC) in the summer of 2022. The pipeline is designed to remove duplicate accounts from an internal dataset, providing a more robust and adaptable solution compared to previous models built using handcrafted rules.

# Context and Motivation
The motivation behind this project was to explore Machine Learning techniques for deduplication tasks, especially in the context of messy datasets. The pipeline employs a novel approach of embedding structured database records into a vector space where similar records have smaller Euclidean distances. It further optimizes this vector space by skewing the embeddings with weights learned from a supervised classifier trained on similar features.

# Features

- **Vector Space Embedding:** Structured records are transformed into vector representations, making it easier to identify duplicates by measuring Euclidean distances.
- **Weighted Embeddings:** The embeddings are skewed with weights derived from a supervised classifier, enhancing the model's ability to recognize duplicates.
- **Flexible Preprocessing:** A series of preprocessing steps are included to clean and prepare data for the deduplication process.
- **Modular Design:** The pipeline comprises several standalone scripts (e.g., similarity measurement, clustering, and classification), each addressing specific aspects of the deduplication task.

# Datasets
The pipeline is designed to work with general datasets, but it requires the following:

Main Dataset: The primary dataset that you want to deduplicate. The dataset should be in a format compatible with pandas DataFrames.
Duplicates Dataset: A separate dataset used to train the classifier for identifying duplicates. This dataset should have the same structure as the main dataset.

# Column Specifications
Columns for Comparison: Specify the columns used for comparison in the default_cols variable in global_variables.py.
Match Column: Define the column used to find 'gold standard pairs' within the duplicates dataset in the match_col variable.
Unique Identifier: Ensure that a unique identifier column exists in your dataset, specified by the unique_identifier variable.

# Installation

Clone the repository:

`git clone https://github.com/yourusername/MAEPy.git`

Navigate to the project directory:

`cd MAEPy`

Create and activate a virtual environment:

`python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
`
# Usage

To use the deduplication pipeline, follow these steps:

1. **Preprocess the Datasets:** Ensure that your main and duplicates datasets are preprocessed and formatted correctly.
2. **Configure the Pipeline:** Update the global_variables.py file with the appropriate column specifications and paths to your datasets.
3. **Run the Pipeline:** Execute the main script (maepy.py) to start the dedupe.





