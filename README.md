# MAEPy
Machine Learning deduplication pipeline 

This is an experimental deduplication model/pipeline that I developed during my 4 month Co-op placement with the Public Service Commission of Canada (PSC) during the summer of 2022.

The context and motivation behind the project was the need to remove duplicate accounts from an internal dataset, in which the previous model was built specifically using handcrafted rules. It seemed reasonable to try and create a more robust and adaptable model using Machine Learning both to improve upon the previous solution and also to encourage further exploration of Machine Learning techniques for future projects. 

The dataset in which the model was developed on was very messy, and so a lot of experimentation was needed in order to optimize the final product. The design of the pipeline is very experimental, and involves creating vector representations that embed the structured account records into a vectorspace where similar records have a smallar Euclidean distance. A trick used to optimize the vector space was to skew the embeddings with weights learned from a supervised classifier trained to detect duplicate accounts from a similar set of features used to create the vector representations. 

Note that in my eyes, there are a lot of improvements that could be made here, both in the design of the algorithm itself and the codebase. Although it is currently being used by developers at the PSC, I still feel that the project is unfinished and that a lot of work would need to be done in order to turn this pipeline into something that could be "released into the wild" and used in production. I ran out of time at the end, leaving sections like "preprocessing.py" unfinished with hardcoded logic and missing features.
