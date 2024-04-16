# TACEI: A Trustworthy Approach to Classify and Analyze Epidemic-related Information from Microblogs

eval: cross-validation python main.py -mode eval 

train python run.py -mode train -saved_model_path ../data/saved_models/

prediction python run.py -mode prediction -saved_model_path ../data/saved_models/ -input_new_data_path ../data/unlabeled_data/new_data.csv -output_new_data_path ../data/output_data/new_data.csv

dependencies emoji==0.6.0 HTMLParser==0.0.2 nltk==3.5 numpy==1.21.5 pandas==1.1.5 scikit_learn==1.1.1 torch==1.9.0 transformers==4.2.1
