# AsthmaGNN: Predicting Asthma Readmissions using a Heterogenous Graph Neural Network

To reproduce the results obtained by this code please follow the following steps.

1. **Get Access to MIMIC-III**  
   Obtain access to the [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/clinical) database, which contains 11 years of Electronic Health Record data.  
   *Note:* A MIMIC-III demo database is also available. It can be used to run these files but does not capture comorbidities, so the model may not perform as expected.  

2. **Prepare the Data**  
   - In `data_prep.py`, set the file path to your full MIMIC-III dataset.  
   - If you only have access to the demo database, use `data_prep2.py` with the demo dataset link.  

3. **Create Global Mappings**  
   Add the output file from the previous step to `create_mappings.py` to generate the global mappings for the graph data structure.  

4. **Generate the Graph**  
   Paste the updated `.pkl` file path into `create_graph.py` to create the graph file.  

5. **Run the Model**  
   Execute `create_model_ht.py` to run the model on the prepared graph.  
   *Note:* If the graph from the demo dataset doesn’t work, a pre-processed graph file is provided in the `results` folder.  

6. **Explore Results**  
   The `results` folder contains a complete run of all model configurations. Use `model_run_evaluation.ipynb` to explore the data and results.
