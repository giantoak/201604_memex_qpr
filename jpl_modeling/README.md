# April 2016 QPR CP 1 - JPL Boosted Tree Pipeline

#### To run:
1. Add raw data files to ```raw_data``` folder (```ht_training_all```, ```ht_evaluation_NOCLASS```, ```ht_evaluation_UPDATED```)
    2. ```ht_training_all``` is combination of ```ht_training_UPDATED``` and ```ht_training_2```
2. Run ```flatten_raw_json.py``` for both training and eval data (takes raw json and puts into csv)
3. Run ```join_and_aggregate.py``` for both training and eval data (joins together features from various groups and rolls all data up to phone number level)
4. Run ```modeling.r``` to build model and run on training and eval data
5. Run ```add_eval_class.py``` to add actual class to ```eval_predictions.csv``` (generated from modeling.r)
6. Generate ROC stats with ```model_evaluation.r```

Contact: kyle.a.hundman@jpl.nasa.gov