Repository that constains the GNN-based models to predict the adsorption and reaction energies with the surrounding localized chemical enviroments features.

requirements.txt contains each version of the libraries implemented on the python enviroment to run the code.

Each Folder constains the processed databases to run each model.

Folders:

1_CatBoost_normal -> Contains CatBoost model for predicting adsorption energies of bimetallic alloys.

2_reg_alloys -> Contains GNN model for predicting adsorption energies of bimetallic alloys.

3_reg_NiGa -> Constains GNN model for predicting reaction energies of key intermediates of CO2 hydrogenation to methanol on Ni-Ga-based.

4_clustering -> Contains a clustering model for node attributes and several physicochemical properties. Before running this model, it is necessary to first run both GNN models (alloys and Ni-Ga-based).