# Datasets
Because of the lack of space in Git Storage for Large Files (LFS), I have to delete the large `.csv` files from the repository. I will leave the filtered `.csv` datasets with Taiwan data only, as well as `GlobalTemperatures.csv` files.

Please note for reference the source for temperature data is the [Berkeley Earth Surface Temperature](https://berkeleyearth.org/data/) dataset, available in Kaggle. You can always retrieve it and work locally with this data if you want to make some other experiments and require other countries data, you will not be able to push the experiment results and so on however not the large dataset files to GitHub.

Here it is also stored the dataset used as input to the LSTM model, that is `taiwan_clean_dataset.csv` which is the output of `data-summary.ipynb`.