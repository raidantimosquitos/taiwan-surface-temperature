# Study of Taiwan Land Surface Temperature

**UPDATE:** *Due to lack of storage space on Git Large File Storage (LFS) I will have to remove the original datasets from the repository. It topped up the amount of free space to use, I will just keep the Taiwan related datasets. You are welcome to check the transformations applied to the dataset on the data-summary folder.*

After the meeting on 10th Nov., we have decided to focus our effort on working only with the Berkeley's Earth Surface Temperature dataset (instead of using also Air Quality Index). Our plan is to build various models to perform regression tasks. The target of the models being average temperature and the input features being the rest of the fields in the dataset (datetime, city, latitude and longitude).

Some of the possible models that can be explored are:
- Linear regression.
- Random Forest.
- Polynomial regression.
- Gradient boosting model (as explained by professor on the last class).
- Adaboost model.

You can also add your models or other scopes if you have.

As pointed out by Minh there is work to do in the following areas:
- Handling missing values (Nulls).
- Categorical encoding for features such as City.
- Handling outliers (setting up thresholds on the features to not train with outlier data)

What would be nice to do if we have time is also develop some time series forecasting model. To do this we might need more data samples, but we can always use data up-sampling methods to interpolate more samples and increase our dataset size. 

## Repostory structure
### Datasets
Directory for storing all dataset files. For now Taiwan related only. 
*Possible `TODO`*: Include new features or new dataset files for different types of models (possibly include temperature anomalies).

### data-summary
Jupyter notebooks performing Exploratory Data Analysis. It includes a set of figures to help illustrate the dataset, also small transformations applied to the coordinates in the dataset.

### Source code
This directory includes the source code to run the program. 

### Checkpoints
Directory to save the best models after training, so it can later be used for testing

### Logs
Small folder to store possible logs during training, or other procedures of the code.

### Miscellaneous: set-up-github 
Small guide on how to set up Git and GitHub in your computer in case you want to collaborate on the repository.