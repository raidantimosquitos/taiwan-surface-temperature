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

For now I will move all Data summary to a new directory data-summary, you can refer to that directory if you have doubts about the dataset. 