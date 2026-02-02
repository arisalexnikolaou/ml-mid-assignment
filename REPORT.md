# ML Machine Learning Mid Term Assignment

Student Number  2417877
Student:        Aris Nikolaou

## Problem and Dataset

This report analyses weather patterns using Kaggle Weather data from World Ward II to create an optimal model using LSTM to predict mean temperatures.  The license to the data Kaggle data follows the  U.S. Government Works license.

The key features of the dataset are the station (location) identifier for time series data, the mean, max, min temperatures and date-oriented features such as day of year (DA). Snowfall and Precipitation were also evaluated.

Several columns in the original data set were absent any data and dropped from the final dataset evaluated. Precipitation and Snowfall included the value T (Trace) as well as numeric values which was converted to 0 to ensure the feature represented a continuous numerical value to drive the regression analysis. Data was sorted by the ‘Date’ field and all mean temperature values were forward filled and then backfilled with data to present a continuous set of data to the model. A ‘DayOfYear’ feature column was extracted.

All rows and columns with complete null values were removed as well as duplicate records. Numeric like fields were coerced into numeric fields as a general principle.
Unlike other values in the dataset the STA column is categorical in nature. In the experiments below models were developed for all stations using a tactical measure the weight the station id more heavily than other features when scaling. Other experiments developed a model per station in its pre-processing steps.
The dataset is relatively small and sparse and not necessarily ideal for machine learning exercises developing complex neural networks.

## Model and Training

The LSTM model was selected because of its ability to track seasonality in temperatures and overcome classic vanishing gradient issues adversely affecting typical Recurrent Neural Networks. LSTM’s feature memory and forget gates capable of tracking this behaviour.

One baseline experiment was created using Linear Regression as well as 3 alternative implementations of LSTM. The ‘naïve’ LSTM implementation uses a single layer network (128 neurons) blending data from multiple stations as well as two other LSTM implementations using a two-layer network focusing on tracking weather from a single station rather than modelling on the entire dataset.

All LSTM models used the same common configuration parameters when possible. The LSTM models were trained over a maximum of 50 epochs. 70 % of the data used training, 15 % for validation and 15 % for the test set. A lookback window of 60 (days) was used to create the sequences that was used to test/train the data sets.
A ‘Dropout’ rate of 0.2 was introduced as a layer in all LSTM model implementations to prevent overfitting the model. The activation function used in all LSTM networks is ‘RELU’ and Adam chosen as an optimizer.

The models were initialized with a consistent random seed of 42 to ensure consistency in behaviour in NumPy and TensorFlow.
Training incorporated early stopping with a configurable patience of 10 as well as model checkpointing to persist the best trained model to disk to rely on for prediction.

## Results

The primary measures used to determine accuracy were RMSE as well as MAE (mean absolute error) as well as r2 and MSE. In weather prediction RMSE is relied on more heavily as it penalizes large errors more heavily other measures.

A common graph, Actual vs Predict MeanTemp was created to visualize the performance of each model. This graph measures the de-scaled predicted MeanTemp against the test data used to create these predictions.

A residuals analysis graph measuring the distribution of results ideally around zero to identify systematic bias for the LSTM By Station Multivariate model.
The LSTM models took about an hour on a GPU enabled power workstation with the following results:

### Linear Regression Baseline

#### Train

RMSE: 	0.5152
MSE: 		0.2654
MAE: 		0.1659
R²: 		0.9959

#### Test

MSE: 		0.2497
RMSE: 	    0.4997
MAE: 		0.1661
R²: 		0.9962
 
See predictions graph: /results/figure-1-linear-predictions.png

### LSTM Naïve

#### Test

MSE: 		49.6010
RMSE: 	    7.0428
MAE: 		4.5681
R²: 		0.0844

See predictions graph: figure-2-lstm-naive-predictions.png
See residuals graph: /results/figure-3-lstm-naive-residuals.png
 
### LSTM Enhanced by Station Univariate

#### Train

RMSE: 	    0.8704
MSE: 		0.7576
MAE: 		0.6663
R²: 		0.7926

#### Test

RMSE: 	    0.8075
MSE: 		0.6520
MAE: 		0.6205
R²: 		0.6786

See predictions graph: /results/figure-5-lstm-mv-predictions.png


### LSTM Enhanced by Station Multivariate

### Train

RMSE: 	    1.7074
MSE: 		2.9151
MAE: 		0.9916
R²: 		0.9621

#### Test

RMSE: 	    1.8531
MSE: 		3.4341
MAE: 		1.0616
R²: 		0.9317

See predictions graph: /results/figure-5-lstm-mv-predictions.png
See residual analysis graph: /results/figure-6-lstm-mv-residuals.png
 
As the dataset is quite small it appears that the linear regression model performed the best.

The worst performing regression model was the LSTM Naïve using a single layer that did not categorize it’s dataset by station id by a significant margin. When factoring in the STA category the best LSTM model used a univariate MeanTemp feature which performed better than the multi-variate approach most likely because the dataset is relatively small and omits other critical factors like wind and humidity.

## Discussion & Limitations

Both the LSTM Univariate and Multivariate models scored better on the train rather than the test set suggesting overfitting. Another experiment increasing the dropout rate may improve the scores.

Ad hoc experiments adjusting the test train split and ratios, the dropout rate and batch size were tried that yielded, generally, poorer results as well as well as the lookback period.

A limitation of the LSTM Enhanced models that categorize datasets by station is that it doesn’t model on all stations but just the station with the highest data population unlike the Linear Regression and LSTM Naïve approaches. This can be overcome by simple aggregating the predictions from all stations into another data frame or using a Kera ‘embedding’ to rely on STA (station) as a categorical value.

The LSTM univariate approach yielded better results than the multi-variate alternative, counterintuitively, most likely given the small size of the dataset worked with. More experimentation would need to be invested to enhance the multi-variate model to influence seasonal behaviour.
The biggest influence on the model accuracy among the LSTM implementations as the use of a dual layer network (Salman, 2018) as well as the categorization of training data by station id.

## Ethics and Responsible ML

Give the public nature of the dataset and the time since it was published there are no significant ethical considerations to be evaluated when constructing the solution. There is no PII data and no (political) bias in the underlying dataset to be cautious of.

### References:

- GeeksforGeeks (2025) What is LSTM Long Short-Term Memory? <https://www.geeksforgeeks.org/deep-learning/deep-learning-introduction-to-long-short-term-memory/>.
- Salman, A. (2018) Single Layer & Multi-layer Long Short-Term Memory (LSTM) Model with Intermediate Variables for Weather Forecasting. <https://www.sciencedirect.com/science/article/pii/S187705091831439X#abs0001> (Accessed: January 14, 2026).
- Team, K. (no date) Keras documentation: LSTM layer. <https://keras.io/api/layers/recurrent_layers/lstm/>.
- Tensorflow Technical Docs (no date). <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM> (Accessed: January 16, 2026).