# Matrix_test

First test using Matrix Profile (MP) analysis. 

The library used is : https://matrixprofile.docs.matrixprofile.org/

### Data
UCDP Fatalities, gathered by country at a **monthly** level (any spacial/temporal level could be adapted). 

The data is normalized : (X-mean)/std

Only the sequences with values above zero during a period of  minimum 60 months (5 years) are kept, in order to have interesting dynamic for the Matrix profile. 

We are computing a range from 12 to 36 months window Matrix Profile. (also adaptable)  
We then extract the motifs, or shape, with the highest interest. 

### Figures 
For the two most interesting shapes, some figures are presented in *Motifs_figures* folder.
- In *General_shape_mot*, you'll find a mean (+/-std as confidence interval) of the normalized motifs (from 0 to 1). The motifs are in blue and the following five months are in red. 
- In *Individual_seq_mot*, you'll find every sequence related to a motif, with the country as title and the dates in x-axis. Also, a forecast comparison between an ARIMA model and the mean value of the other sequences of these motifs are printed. 
- In *MSE_horizon_mot*, you'll find the MSE improvement of the MP model compared to ARIMA model boxplot. Positive values correspond to a better score of the MP model. 
- In *MP*, you'll find the Matrix profile and the index of each sequence that belong to this motif (the red stars).   

# Motifs 

### First 
In the first motif, we can observe three peaks of fatalities through 36 months (3 years) with increasing trend. In the last two peaks, we can also spot a 'M' shape with an increase-decrease-increase dynamic.

### Second
In the second motif, we can see a little plateau from month 2 to 7 and then, a huge peak of fatalities in months 17-18. This sequence takes 18 months (1 year and a half). 
