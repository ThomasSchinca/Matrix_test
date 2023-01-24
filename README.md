# Matrix_test

First test using Matrix Profile (MP) analysis. 

The library used is : https://matrixprofile.docs.matrixprofile.org/

### Data
UCDP Fatalities, gathered by country at a **monthly** level (any spacial/temporal level could be adapted). 

The data is normalized : (X-mean)/std

Only the country with less than 50 zero value are kept, in order to have interesting dynamic for the Matrix profile. 

We are computing the 12 months window MP, corresponding to one year. (also adaptable)  

### Figures 
- In *Figures_Matrix_analysis*, you'll find the images related to the MP (cumulated normalized time series (TS), Matrix profile, 
and the specific motif location in TS and MP) 
- In *motifs_mean_std*, you'll find a mean (+/-std as confidence interval) of the normalized motifs (from 0 to 1). 
- In *specif_motifs*, you'll find every sequence related to a motif, with the country as title and the dates in x-axis. 

In *motifs_mean_std* and *specif_motifs*, the motifs are in blue and the following three months are in red. 
It's just to check if there is a potential forecasting power. 
