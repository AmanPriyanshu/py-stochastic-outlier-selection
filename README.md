# py-stochastic-outlier-selection
Stochastic Outlier Selection (SOS) is an unsupervised outlier selection algorithm. It uses the concept of affinity to compute an outlier probability for each data point.

For more information about SOS, see the technical report: J.H.M. Janssens, F. Huszar, E.O. Postma, and H.J. van den Herik. Stochastic Outlier Selection. Technical Report TiCC TR 2012-001, Tilburg University, Tilburg, the Netherlands, 2012.

All references and code usderstanding has been taken from: https://pure.uvt.nl/ws/portalfiles/portal/1517370/Janssens_outlier_11-06-2013.pdf on pg. 62 (in pdf)[search pg.82]





Let d be the dissimilarity measure (e.g., the Euclidean distance as stated by Equation .). Let h
be the perplexity parameter, where h ∈ [1, n−1]. en matrix D is the dissimilarity matrix of
size n×n that is obtained by applying dissimilarity measure d on all pairs of data points in data
set X. en {σ
2, 1, . . . , σ2, n
} are n variances, whose values are determined using perplexity h. en matrix A is the affinity matrix of size n × n as obtained by applying
Equation . with the n variances to each cell of the dissimilarity matrix D. en matrix B is
the binding matrix of size n×n as obtained by applying Equation . to each cell of the affinity
matrix A.
