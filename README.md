# py-stochastic-outlier-selection
Stochastic Outlier Selection (SOS) is an unsupervised outlier selection algorithm. It uses the concept of affinity to compute an outlier probability for each data point.

For more information about SOS, see the technical report: J.H.M. Janssens, F. Huszar, E.O. Postma, and H.J. van den Herik. Stochastic Outlier Selection. Technical Report TiCC TR 2012-001, Tilburg University, Tilburg, the Netherlands, 2012.

All references and code usderstanding has been taken from: https://pure.uvt.nl/ws/portalfiles/portal/1517370/Janssens_outlier_11-06-2013.pdf 

Please refer to a mock implementation of the code over here: https://github.com/AmanPriyanshu/py-stochastic-outlier-selection/blob/master/SOS_outlier_detection.ipynb
Use the code here for implementation on dataset: https://github.com/AmanPriyanshu/py-stochastic-outlier-selection/blob/master/py_SOS.py


# Understanding Stochastic Outlier Selection
Updated: Apr 10

Outliers are data points/observations which defect or diverge from the general pattern. In this article, we will implement the Stochastic Outlier Selection Algorithm for Pseudo Dataset. The dataset will be generated with two features and hence the outlier here will be multivariate (i.e. Having or involving multiple variables). The SOS algorithm is an unsupervised outlier selection program. It was fashioned and engineered by Jeroen Janssens and his publication (https://research.tilburguniversity.edu/en/publications/77b71572-7266-44d0-9510-ebf4eeb43062) here details the algorithm, its implementation. All references and justifications and understanding was developed from the cited paper. This article attempts to implement that SOS algorithm. Following is my GitHub Repository of the code below: https://github.com/AmanPriyanshu/py-stochastic-outlier-selection/blob/master/SOS_outlier_detection.ipynb


## DATASET:

We will be generating our own dataset since this is an implementation-oriented article. We will evaluate this dataset and look at potential use cases for this algorithm following our implementation. 

Defining our dataset: Our data will consist of 50 points plotted in 2-D space. They will be generated using the following:

This will generate us a list of fifty 2-D points, this we can use as data. As we can see most of the points will be closely linked to each other as they are restricted within the range 0 to 100. However, some points precisely 5 will be an outlier as they will deviate from the general trend.

Let us see the plot of any such instance of the data.

![Not found](https://static.wixstatic.com/media/a27d24_0027de9423c1424e98763ab64ec1474f~mv2.png/v1/fill/w_545,h_389,al_c,lg_1,q_90/a27d24_0027de9423c1424e98763ab64ec1474f~mv2.webp "Data Plot")

We can clearly differentiate between the cluster and outliers here. Now let us begin with normalizing this dataset. Normalization is a technique often applied as part of data preparation for machine learning. The goal of normalization is to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values. Thereby, any feature which inherently has large values will not bias or over-influence the clustering algorithm. Normalization will be enacted using the code:

```
normalized_data = [(i-min(i))/(max(i)-min(i)) for i in data.T]
normalized_data = np.array(normalized_data)
```

Once this data is normalized we are ready to begin our clustering algorithm.


## CLUSTERING:

### Step 1: DISSIMILARITY MATRIX:

A dissimilarity matrix is one where the euclidean distance of each point w.r.t. every other point is generated it is done feature-wise and then per data-point is simply summed.

It is given by the following formula:

![Not found](https://static.wixstatic.com/media/a27d24_fadd2ff3cb7f4e6d9ab71d082fddb6de~mv2.jpg/v1/fill/w_238,h_122,al_c,lg_1,q_90/a27d24_fadd2ff3cb7f4e6d9ab71d082fddb6de~mv2.webp "Distance")


It is implemented in the following manner:

![Not Found](https://static.wixstatic.com/media/a27d24_5cb04252429f4c658bc6b860e54cf105~mv2.png/v1/fill/w_1156,h_263,al_c,lg_1,q_90/a27d24_5cb04252429f4c658bc6b860e54cf105~mv2.webp "code")


Once Calculated the dissimilarity matrix gives us the euclidean distance of all the points w.t.r. to each other. For example for the dataset:

```
[[827, 823], [17, 24], [1, 8], [99, 64], [8, 21], [75, 58], [81, 87], [98, 16], [25, 58], [91, 43], [590, 877], [25, 2], [69, 8], [6, 15], [49, 16], [73, 21], [60, 1], [68, 63], [31, 78], [34, 39], [839, 590], [94, 12], [46, 90], [42, 31], [53, 86], [97, 23], [81, 34], [74, 91], [56, 73], [20, 96], [590, 670], [95, 26], [42, 37], [79, 92], [53, 54], [27, 86], [56, 48], [80, 26], [15, 69], [51, 97], [640, 588], [100, 5], [35, 28], [17, 25], [23, 81], [26, 73], [26, 63], [73, 39], [33, 8], [53, 59]]
```

The dissimilarity matrix is:

```
[[0.0, 1.766217888384938, 1.8371450242899034, 1.5054160703925716, 1.7933542389323973, 1.567912711925819, 1.4983892677154649, 1.605445435923194, 1.6785580146980745, 1.5642102848809358, 0.0837849819214349, 1.7942978866773913, 1.683762439319437, 1.8106133331930878, 1.7105983457380163, 1.6477566973371682, 1.7182396338696746, 1.5730379695768253, 1.6255497481403067, 1.696469178025393, 0.07095134037040593, 1.6222069616700194, 1.5687523461464878, 1.6949225363941718, 1.5609145114746337, 1.5928640154724705, 1.6037157308834553, 1.505679316987901, 1.5795052821201911, 1.6161312001939918, 0.11049027372102202, 1.5907844564176497, 1.6825843866475028, 1.4930856514799722, 1.6237155373081236, 1.619190552280963, 1.6291875760673566, 1.6223760399698053, 1.6797672771791816, 1.544357465133975, 0.12176210675161975, 1.6245923947672618, 1.716846403535083, 1.7641367703932365, 1.63796371389496, 1.6466614272777147, 1.6663388735361329, 1.6105545956295306, 1.763324528849572, 1.6137269528332376]...[1.6137269528332376, 0.00344186150299917, 0.007239986193299288, 0.0030457762453782898, 0.00476535034798271, 0.0006905222739480543, 0.002138084507891524, 0.005293122913191943, 0.0011177242151536749, 0.002389869730567409, 1.2826030007674556, 0.005350329873918088, 0.003754018353061423, 0.005668513469657818, 0.0024322939135849684, 0.0024513398331189305, 0.004453546463641012, 0.000341251730085321, 0.0011596534430229272, 0.0010353231904450674, 1.2471807614622268, 0.005272395856351038, 0.0013220959100662326, 0.0011939682178271022, 0.0009499906173766184, 0.004445748735916484, 0.0019308849082167886, 0.0019624044011630135, 0.0002682319169547108, 0.003334744631197241, 0.897129647417859, 0.003931069200740542, 0.0008030255769313743, 0.002381750160634824, 3.2578553407977314e-05, 0.0019126189915599505, 0.00017049625673077882, 0.00245722250358115, 0.0021865795573016308, 0.0018874332707275111, 0.8553410841672698, 0.006945592763250528, 0.0017136976895047182, 0.0033519446955931525, 0.0019123266175953028, 0.001293516575848201, 0.0010589509913107643, 0.0010908594428017978, 0.0039590752848401205, 0.0]]
```

Now here take a look at the first and the last row, the first row being an outlier has most of its distance greater than 0.9 breaking the sequence with a 0.08 when another outlier is compared. Similarly, the last row which has an index 49 is a normal case. Therefore, most of its distance is very low even below 0.01, however, when compared to an outlier such as the first datapoints it gives a value greater than 1 (Eg: 1.613 when compared to first data-point). This is the importance of the distance matrix, it highlights all points w.r.t. each other.


### Step 2: AFFINITY MATRIX

Now let us create the affinity matrix the most important part of SOS. Here we employ affinity in order to quantify the relationship from one data point to another data point. 

We note that the affinity that the data point x[i] has with data point x[j] decays Gaussian-like with respect to the dissimilarity d[i][j], and a data point has no affinity with itself, i.e.,
a[i][i] = 0.

![Not FOund](https://static.wixstatic.com/media/a27d24_b7102d77d2fd4ed8968cf93c5172217a~mv2.png/v1/fill/w_906,h_297,al_c,lg_1,q_90/a27d24_b7102d77d2fd4ed8968cf93c5172217a~mv2.webp "affinity")

The value of each variance σ2[i] is determined using an adaptive approach such that each
data point has the same number of effective neighbours, i.e., h (perplexity). The adaptive approach yields a different variance for each data point, causing the affinity to be asymmetric. Now, here the perplexity parameter h can be compared with the parameter k as in k-nearest neighbours, with two important differences. First, because affinity decays smoothly, ‘being a neighbour’ is not a binary property, but a smooth property. Now we need to adapt the variance in such a way that the perplexity is the same for every row. We take perplexity = 25 here. Therefore, the implementation of it in code:

![Not Found](https://static.wixstatic.com/media/a27d24_0143e1e760c34487b6365bfbc8e31048~mv2.png/v1/fill/w_1112,h_779,al_c,q_90/a27d24_0143e1e760c34487b6365bfbc8e31048~mv2.webp "affinity code")

and

![Not Found](https://static.wixstatic.com/media/a27d24_6d7e14255709496cb193beaac59230bd~mv2.png/v1/fill/w_1156,h_175,al_c,lg_1,q_90/a27d24_6d7e14255709496cb193beaac59230bd~mv2.webp "Affinity code 2")


Let us take a look at the output of the first (outlier) and last (normal) row/data-point:

affinity_matrix:

```
[[0.0, 0.16526990980159043, 0.15374404405838082, 0.21559377538481092, 0.16076149648491322, 0.2022890794761313, 0.21714337898380143, 0.19469679089841166, 0.18071571900232183, 0.20305388173137262, 0.9181487741004127, 0.1606069519546976, 0.17975965309239111, 0.1579582818855593, 0.1749095253948683, 0.18647901060688127, 0.1735525859278526, 0.20123511778166098, 0.19074788661105468, 0.17744658936174207, 0.9302374004529186, 0.19139888353835063, 0.20211603916878085, 0.17772653275902464, 0.20373711467469016, 0.1972095249031128, 0.1950403363853837, 0.21553593765406087, 0.19991300783077334, 0.19258781285442725, 0.8934949753799067, 0.19762796182999265, 0.17997562049768667, 0.21832034397473496, 0.19110481864674683, 0.19198822633366272, 0.19004194703482913, 0.1913659027997556, 0.18049312194186806, 0.2072044283708682, 0.8832887567030782, 0.19093410151305154, 0.1737992085581755, 0.16562084077268355, 0.18834962688244059, 0.1866872985482597, 0.18298043793358834, 0.19368556422236768, 0.16575800804057245, 0.19306032251944427]... [1.0706640257875913e-196, 0.38196338613776937, 0.1320623473760576, 0.4266995332648199, 0.2638139679583371, 0.8244100075455878, 0.5499863163002735, 0.22761731695123086, 0.7315837065737711, 0.5125959718216472, 1.7426560545168542e-156, 0.22400520495420767, 0.3500366618033484, 0.2049358903091943, 0.50655104242842, 0.5038604667963039, 0.28784793841552164, 0.9089887918987106, 0.7230563693148032, 0.7486361106276839, 3.4903908146493835e-152, 0.22894037252223287, 0.6909477529515976, 0.7161516284169217, 0.7667142316430542, 0.2884762582212923, 0.5827927221561663, 0.5776787673269365, 0.9277394508586955, 0.39357722944453555, 1.1296382820870298e-109, 0.3331290837394312, 0.7988788058993003, 0.513761109709143, 0.9909315903168462, 0.5857770131163035, 0.9534435908478411, 0.5030323269969724, 0.5425786057292041, 0.5899169419624148, 1.3419167617549811e-104, 0.14339368686688225, 0.6192831271600536, 0.39168883780781033, 0.5858249053146242, 0.6964915816230395, 0.7437062324757587, 0.7371001025257826, 0.3305304712462848, 0.0]]
```

Take a look at the first row which is an outlier. Here, we can see that the outlier has low values for every normal case it is in the order of 0.1 or 0.2 generally. However, any other outlier has a value near 0.9 or 0.8. On the other hand, the last row which is a normal data-point has a very very low affinity towards outlier even equal to e^-196 in some cases. Whereas, all normal cases have a value greater than 0.5 even in the range of 0.5 to 0.7.


Step 3: BINDING MATRIX:

The binding matrix is a row-wise normalized version of the affinity matrix. It is given by the following code:

![Not Found](https://static.wixstatic.com/media/a27d24_20fcac514c494ca2854e4da6e0bb0f30~mv2.png/v1/fill/w_1156,h_113,al_c,lg_1,q_90/a27d24_20fcac514c494ca2854e4da6e0bb0f30~mv2.webp "binding")

Now let us take a look at the matrix it gives as output:

```
[[0.0, 0.013675174748784148, 0.012721472841651506, 0.01783919744784078, 0.013302128378641377, 0.01673830714210361, 0.017967418610622198, 0.016110087079733156, 0.014953230386437477, 0.016801590316283622, 0.07597175400094675, 0.013289340669979197, 0.014874121198289713, 0.013070177810322383, 0.014472799845254737, 0.015430111023179933, 0.014360520578222773, 0.016651097616983704, 0.015783337000056463, 0.014682727914662498, 0.07697202124887915, 0.015837203410175412, 0.016723989009756097, 0.0147058916888922, 0.016858124079176157, 0.016318001978777714, 0.016138513576571808, 0.017834411694184656, 0.01654170030057444, 0.015935580762597393, 0.07393178794705937, 0.016352625329775456, 0.01489199131155476, 0.01806480598103669, 0.01581287116007016, 0.015885968280456426, 0.015724924388356265, 0.015834474435710715, 0.014934811705723453, 0.017145025189977337, 0.07308727956620836, 0.015798745257546844, 0.014380927242513442, 0.013704212353755435, 0.015584894216847214, 0.015447345703103424, 0.015140623404193278, 0.016026413642008557, 0.013715562189672003, 0.015974678334849682]...[4.509741909841889e-198, 0.01608867253407052, 0.005562595625978932, 0.017973002937674995, 0.011112103134577802, 0.03472495827238146, 0.023165963184710252, 0.0095874646848775, 0.030815023411907786, 0.02159104519496415, 7.34023825794769e-158, 0.009435318984044455, 0.014743887584665005, 0.008632100744005824, 0.021336426838784502, 0.021223097153677848, 0.012124437547802682, 0.038287499641978466, 0.030455843601155903, 0.03153328739923956, 1.4701868522174763e-153, 0.00964319308345901, 0.029103397181060055, 0.030165009140916427, 0.03229475559122664, 0.01215090299440983, 0.024547801182776635, 0.02433239638852803, 0.039077295791959246, 0.01657785900744934, 4.758147262809355e-111, 0.014031723810107607, 0.03364955901267261, 0.021640121945797532, 0.04173901069806493, 0.02467350244563725, 0.040159979384323544, 0.02118821509291648, 0.022853943148417766, 0.02484788030994756, 5.65228504390447e-106, 0.006039882761491192, 0.026084812838997, 0.016498318099179163, 0.02467551971201325, 0.029336908683253223, 0.03132563609522011, 0.03104737942104677, 0.013922267702631125, 0.0]]
```

Taking a look at the first row (which is an outlier) we can see that most of the values are of the order of 0.01 on the other hand upon outlier data-point it is in the order of 0.07. In the case of a normal data-point, we can see that normal-points are of the order 0.01 or 0.02 and outliers are very low even to the order of e^-100. This clearly highlights the value of the binding matrix which allows us to analyse the bound-affinity of all the data-points w.r.t. each other.


Step 4: OUTLIER PROBABILITY MATRIX:

Now, the probability matrix gives us the probability of whether a particular data-point is an outlier or not. It is given by the mathematical formula:

![Not Found](https://static.wixstatic.com/media/a27d24_cf8bf0ec21b54251948a73038a395b22~mv2.png/v1/fill/w_389,h_106,al_c,lg_1,q_90/a27d24_cf8bf0ec21b54251948a73038a395b22~mv2.webp "outlier")

Here we can clearly see that what we are attempting to do is we are calculating the product of the probability with which data-point[i] is not a neighbour of data-point[j] where j ranges from 0 to n-1 except i. But since j=i will anyways give us (1-0.0) it outputs 1 and hence does not affect the probability that a particular data-point is an outlier or not.

The code implementation is given by:

![Not FOund](https://static.wixstatic.com/media/a27d24_636fef55f16f497aa66b762ab46afe4a~mv2.png/v1/fill/w_1156,h_126,al_c,lg_1,q_90/a27d24_636fef55f16f497aa66b762ab46afe4a~mv2.webp "outlier code")

Let us take a look at the output, as this is the final probability of whether a data-point is an outlier or not:

```
[0.7383609046814931, 0.349245148521586, 0.4843037458574838, 0.38995196375942603, 0.40151507251977814, 0.27978882761799634, 0.38167677830089847, 0.414264994504904, 0.29581977528612574, 0.33214200318486636, 0.7423157630163191, 0.40163062529616933, 0.3427893712689883, 0.43085668050663306, 0.29615886198590347, 0.30404607496044, 0.3639295228462525, 0.271059234615385, 0.3246529731564139, 0.26604938480758605, 0.7388204287748368, 0.4087152696419914, 0.3476761448565691, 0.26241192966658866, 0.3269999523844157, 0.3896305774937709, 0.2995670199864767, 0.3786160023490368, 0.2803100013423754, 0.4374623652563397, 0.7214365702632334, 0.372764112150474, 0.25386139886019343, 0.3978549727427733, 0.24471147175244654, 0.3672290868445459, 0.24286122327678045, 0.311042293120391, 0.3574368257959728, 0.38060459330671426, 0.7244097819403164, 0.46393703973576, 0.2802584428226958, 0.34675019645176863, 0.3609698523912664, 0.3238516983070249, 0.30011494756824963, 0.2702975617451561, 0.35069966867760716, 0.24986131247865667]
```

Now take a look here, we can see the different probabilities with which a particular data-points are an outlier or not. We can see that the first point is an outlier with the probability of 73.83% whereas the second is an outlier with a probability of only 34.92% and hence we can conclude that it is not an outlier but rather part of the cluster. We can create an if-condition where every data-point having a probability above 70% is categorised as an outlier.


Step 5: CLASSIFIER:

We can represent the output as:

![Not FOund](https://static.wixstatic.com/media/a27d24_de82c968d1f84b4e9a5db6c92a5d07a2~mv2.png/v1/fill/w_1156,h_308,al_c,q_90,usm_0.66_1.00_0.01/a27d24_de82c968d1f84b4e9a5db6c92a5d07a2~mv2.webp "classifier")

We can clearly see the accuracy of the SOS algorithm with the only parameter to set up being perplexity (used: 25, here).

CONCLUSION:

 SOS is a very useful outlier selection algorithm and it has far superior performance than 1) KNNDD 2)Outlier-score plots 3)LOF 4)LOCI 5)LOCD. It is clearly a very useful outlier selection algorithm and can be used in many different datasets.
