# UMaReRe2D

Recognition of Magnetic Reconnection structures in 2D dataset, via Unsupervised ML techniques (in particular using KMeans and DBscan).
The codes and their instructions referring to a journal article titled "Detecting Reconnection Events in Kinetic Vlasov Hybrid Simulations Using Clustering Techniques". This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 776262 (AIDA, www.aida-space.eu).

**TO DO**
* [x] upload script crossvalidation (FF)
* [x] upload script KMeans clusterization (FF)
* [x] upload script dataframe generation (MS)
* [x] upload script DBscan (MS)
* [x] upload script aspect ratio (MS)
* [x] upload script utilities (MS)
* [x] uniform scripts
* [x] decide if scripts, functions or objects
* [ ] separate functions in different modules: plots, preprocessing, clusterization, utilities, ...
* [ ] manage inputs via config file
* [ ] magage metadata via dictionaries and/or log files
* [ ] write documentation
* [ ] test (Jupiter notebook, with a subset of the data)
* [ ] porting to python 3.x

**IMPORTANT**
The simulation data-set (TURB 2D) is available at Cineca AIDA-DB. In order to access the meta-information and the link to "TURB 2D" simulation data look at the tutorial at http://aida-space.eu/AIDAdb-iRODS.

**INSTRUCTIONS**
Library "utilities_unsup.py" contains some functions used by the other scripts.
* "quantities_alldata.py" creates quantities that can be "correlated" to obtain regions interesting for reconnection. 
Basic fields (J,B,Ve,n,E) must be loaded in the format [3,nx,ny,nz], where nx,ny and nz are the grid dimensions.
* "KM_crossvalidation.py" is used to find the optimal K to be used for the KMeans algorithm.
* "KM_clusterization.py" KMeans algorithm.
* "DBscan_over_kmeans_cluster.py" DBscan algorithm applied to one selected cluster, among those found using KMeans algorithm.
* "clusters_aspect_ratio.py" computes the aspect ratio of the structures found using KMeans+DBscan.

**EXAMPLES**
In folder "output_examples" some examples have been collected. 


