# machinelearning_JBG_SW
Here is a descrition of the files in our repository and their contents, by order of importance.
1. Main.jl : This is the file you want to test our results, it contains our best linear (Logistic Classification with ridge regularization) and non linear (Neural-Network) machines. TO GET OUR PREDICTION RESULTS, SIMPLY RUN THIS FILE IN ORDER.
    - Pkg.add("Optim", preserve = Pkg.PRESERVE_ALL)
      using Optim
      Pkg.add(url = "https://github.com/JuliaAI/MLJLinearModels.jl", rev = "a41ee42", preserve = Pkg.PRESERVE_ALL)
      (These are line 55-57 of the file, you will need to run them first (after lines 1-16) and then restart julia before running the whole file again)
    - This file also assumes that you have the train and test csv files in the same folder as our julia file, if you don't, please add them before running our code.
    - On line 18, change the path needs to be changed to the path towards the folder containing all the files in our git repository.
2. pca_Mach_Logistic_whole.jlso and nndeeper_mach_wholedata.jlso : the saved machines that can be used to run the code faster, they are not necessary and the lines of code to use them are indicated in comments in the Main.jl file
3. ML_Report.pdf : The report, in pdf format
4. Sabirna.jl and juliette.jl : These are simply the files we used for the project to avoid conflicting changes. They contain all the machines we tested, as well as some code for different visualization methods we tried. Specifically Sabirna.jl contains the KNN, logistic classification, PCA, and all neural networks. juliette.jl contains the tree and random forest, ridge and lasso regression, and clustering tests.
5. visualisation.jl : The codes used to produce the figures in the report, plus a few others. 
6. pca.py : This was a test to use PCA and euclidian distance to classify data, done with python, but it didn't give good results, and was not used for our final results. 
7. test.jl and test.py : Like their names suggests, these are our test files, which we used to get better acquainted with julia outside of a pluto notebook and with git, they contain no relevant code. 