# ECIR-2018
Medical Forum Question Classification using Deep Learning

       Raksha Jalan, Manish Gupta, Vasudeva Varma
------------------------------------------------------------------------------------------------------------------
Following are folder names having respective files.

1)Dataset: It contains both training and test dataset in tsv format.
           Subfolder named "Crawled_questions_for_Weak _Supervision" contains text file having questions those re crawled seperately from "med-help.org" where title and corresponding question is seperated by sign "<==>".

2)Self_training:Contains codes corresponding to basic self-training algorithm and self-training with lookups.
3)Supervised: Contains codes  for all supervised models mentioned in Table 2 of paper.
4)Weakly_Supervised:Contains codes for weakly supervised models mentioned in Table 3 of paper.

How to Run code?
For Supervised models:
  Inputs are Training,Test DataSet and Glove vectors.
  For SoA based networks: Aditional input is output file of MetaMap.Where it contains all medical entities of respective questions along with their CUIs. 

For Self_trainig Models:
  Inputs are Training,Test DataSet,Crawled Data and Glove,TF-IDF vectors.
  *Models stores tuned parameters those are learned from self training in pickle format.

For Weakly Supervised Models: 
  Inputs are Training,Test DataSet,Pickle files to load the same parameters those are obtained from self-training Models.  
Make changes in code to provide corresponding inputs and run the codes.
----------------------------------------------------------------------------------------------------------------
