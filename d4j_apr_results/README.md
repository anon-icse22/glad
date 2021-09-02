# APR repair results over D4J

We present the raw data that is the basis of our analysis on what bugs have been fixed by APR tools on the Defects4J dataset. 

## Structure

 * `all_bug_stats.csv`: contains all repair results, along with features from Defects4J-dissection and character-level added/removed features that we obtain. The character-level features are obtained using the git command `git diff --word-diff=porcelain HEAD^ -- . ':(exclude).defects4j.config'`. 
   * In the .csv file on columns with APR tools, the number 3 indicates correctly fixed, the number 2 indicates plausibly fixed, and the number 1 means partially fixed (i.e. some, but not all, failing tests pass while all originally passing tests still pass). Only correctly fixed results are reliable at this moment.
   * The source for each result can be found in the supplementary material of our submission.
   * There are some caveats regarding the feature data. Some data from Defects4J-dissection is inaccurate; for example, Closure-51 also adds a new method `isNegativeZero`, but this is not reflected in the feature data (it is correctly reflected in the character-level data). On the other hand, the character-level data is an overestimate, as noted in our submission.
 * `APR-analysis.ipynb`: based on `all_bug_stats.csv`, we perform simple analysis on the features used. Figure 1 of the paper is drawn here.
 * `unfixed_categorization.csv`: The raw categorization data for Table 2 is provided.