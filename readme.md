This is the repository used for my M6 competition submissions. 
After some deliberations, I decided to publish it as is, without any attempts to refactor the admittedly quite horrible code I had written over the span of the competition.
This is because the codebase evolved during the competition, and any attempts to retroactively polish it would likely make replicating the results more difficult, rather than less.

To compensate for this, I detailed the approach in two additional working papers and repositories:
-   MtMs meta-learning method which I used for the forecasting challenge:
    - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4355794
    - https://github.com/stanek-fi/MtMs_sinusoidal_task
-   Rank optimization method which **ex-post** formalizes what I was attempting to do in the investment challenge:
    - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4527154
    - https://github.com/stanek-fi/rank_optimization


This repository is intended to be the exact opposite: unaltered snapshots of the code aimed at maximum transparency, serving as a testament to the poor coding habits I practice when no one else is looking.
You have been warned :)

Explanation of the code-base step by step:

1.  Run `R\GenData\GenStocksNames.R`.\
    The MtMs meta-learning method leverages not only assets from the M6 universe but also other assets to make the learning process more stable.
    This function fetches all available asset tickers and saves them to `Data\StockNames.RDS`.
    It needs not be run at every submission, only when one wishes to change the universe on which the model is trained.

2.  Run `R\GenData\GenStocksAll.R`.\
    This script downloads actual price data for stocks specified in `Data\StockNames.RDS`.
    It can be run in two modes:

    -   Thorough (obtained by commenting line 16 and uncommenting line 17 and the section `DatasetsGeneration`):\
        Downloads data for all assets and then chooses out of these 900 assets (450 ETFs and 450 stocks) with similar trading activity and volatility as the 100 assets in the M6 universe.
        These 900 assets are then grouped into 9 auxiliary M6-like universes, each consisting of 50 ETFs and 50 stocks.

    -   Quick (current state of the file):\
        Only downloads data for the M6 universe and the additional 9 auxiliary M6 universes.

    Thorough mode needs not be run on every submission, only when one wishes to regenerate auxiliary datasets (e.g., if too many stocks cease to exist in existing universes).

3.  Run `R\QuantilePrediction\QuantilePrediction.R`.\
    This script trains the MtMs model and outputs the predictions to the `Precomputed\QuantilePredictions.RDS`.
    To run it, it is first necessary to set `tempFilePath` to some existing location (I need it to be outside the directory because of performance issues stemming from automatic backups to the cloud, hence the absolute path.).
    Then specify the parameter `Submission` to the number corresponding to the submission one wants to generate and set the parameter `GenerateStockAggr` to `TRUE`.
    The script then performs the following:

    -   It generates tensors of features as specified in `featureList` for all assets.
        In order to generate a diverse set of features without the need to manually design them, I utilized the [TTR](https://cran.r-project.org/web/packages/TTR/index.html) package, which provides a unified interface to compute a plethora of different trading indicators. 
        In addition to seven lags of returns and volatility, and an indicator whether the given asset is an ETF, 38 stationarized trading indicators, are used as inputs (as of submission 12).
        All are generated at a 4-week frequency as specified in the competition but also additionally with 1, 2, and 3-week offsets. This is done to augment the dataset and consequently stabilize the learning process.
        Missing features are then imputed and standardized.
        Tensors are then split into training, test, and validation segments.
        By default, the validation segment corresponds to the 4-week interval for which one wants to make the submission, and therefore no performance measure can be obtained at the time of submission, merely the corresponding predictions.
        This can be changed by setting `q > 0`, which will shift the validation segment earlier by `q` 4-week intervals.

    -   An NN consisting of three layers with 32, 8, and 5 neurons, leaky ReLU nonlinearity, and dropout of 0.2 is trained via Adam under the RPS loss function with an early stopping and the patience 5.

    -   This trained model is then used as a starting point for MtMs meta-learning/multi-task-learning (see [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4355794) for more details).
        Two (until submission 8) or one (from submission 9 onwards) mesa parameter(s) are allowed per stock.
        These affects the whole base model (until submission 7), or only the weights and biases of the very last layer (from submission 8 onwards).
        The model is trained with a minibatch sampler such that each minibatch consists of 100 randomly drawn assets to mimic the setup of M6.
        The model is then repeatedly trained with progressively decreasing learning rates.
        This is because MtMs models are generally hard to train, and the optimizer frequently fails to adjust the model weights for better test loss on the first trial.
        Multiple attempts are then needed.
        This was an issue especially in the later submissions, where the optimizer had a hard time meaningfully altering an already quite good base model (either due to changes in the training/test data, improvements to the base model, or alterations of the training algorithm itself, see earlier commits).
        In early submissions, the MtMs delivered more tangible improvements.

    -   Predictions for the last month are saved to the file `Precomputed\QuantilePredictions.RDS`.

4.  Run either `R\Submission\Submission_EqualPosition.R` or `R\Submission\Submission_ManualPosition.R`.\
    These scripts load quintile predictions from the previous step, add investment decisions, and create the final submission CSV in the folder `Results`.

    -   Script `R\Submission\Submission_EqualPosition.R` was used in earlier submissions (1-4).
        It merely loads the predictions and appends an investment weight of 0.0025 to each stock.
        See [here, p. 3](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4527154) as to why equal weights scaled to 0.0025 are preferable to those scaled to 0.01.

    -   Starting from submission 5, it became apparent that a decent performance in the forecasting challenge combined with the scaling trick alone would be insufficient to secure a sufficiently good rank in the global leaderboard, and that some risk indeed needs to be taken.
        Script `R\Submission\Submission_ManualPosition.R` loads the predictions and allows the user to flip signs for positions for ($n_{-}$) of assets (i.e., -0.0025 instead of 0.0025) to induce some negative correlation between own IR and IRs of a majority of competitors who rationally opt for predominantly long portfolios.
        This negative correlation, in turn, allows one to recover from positions within the public leaderboard which would be otherwise unsalvageable (see the [rank optimization WP](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4527154)).
        During submissions 5-7, signs of positions of $n_{-}=10$ assets were flipped.
        The decision regarding which assets these should be turned out to be mostly inconsequential.
        If anything, my attempts to select stocks based on a very simple criterion derived from the quintile predictions turned out to do more harm than good (see also comments below).
        In submission 8, I have set $n_{-}=100$ to reflect the approaching end of the competition and still not sufficiently good ranking in the global leaderboard.
        This submission alone was predominantly responsible for the relatively good position within the global IR ranking.
        In submissions 9-12, I have set $n_{-}=0$ to minimize the chance of losing an already sufficiently good ranking in the global leaderboard.

Comments:

-   During the competition, the DRE stock has been acquired by PLD, which resulted in it exhibiting zero price changes from that point forward.
    To account for this, `R\QuantilePrediction\QuantilePrediction_ZeroReturnCorrection.R` computes the frequency with which a hypothetical stock with zero return would fall into individual quintiles.
    This distribution then overrides the quintile predictions for DRE in `R\Submission\Submission_ManualPosition.R` (submissions 10-12).

-   In addition to the scripts mentioned above, there are many helper/auxiliary scripts and functions that are sourced, and their purposes should be apparent from their names. Most notably, these are located in:

    -   `R/QuantilePrediction/QuantilePrediction_Helpers.R`

    -   `R/QuantilePrediction/QuantilePrediction_Features.R`

    -   `R/QuantilePrediction/QuantilePrediction_Models.R`

    -   `R/MetaModel/MetaModel.R`

    Secondly, there are some scripts that are not part of the main pipeline but have influenced its development, such as:

    -   `R\QuantilePrediction\dev_QuantilePrediction_FeaturesTesting.R`\
        In this script, I manually go over the technical training rules to decide how to best stationarize them.

    -   `R\QuantilePrediction\QuantilePrediction_XGBoost.R`
        In this script, I employ XGBoost to gauge the relative performance of MtMs and to prune features that seem redundant.

    Lastly, the remainder of the codebase consists of various testing scripts for ideas that turned out to be dead ends. Most notably:

    -   Anything located in either `R\PortfolioOptimization` or `R\PortfolioOptimizationNN` where I tried to leverage existing quintile predictions to construct portfolios.
        However, the quintile predictions contain only very limited directional information and predominantly model volatility, as one can easily verify by inspecting the final submissions.
        Consequently, none of my attempts to systematically utilize them for portfolio formation passed back-testing.

    -   Anything located in the `R\Surface` folder where I tested alternative models for quintile predictions.

    -   Anything located in the `R\Development` and `R\MetaModel\Old` folders where I stored some drafts.

-   Unfortunately, I failed to properly commit the codebase after each submission, as at the time, I did not expect that I would be publishing it.
    Hopefully, the commits I did make will make it possible to at least partially reconstruct how the model evolved.

-   When computing the 9th submission, I have accidentally overwritten the 8th submission file (`Submission_2022-09-19 - 2022-10-16.csv`), so the version you can see in the current commit does not actually correspond to what I submitted.
    To rectify this and facilitate easy viewing of all submissions at a glance, I added another folder `Results_corrected` which contains my submissions downloaded back from my M6 competition account.
    These are identical to those in the `Results` folder, except for the aforementioned erroneous submission.


