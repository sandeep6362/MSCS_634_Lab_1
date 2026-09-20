# MSCS_634_Lab_1: Data Visualization, Data Preprocessing, and Statistical Analysis

**Name:** Sandeep Reddy Banda  
**Course:** MSCS-634 – Advanced Big Data and Data Mining, University of the Cumberlands

## Purpose of This Lab

The goal of this lab was to practice the basic steps that come before any deeper data mining work: exploring data with plots, cleaning it, reducing and scaling it, and describing it with statistics. I did all of it in a Jupyter Notebook using pandas, NumPy, Matplotlib, and Seaborn.

For the dataset I built a **simulated set of 1,200 IVR (voice self-service) call sessions**, since call data has a good mix of numbers, categories, and dates. I did not use any real customer data. The columns include call duration, dialogue turns, re-prompts, speech recognition confidence, API latency, agent transfers, and a survey score. I added missing values and a few extreme values on purpose so the cleaning steps had something to do. The relationships between columns (for example, more re-prompts leading to longer calls) are ones I built into the generator script, so the results are for practice and should not be treated as real system performance.

## What's in This Repository

| File / Folder | Description |
|---|---|
| `MSCS_634_Lab_1.ipynb` | The full notebook (all four steps, with my explanations) |
| `generate_dataset.py` | Script I used to create the simulated dataset (seeded, so it is repeatable) |
| `data/ivr_call_sessions_raw.csv` | Raw dataset (1,200 rows, 12 columns, includes missing values and outliers) |
| `data/ivr_call_sessions_clean.csv` | Dataset after missing-value handling and outlier handling (1,163 rows) |
| `data/ivr_call_sessions_processed.csv` | 50% sample after column reduction, scaling, and discretization (582 rows) |
| `screenshots/` | All required screenshots, numbered in the order of the lab steps |

## Key Insights

**From the visualizations**
- Most calls last about 100 to 500 seconds, but a small group of sessions run past 1,500 seconds (14 calls were over 1,000 seconds). These looked like stuck or abandoned sessions.
- The New Platform had a lower median call duration than the Legacy IVR (about 221 vs. 276 seconds in the raw data), and its weekly median stayed lower in every week.
- Rewards Redemption was the longest call type (median about 320 seconds) and Balance Inquiry the shortest (about 148 seconds).
- More dialogue turns meant longer calls, which shows up clearly in the scatter plot.
- Make a Payment was the largest share of calls (32.8%), and payment-related intents together were about 47% of calls.

**From the statistics (cleaned data, 1,163 rows)**
- Average call duration was about 253 seconds (median 248, standard deviation about 88). Mean and median are close, so the cleaned data is fairly balanced.
- Latency still has a slight right skew (mean 226 ms vs. median 213 ms).
- Call duration correlated most with dialogue turns (0.56), agent transfers (0.46), and re-prompts (0.44).
- Survey score went down as re-prompts (−0.42) and transfers (−0.40) went up.
- `hour_of_day` had almost no correlation with anything, which supported dropping it in the data reduction step.

## Challenges and Decisions

- **Missing values:** I chose a different method for each column based on what it means. I dropped the 12 rows with a missing call duration (it is the main measure), used "Unknown" for intent, the mean for speech confidence, forward fill for latency (the rows are in time order), and the mode for the survey score. The mode fill makes the score of 4 a little more common than it really is, which is a downside I noted.
- **Outliers:** The IQR method flagged 25 calls. Fourteen were clearly extreme, but 11 were only slightly over the upper fence (about 502 to 555 seconds). I removed all 25 to stay consistent with the method, even though a few were probably normal long calls. For latency I used the standard deviation method and capped values at mean + 3 SD instead of deleting more rows.
- **Data reduction:** I used a 50% random sample for the scaling step, but I ran the statistics on the full cleaned data because more rows gives more reliable results. I dropped `call_id` (just an identifier) and `hour_of_day` (near-zero correlation).
- **Plots:** The extreme calls squashed the first histogram and scatter plot, so I added a zoomed-in histogram and limited the scatter plot's y-axis to 600 seconds so the main pattern was readable.
- **Choosing a dataset:** I was not sure at first whether to use a public dataset. I decided to simulate one so I could control which problems (missing data, outliers) were in it, and so I would not have to use any real customer data.

## How to Run

1. Install the libraries: `pip install pandas numpy matplotlib seaborn jupyter`
2. (Optional) Recreate the raw data with `python generate_dataset.py`. The CSV is already included.
3. Open `MSCS_634_Lab_1.ipynb` in Jupyter and run all cells from the top. Everything is seeded, so the outputs should match the screenshots closely (small things like the dtype labels in `.info()` can differ depending on your pandas version).
