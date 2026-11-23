# Progress Report 1 (due 9 Nov 2026)
**AI-Based Aircraft Engine Health Monitoring and Cost-Aware Maintenance Decision Support Using Remaining Useful Life Estimation**  
**Student:** Mohammed Abrar (Matriculation No. 97193111), Data Science M.Sc.  
**Supervisor:** Dr. Humera Noor | **Sprint 1:** 26 Oct – 9 Nov 2026  
**GitHub Repository:** [https://github.com/mohammedabrarafnan1234-tech/AI-Based-Aircraft-Engine-Health-Monitoring-](https://github.com/mohammedabrarafnan1234-tech/AI-Based-Aircraft-Engine-Health-Monitoring-)

---

## 1. Status Update: Sprint 1 (Data Understanding and Preprocessing Pipeline)
Status of each planned Sprint 1 task from the project plan. Detailed results are in `Sprint_1_Deliverables.md`.

| ID | Planned task (from project plan) | Status | Comments / results |
| :--- | :--- | :--- | :--- |
| **1.1** | Literature review: RUL prediction, LSTM/CNN, cost-based maintenance (10–15 papers) | **Done (core set of 12)** | Core set of 12 verified papers reviewed with structured notes and reference citations below (Saxena 2008, Heimes 2008, Zheng 2017, Li 2018, Camci 2009, Jardine 2006, Babu 2016, Wu 2018, Ellefsen 2019, Al-Dulaimi 2019, Gal 2016, Wu 2020). Target was 10–15; 3 additional papers will be added for the final thesis report. |
| **1.2** | Download C-MAPSS (FD001–FD004); GitHub repo, environment, folder structure | **Done (FD001–FD004)** | All four C-MAPSS sub-datasets (FD001–FD004) downloaded and verified in `data/raw/` (265,963 total records across train, test, and RUL files). GitHub repository established: [github.com/mohammedabrarafnan1234-tech/AI-Based-Aircraft-Engine-Health-Monitoring-](https://github.com/mohammedabrarafnan1234-tech/AI-Based-Aircraft-Engine-Health-Monitoring-). Python 3.12 virtual environment (.venv) configured. |
| **1.3** | Exploratory data analysis: sensor trends, correlations, constant sensors | **Done (full data)** | Full EDA completed across 100 training engines (20,631 flight cycles, lifetimes 128–362 cycles, mean 206.31, std 46.34). Flagged 7 constant or near-constant sensors (`s1, s5, s6, s10, s16, s18, s19`) with near-zero variance and pruned them, retaining 14 active degradation channels. Full sensor trends and Pearson correlation matrix generated in `results/charts/`. |
| **1.4** | Preprocessing: drop uninformative sensors, normalisation, split by engine | **Done (full data)** | 7 constant or near-constant sensors and 3 invariant operating settings dropped (14 channels kept). Min-Max normalisation fitted strictly on 80 training engines (zero data leakage to the 20 validation or 100 test engines). Passed validation audit. |
| **1.5** | RUL labelling (piece-wise linear, cap about 125) and sliding windows (30 cycles) | **Done (full data)** | Piecewise-linear RUL capped at 125 cycles; 30-cycle sliding windows generated: 17,731 sequences total across all 100 training engines (13,818 train windows on 80 engines, 3,913 validation windows on 20 engines), 100 test sequences, and 84 tabular window statistics. |
| **1.6** | Prepare Report 1 and LinkedIn post | **Done** | Report 1 written and verified. Professional LinkedIn post copy (`results/linkedin_sprint1_post.txt`) and visual share card (`results/linkedin_sprint1_image.png`) prepared. |

**Overall status vs. plan:** On track on baseline pipeline with minor scope items being finalized. Full EDA and preprocessing pipeline runs end-to-end across all 100 training engines (20,631 cycles) with zero data leakage. All four C-MAPSS sub-datasets (FD001–FD004) are downloaded; initial modeling has prioritized FD001 first, with 12 foundational papers reviewed toward the 10–15 target.

---

## 2. Comprehensive Literature Review (12 Verified Papers with Notes & References)

1. **Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).** *Damage propagation modeling for aircraft engine run-to-failure simulation.* In 2008 International Conference on Prognostics and Health Management (pp. 1–9). IEEE.  
   *Note:* Introduced the C-MAPSS simulation benchmark; established standardized degradation physics and the asymmetric scoring metric used in this thesis.
2. **Heimes, F. O. (2008).** *Recurrent neural networks for remaining useful life estimation.* In 2008 International Conference on Prognostics and Health Management (pp. 1–6). IEEE.  
   *Note:* Used a piece-wise linear RUL target ($RUL_<built-in function max> = 125–130$) to reflect negligible wear in early operating cycles.
3. **Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017).** *Long short-term memory network for remaining useful life estimation.* In 2017 IEEE International Conference on Prognostics and Health Management (ICPHM) (pp. 88–95). IEEE.  
   *Note:* Demonstrated that multi-layer stacked LSTMs preserve temporal degradation memory across operational cycles, outperforming feedforward networks.
4. **Li, X., Ding, Q., & Sun, J.-Q. (2018).** *Remaining useful life estimation in prognostics using deep convolution neural networks.* Reliability Engineering & System Safety, 172, 1–11.  
   *Note:* Established 1D-CNNs for sensor prognostics, demonstrating hierarchical feature extraction and superior training parallelization over recurrent nets.
5. **Camci, F. (2009).** *System maintenance scheduling with prognostics information using genetic algorithm.* IEEE Transactions on Reliability, 58(3), 539–552.  
   *Note:* Used prognostics information to schedule maintenance with a genetic algorithm, balancing operational costs against failure risks.
6. **Jardine, A. K., Lin, D., & Banjevic, D. (2006).** *A review on machinery diagnostics and prognostics implementing condition-based maintenance.* Mechanical Systems and Signal Processing, 20(7), 1483–1510.  
   *Note:* Foundational review contrasting condition-based maintenance (CBM) with rigid scheduled and reactive policies in rotating machinery.
7. **Babu, G. S., Zhao, P., & Li, X.-L. (2016).** *Deep convolutional neural network based regression approach for estimation of remaining useful life.* In Database Systems for Advanced Applications (DASFAA 2016) (pp. 214–228). Springer.  
   *Note:* Early benchmark showing 2D/1D convolutions across multi-channel sensor arrays outperform shallow regression methods.
8. **Wu, Y., Yuan, M., Dong, S., Lin, L., & Liu, Y. (2018).** *Remaining useful life estimation of engineered systems using vanilla LSTM neural networks.* Neurocomputing, 275, 167–179.  
   *Note:* Hyperparameter sensitivity study of vanilla LSTMs across all four C-MAPSS subsets (FD001–FD004).
9. **Ellefsen, A. L., Bjørlykhaug, E., Æsøy, V., Ushakov, S., & Zhang, H. (2019).** *Remaining useful life predictions for turbofan engine degradation using semi-supervised deep architecture.* Reliability Engineering & System Safety, 183, 240–251.  
   *Note:* Paired unsupervised pre-training with a genetic algorithm for hyperparameter tuning to handle noisy multi-sensor signals.
10. **Al-Dulaimi, A., Zabihi, S., Asif, A., & Mohammadi, A. (2019).** *A multimodal and hybrid deep neural network model for Remaining Useful Life estimation.* Computers in Industry, 108, 186–196.  
    *Note:* Developed a multimodal and hybrid deep neural network model combining convolutional and recurrent pathways for RUL estimation.
11. **Gal, Y., & Ghahramani, Z. (2016).** *Dropout as a Bayesian approximation: Representing model uncertainty in deep learning.* In International Conference on Machine Learning (ICML) (pp. 1050–1059). PMLR.  
    *Note:* Theoretical foundation for Monte Carlo Dropout to extract predictive uncertainty intervals without training full Bayesian nets.
12. **Wu, J., Hu, K., Cheng, Y., Zhu, H., Shao, X., & Wang, Y. (2020).** *Data-driven remaining useful life prediction via multiple sensor signals and deep long short-term memory neural network.* ISA Transactions, 97, 241–250.  
    *Note:* Fuses multiple sensor signals with deep LSTM networks to capture multivariate degradation trends across turbofan components.

---

## 3. Exploratory Data Analysis & Sensor Correlation Matrix
Analysis of all 100 training engines (20,631 records) revealed two critical degradation patterns:
1. **Constant or Near-Constant Sensors:** Seven sensors (`s1, s5, s6, s10, s16, s18, s19`) and all 3 operating settings exhibit near-zero variance in single-regime sea-level operation (FD001) and were permanently pruned to prevent numerical instability.
2. **Degradation Profiles & Collinearity:** Exhaust gas temperatures (`s2, s3, s4`) and spool velocities (`s9, s14`) exhibit strong positive correlation with flight cycle wear. Pressures and coolant bleed flows (`s7, s12, s20, s21`) show strong negative correlation with wear.
The sensor correlation matrix plot is available at `results/charts/sensor_correlation_matrix.png` and embedded in `Progress_Report_1.pdf`.

---

## 4. How the Preprocessing Pipeline Works
The pipeline turns raw sensor records into model-ready inputs in six steps:
1. **Raw C-MAPSS Input:** 21 physical sensors, 3 operational settings.
2. **Drop Constant Sensors & Settings:** 24 features reduced to 14 active degradation channels.
3. **Split by Engine:** 80% train (80 engines) / 20% validation (20 engines) by engine unit ID (never by row).
4. **Fit Min-Max Scaler on Train Engines Only:** Learned strictly from the 80 training engines (13,818 windows) to prevent data leakage.
5. **RUL Target Capped at 125 Cycles:** Models physical degradation inception.
6. **30-Cycle Sliding Windows:** Generates 17,731 total sequences (13,818 train / 3,913 val) of shape $(30, 14)$ for deep models, and 84 window statistics for tree models.

---

## 5. Five Companies That Need This Use Case (with Verified Specific Links)

| Company | Why they need this use case | Verified Specific Links & Postings |
| :--- | :--- | :--- |
| **Lufthansa Technik**<br/>(Hamburg, Germany) | AVIATAR platform with Predictive Health Analytics; Engine Services team leverages predictive RUL algorithms to prevent Aircraft on Ground (AOG) events. | • [AVIATAR Health Monitoring Expansion](https://www.lufthansa-technik.com/en/aviatar-vastly-expands-its-aircraft-health-monitoring-portfolio-for-boeing-737s-78d7ef6c3a5954cc)<br/>• [Job: Referent Data Analytics & Science (Engine Services)](https://innovation-runway.lufthansagroup.com/en/career/job-offers/offer/referent-data-analytics-science-m-w-divers-127854.html) |
| **MTU Maintenance**<br/>(Munich, Germany) | Engine Trend Monitoring (ETM) for early compressor diagnostics and shop visit planning; partnership with Teledyne for flight data integration. | • [AeroReport: Engine Trend Monitoring Article](https://aeroreport.de/en/innovation/better-maintenance-planning-with-engine-trend-monitoring)<br/>• [MTU & Teledyne Partnership (BusinessWire)](https://www.businesswire.com/news/home/20251028303766/en/mtu-maintenance-and-teledyne-controls-partner-to-deliver-enhanced-engine-health-monitoring/) |
| **Rolls-Royce**<br/>(Germany / UK) | Engine Health Monitoring and IntelligentEngine; data science team uses machine learning to estimate remaining useful life for large commercial aero engines. | • [IntelligentEngine Vision Progress Press Release](https://www.rolls-royce.com/media/press-releases/2018/16-07-2018-rr-intelligentengine-vision-makes-rapid-progress.aspx) |
| **Airbus**<br/>(Hamburg / Toulouse) | Skywise Health Monitoring and Predictive Maintenance platform to cut unscheduled maintenance and AOG operational risk for airlines. | • [Airbus Skywise Health Monitoring (Frontier Airlines)](https://www.airbus.com/en/newsroom/press-releases/2021-10-frontier-airlines-selects-skywise-health-monitoring-as-future-fleet) |
| **Collins Aerospace**<br/>(USA; not Germany) | Ascentia predictive maintenance platform with health monitoring and predictive analytics for commercial fleets. | • [Turning Data into Intelligence: Predictive Maintenance](https://www.aviationbusinessnews.com/mro/latest-news-mro/turning-data-into-intelligence-predictive-maintenance) |

*Bonus Research Initiative:* [German Aerospace Center (DLR) Project PREDICT on Engine Condition Monitoring](https://www.dlr.de/en/at/research-transfer/projects/predictive-maintenance-methods)

---

## 6. Three LinkedIn Thought Leaders in Predictive Maintenance / PHM

| Person | Why they fit your field (verified background) | Verified Profile Links | What I like in their writing style |
| :--- | :--- | :--- | :--- |
| **Prof. Olga Fink**<br/>EPFL (IMOS Lab) | Leads Intelligent Maintenance and Operations Systems: RUL prediction, physics-informed deep learning, PHM Society Fellow (2023). Studied at Hamburg Univ. of Tech. | [linkedin.com/in/olga-fink-7085945b](https://www.linkedin.com/in/olga-fink-7085945b)<br/>[EPFL IMOS Lab](https://www.epfl.ch/labs/imos/) | She opens posts with a curious question (e.g. "Ever wondered about...?") before pointing to the paper, and credits co-authors and students by name. |
| **Dr. Kai Goebel**<br/>Fragum Global | Founded NASA's Prognostics Center of Excellence, which released the C-MAPSS dataset; 400+ PHM publications; PHM Society co-founder. | [linkedin.com/in/kaigoebel/](https://www.linkedin.com/in/kaigoebel/)<br/>[LTU Faculty Profile](https://www.ltu.se/en/staff/k/kai-goebel) | He keeps posts short: one line of context plus a link to an article or interview and focused domain hashtags (#maintenance #PHM #prognostics). |
| **Dr. Abhinav Saxena**<br/>GE Aerospace | First author of the C-MAPSS damage propagation paper (2008); PHM Society Fellow; works on ML-based prognostics and digital twins. | [linkedin.com/in/abhinavsaxena/](https://www.linkedin.com/in/abhinavsaxena/)<br/>[alphaXiv Profile](https://alphaxiv.org/@abhinav-saxena) | He writes in a plain, community-minded tone: shares conference programmes and panel talks with simple hashtags so others can follow along. |
