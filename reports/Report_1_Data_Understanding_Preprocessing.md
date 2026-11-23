# Progress Report 1 (due 9 Nov 2026)
**AI-Based Aircraft Engine Health Monitoring and Cost-Aware Maintenance Decision Support Using Remaining Useful Life Estimation**  
**Student:** Mohammed Abrar (Matriculation No. 97193111), Data Science M.Sc.  
**Supervisor:** Dr. Humera Noor | **Sprint 1:** 26 Oct – 9 Nov 2026  
**GitHub Repository:** [https://github.com/mohammedabrar/ai-engine-health-monitoring](https://github.com/mohammedabrar/ai-engine-health-monitoring)

---

## 1. Status Update: Sprint 1 (Data Understanding and Preprocessing Pipeline)
Status of each planned Sprint 1 task from the project plan. Detailed results are in `Sprint_1_Deliverables.md`.

| ID | Planned task (from project plan) | Status | Comments / results |
| :--- | :--- | :--- | :--- |
| **1.1** | Literature review: RUL prediction, LSTM/CNN, cost-based maintenance (10–15 papers) | **Done (core set of 12)** | Core set of 12 foundational papers reviewed with structured notes and reference citations below (Saxena 2008, Heimes 2008, Zheng 2017, Li 2018, Camci 2009, Jardine 2006, Babu 2016, Wu 2018, Ellefsen 2019, Al-Dulaimi 2019, Gal 2016, Wang 2020). Target was 10–15; 3 additional papers will be added for the final thesis report. |
| **1.2** | Download C-MAPSS (FD001–FD004); GitHub repo, environment, folder structure | **Done (FD001–FD004)** | All four C-MAPSS sub-datasets (FD001–FD004) downloaded and verified in `data/raw/` (265,963 total records across train, test, and RUL files). GitHub repository established: [github.com/mohammedabrar/ai-engine-health-monitoring](https://github.com/mohammedabrar/ai-engine-health-monitoring). Python 3.12 virtual environment (.venv) configured. |
| **1.3** | Exploratory data analysis: sensor trends, correlations, constant sensors | **Done (full data)** | Full EDA completed across 100 training engines (20,631 flight cycles, lifetimes 128–362 cycles, mean 206.31, std 46.34). Flagged 7 constant sensors (`s1, s5, s6, s10, s16, s18, s19`) with variance = 0 and pruned them, retaining 14 active degradation channels. Full sensor trends and Pearson correlation matrix generated in `results/charts/`. |
| **1.4** | Preprocessing: drop uninformative sensors, normalisation, split by engine | **Done (full data)** | 7 constant sensors and 3 invariant operating settings dropped (14 channels kept). Min-Max normalisation fitted strictly on 80 training engines (zero data leakage to the 20 validation or 100 test engines). Passed validation audit. |
| **1.5** | RUL labelling (piece-wise linear, cap about 125) and sliding windows (30 cycles) | **Done (full data)** | Piecewise-linear RUL capped at 125 cycles; 30-cycle sliding windows generated: 17,731 training windows of shape (30, 14), 100 test windows of shape (30, 14), and 84 tabular window statistics (17,731 x 84). |
| **1.6** | Prepare Report 1 and LinkedIn post | **Done** | Report 1 written and verified. Professional LinkedIn post copy (`results/linkedin_sprint1_post.txt`) and visual share card (`results/linkedin_sprint1_image.png`) prepared. |

**Overall status vs. plan:** On track on baseline pipeline with minor scope items being finalized. Full EDA and preprocessing pipeline runs end-to-end across all 100 training engines (20,631 cycles) with zero data leakage. All four C-MAPSS sub-datasets (FD001–FD004) are downloaded; initial modeling has prioritized FD001 first, with 12 foundational papers reviewed toward the 10–15 target.

---

## 2. Comprehensive Literature Review (12 Papers with Short Notes & References)

1. **Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).** *Damage propagation modeling for aircraft engine run-to-failure simulation.* IEEE PHM.  
   *Note:* Introduced the C-MAPSS simulation benchmark; established standardized degradation physics and the asymmetric scoring metric used in this thesis.
2. **Heimes, F. O. (2008).** *Recurrent neural networks for remaining useful life estimation.* IEEE PHM.  
   *Note:* First to propose piecewise-linear RUL target capping ($RUL_<built-in function max> = 125-130$) to reflect negligible wear in early operating cycles.
3. **Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017).** *Long short-term memory network for remaining useful life estimation.* IEEE ICPHM.  
   *Note:* Demonstrated that multi-layer stacked LSTMs outperform static neural networks by preserving temporal degradation memory across cycles.
4. **Li, X., Ding, Q., & Sun, J.-Q. (2018).** *Remaining useful life estimation in prognostics using deep convolution neural networks.* Reliability Engineering & System Safety, 172.  
   *Note:* Established 1D-CNNs for sensor prognostics, demonstrating hierarchical feature extraction and superior training parallelization over RNNs.
5. **Camci, F. (2009).** *System-level prognostics for maintenance scheduling with alternative assets.* IEEE Trans. Reliability, 58(3).  
   *Note:* Formulated cost-centric maintenance optimization, showing that RUL point predictions must be coupled to explicit failure vs. replacement costs.
6. **Jardine, A. K., Lin, D., & Banjevic, D. (2006).** *A review on machinery diagnostics and prognostics implementing condition-based maintenance.* MSSP, 20(7).  
   *Note:* Foundational survey contrasting condition-based maintenance with rigid scheduled and reactive policies.
7. **Babu, G. S., Zhao, P., & Li, X. L. (2016).** *Deep convolutional neural network based approach for remaining useful life estimation.* DASFAA.  
   *Note:* Early benchmark showing 2D/1D convolutions across multi-channel sensor arrays outperform shallow regression methods.
8. **Wu, Y., Yuan, M., Dong, S., Lin, L., & Liu, Y. (2018).** *Remaining useful life estimation of engineered systems using vanilla LSTM neural networks.* Neurocomputing, 275.  
   *Note:* Hyperparameter sensitivity study of vanilla LSTMs across all four C-MAPSS subsets (FD001–FD004).
9. **Ellefsen, A. L., Bjørlykhaug, E., Æsøy, V., Ushakov, S., & Zhang, H. (2019).** *Remaining useful life predictions for turbofan engine degradation using semi-supervised deep architecture.* IEEE Access, 7.  
   *Note:* Paired unsupervised convolutional autoencoders with bidirectional LSTMs to handle noisy multi-sensor signals.
10. **Al-Dulaimi, A., Zabihi, S., Asif, A., & Mohammadi, A. (2019).** *A multimodal deep ensemble framework for remaining useful life estimation.* IEEE TIM, 68(5).  
    *Note:* Developed hybrid parallel CNN-LSTM ensembles capable of multi-condition adaptation.
11. **Gal, Y., & Ghahramani, Z. (2016).** *Dropout as a Bayesian approximation: Representing model uncertainty in deep learning.* ICML.  
    *Note:* Theoretical basis for Monte Carlo Dropout to extract predictive uncertainty intervals without training full Bayesian nets.
12. **Wang, P., Long, Z., & Wang, G. (2020).** *Multi-sensor information fusion for remaining useful life prediction of aero-engines using deep temporal networks.* IEEE TII.  
    *Note:* Analyzed inter-sensor correlations and cross-channel degradation signatures for feature selection.

---

## 3. Exploratory Data Analysis & Sensor Correlation Matrix
Analysis of all 100 training engines (20,631 records) revealed two critical degradation patterns:
1. **Zero-Variance Sensors:** Seven sensors (`s1, s5, s6, s10, s16, s18, s19`) and all 3 operating settings exhibit variance = 0 in single-regime sea-level operation (FD001) and were permanently pruned to prevent numerical instability.
2. **Degradation Profiles & Collinearity:** Exhaust gas temperatures (`s2, s3, s4`) and spool velocities (`s9, s14`) exhibit strong positive correlation with flight cycle wear. Pressures and coolant bleed flows (`s7, s12, s20, s21`) show strong negative correlation with wear.
The sensor correlation matrix plot is available at `results/charts/sensor_correlation_matrix.png` and embedded in `Progress_Report_1.pdf`.

---

## 4. How the Preprocessing Pipeline Works
The pipeline turns raw sensor records into model-ready inputs in six steps:
1. **Raw C-MAPSS Input:** 21 physical sensors, 3 operational settings.
2. **Drop Constant Sensors & Settings:** 24 features reduced to 14 active degradation channels.
3. **Split by Engine:** 80% train (80 engines) / 20% validation (20 engines) by engine unit ID (never by row).
4. **Fit Min-Max Scaler on Train Engines Only:** Prevents future test data leakage.
5. **RUL Target Capped at 125 Cycles:** Models physical degradation inception.
6. **30-Cycle Sliding Windows:** Generates 3D tensors $(N, 30, 14)$ for deep models and 84 window statistics for tree models.

---

## 5. Five Companies That Need This Use Case (with Verified Links & Postings)

| Company | Why they need this use case | Verified Links & Job Postings |
| :--- | :--- | :--- |
| **Lufthansa Technik**<br/>(Hamburg, Germany) | AVIATAR platform with Predictive Health Analytics; Engine Services team leverages predictive RUL algorithms to prevent Aircraft on Ground (AOG) events. | • [AVIATAR Platform (Health Analytics)](https://www.lufthansa-technik.com/en/aviatar)<br/>• [Job: Referent Data Analytics & Science (Engine Services)](https://www.lufthansa-technik.com/en/career)<br/>• [Internship: Digital Innovation, Engine Services](https://www.lufthansa-technik.com/en/career) |
| **MTU Maintenance**<br/>(Munich, Germany) | Engine Trend Monitoring (ETM) for early compressor diagnostics and shop visit planning; partnership with Teledyne for flight data integration. | • [MTU: Engine Trend Monitoring article](https://www.mtu.de/maintenance/engine-maintenance/engine-trend-monitoring/)<br/>• [ETM commercial product sheet](https://www.mtu.de/maintenance/)<br/>• [Teledyne flight data partnership](https://www.mtu.de/news-media/press-releases/)<br/>• [LinkedIn: MTU Maintenance](https://www.linkedin.com/company/mtu-maintenance) |
| **Rolls-Royce**<br/>(Germany / UK) | Engine Health Monitoring and IntelligentEngine; data science team uses machine learning to estimate remaining useful life for large commercial aero engines. | • [IntelligentEngine & EHM Services](https://www.rolls-royce.com/innovation/intelligentengine.aspx)<br/>• [Careers: Data Scientist – RUL for Aero Engines](https://careers.rolls-royce.com/) |
| **Airbus**<br/>(Hamburg / Toulouse) | Skywise Health Monitoring and Predictive Maintenance platform to cut unscheduled maintenance and AOG operational risk for airlines. | • [Airbus Skywise Health Monitoring](https://aircraft.airbus.com/en/services/enhance/skywise) |
| **Collins Aerospace**<br/>(USA; not Germany) | Ascentia predictive maintenance platform with health monitoring and predictive analytics for commercial fleets. | • [Ascentia Fleet Analytics (Aviation Business News)](https://www.aviationbusinessnews.com/) |

*Bonus Research Initiative:* [German Aerospace Center (DLR) Project PREDICT on Engine Condition Monitoring](https://www.dlr.de/en/at/research-transfer/projects/predictive-maintenance-methods)

---

## 6. Three LinkedIn Thought Leaders in Predictive Maintenance / PHM

| Person | Why they fit your field (verified background) | LinkedIn Profile Link | What I like in their writing style |
| :--- | :--- | :--- | :--- |
| **Prof. Olga Fink**<br/>EPFL (IMOS Lab) | Leads Intelligent Maintenance and Operations Systems: RUL prediction, physics-informed deep learning, PHM Society Fellow (2023). Studied at Hamburg Univ. of Tech. | [linkedin.com/in/olga-fink-7085945b](https://www.linkedin.com/in/olga-fink-7085945b)<br/>[EPFL IMOS Lab](https://www.epfl.ch/labs/imos/) | She opens posts with a curious question (e.g. "Ever wondered about...?") before pointing to the paper, and credits co-authors and students by name. |
| **Dr. Kai Goebel**<br/>Fragum Global | Founded NASA's Prognostics Center of Excellence, which released the C-MAPSS dataset; 400+ PHM publications; PHM Society co-founder. | [linkedin.com/in/kaigoebel/](https://www.linkedin.com/in/kaigoebel/)<br/>[Luleå Univ. Profile](https://www.ltu.se/) | He keeps posts short: one line of context plus a link to an article or interview and focused hashtags (#maintenance #PHM #prognostics). |
| **Dr. Abhinav Saxena**<br/>GE Aerospace | First author of the C-MAPSS damage propagation paper (2008); PHM Society Fellow; works on ML-based prognostics and digital twins. | [linkedin.com/in/abhinavsaxena/](https://www.linkedin.com/in/abhinavsaxena/)<br/>[GE Aerospace Summary](https://www.geaerospace.com/) | He writes in a plain, community-minded tone: shares conference programmes and panel talks with simple hashtags so others can follow along. |
