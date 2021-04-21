### Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection implementation (unofficial)
Unofficial pytorch implementation of  
Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection (STPM)  
\- Guodong Wang, Shumin Han, Errui Ding, Di Huang  (2021)  
https://arxiv.org/abs/2103.04257v2  

notice : This code is not official code and not verified yet. 

### Usage 
~~~
python train.py --phase 'train or test' --dataset_path '...\mvtec_anomaly_detection\bottle' --project_path 'path\to\save\results'
~~~

### MVTecAD pixel-level AUC-ROC score (mean of n trials)
| Category | Original paper | This code |
| :-----: | :-: | :-: |
| carpet | 0.988 | 0.985(1)|
| grid | 0.99 | 0.989(1)|
| leather | 0.993 | 0.988(1)|
| tile | 0.974 | 0.937(1)|
| wood | 0.972 | 0.856|
| bottle | 0.988 | 0.968(1)|
| cable | 0.955 | 0.718(1)|
| capsule | 0.983 | 0.975(1)|
| hazelnut | 0.985 | 0.941(1)|
| metal nut | 0.976 | 0.967(1)|
| pill | 0.978 | 0.948(1)|
| screw | 0.983 | 0.984(1)|
| toothbrush | 0.989 | 0.980(1) |
| transistor | 0.825 | 0.569(1)|
| zipper | 0.985 | 0.979(1)|

Under test.    

### Localization results   


![plot](./samples/bent_003_arr.png)
![plot](./samples/bent_009_arr.png)
![plot](./samples/broken_000_arr.png)
![plot](./samples/metal_contamination_003_arr.png)
![plot](./samples/thread_001_arr.png)
![plot](./samples/thread_005_arr.png)