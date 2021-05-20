### Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection implementation (unofficial)
Unofficial pytorch implementation of  
Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection (STPM)  
\- Guodong Wang, Shumin Han, Errui Ding, Di Huang  (2021)  
https://arxiv.org/abs/2103.04257v2  

notice(21/05/20) : A reason for low scores for some categories figured out. ('RGB' conversion in test phase..) All results are updated.

### Usage 
~~~
python train.py --phase 'train or test' --dataset_path '...\mvtec_anomaly_detection\bottle' --project_path 'path\to\save\results'
~~~

### MVTecAD AUC-ROC score (mean of n trials)
| Category | Paper<br>(pixel-level) | This code<br>(pixel-level) | Paper<br>(image-level) | This code<br>(image-level) |
| :-----: | :-: | :-: | :-: | :-: |
| carpet | 0.988 | 0.987(1) | - | 0.985(1) |
| grid | 0.990 | 0.987(1) | - | 0.995(1) |
| leather | 0.993 | 0.989(1) | - | 1.000(1) |
| tile | 0.974 | 0.963(1) | - | 0.938(1) |
| wood | 0.972 | 0.949(1)| - | 0.993(1) |
| bottle | 0.988 | 0.982(1)| - | 1.000(1) |
| cable | 0.955 | 0.944(1) | - | 0.891(1) |
| capsule | 0.983 | 0.981(1) | - | 0.862(1) |
| hazelnut | 0.985 | 0.981(1) | - | 1.000(1) |
| metal nut | 0.976 | 0.968(1) | - | 0.999(1) |
| pill | 0.978 | 0.973(1) | - | 0.972(1) |
| screw | 0.983 | 0.985(1) | - | 0.900(1) |
| toothbrush | 0.989 | 0.984(1) | - | 0.875(1) |
| transistor | 0.825 | 0.800(1)| - | 0.916(1) |
| zipper | 0.985 | 0.978(1) | - | 0.879(1) |
| mean | 0.970 | 0.963(1) | 0.955 | 0.947(1) |


### Localization results   
(will be updated)  

![plot](./samples/bent_003_arr.png)
![plot](./samples/bent_009_arr.png)
![plot](./samples/broken_000_arr.png)
![plot](./samples/metal_contamination_003_arr.png)
![plot](./samples/thread_001_arr.png)
![plot](./samples/thread_005_arr.png)