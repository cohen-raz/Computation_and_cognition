# Computation_and_cognition
### Department of Cognitive and Brain Sciences at The Hebrew University of Jerusalem
This project main goal is to **implement** two simple reinforcement learning models, and **fit** each model to a mice behaviour data set (time series data), in order to determine which model can predict in a better way the mice behaviour.

**Data Set:**  
- The data contains several mice which trained to discriminate among pure tones (auditory task).
- For more inforamtion about the original reaserch and the data set, check the original paper: [Maor et al.(2020). Neural correlates of learning pure tones or natural sounds in the auditory cortex.](https://www.frontiersin.org/articles/10.3389/fncir.2019.00082/full)  

**Modeling:**
- The auditory stimulus in each trial represent the mouse states: 'Go'/ 'NoGo'.  
Where 'Go' means that the required action is licking the drinking device, and 'NoGO' means that the required action is not licking.  
(The states are independent; state ~ Ber(0.5))
 - The mouse can perform two actions as response to the stimulus: Lick / not lick
- For each action in each state, there is a different reward:  
  
![image](https://user-images.githubusercontent.com/83977654/128905279-f74cd744-b963-4b08-8a32-d5e8834bb61a.png)  

  
**RL models:**
1) Reinforce model:  
![image](https://user-images.githubusercontent.com/83977654/128905104-8749e3be-33b9-41a1-8297-7598f4aac301.png)

2) TD-learning model:  



