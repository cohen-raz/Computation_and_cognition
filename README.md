# Computation_and_cognition
### Department of Cognitive and Brain Sciences at The Hebrew University of Jerusalem
This project main goal is to **implement** 2 simple reinforcement learning models, and **fit** each model to a mice behaviour data set (time series data), in order to determine which model can predict and represent in a better way, the mice behaviour.

**Data Set:**  
- The data contains several mice which trained to discriminate among pure tones (auditory task).
- For more inforamtion about the original reaserch and the data set, check the original paper: [Neural Correlates of Learning Pure Tones or Natural Sounds in the Auditory Cortex](https://www.frontiersin.org/articles/10.3389/fncir.2019.00082/full)  

**Modeling:**
- The auditory stimulus in each trial represent the mice *states*: 'Go'/ 'NoGo'. Where 'Go' means that the mice should lick the drinking device, and 'NoGO' means they shouldn'tdon't lick. (state in trail t ~ Ber(0.5))
 - Each mice can perform 2 actions: Lick / not lick
- For each action in each state, there is a different reward:  
![image](https://user-images.githubusercontent.com/83977654/128196323-d6cf5f38-1061-4cfb-a15c-28b5f3aac5fc.png)
  
  
**RL models:**
1) Reinforce model:  

2) TD-learning model:  



