# 106 Project Ver 2

This is Version 2 of my individual project to test the reliability of the proposed EZ Diffusion Model. Being transparent, much of the code is written by Chat GPT for structure, debugging, and variables. 

My interpretation of the EZ diffusion model is that it serves as a model of human decision making. Used in Cognitive Sciences, the model is implemented when data is collected about how subjects respond to stimuli and how quickly decisions are made. Key parameters for this model include: the boundary separation a, which represents the amount of information a person needs before they can confidently make a decision. Higher boundaries means the person is more cautious and lower boundaries could mean less accuracy. The Drift Rate v is the rate at which someone gathers evidence that is used for making a decision, with higher drift rates meaning fast and efficient processing. Lastly, Non-decision Time t, represents the time spent on processes that are unrelated to the decision. 

The simulate and recover portion of this project is to put the EZ diffusion model to the test, confirming it is able to estimate parameters from simulated data. 3 phases ensure reliability: Simulation Phase, Recovery Phase, and Evaluation Phase. The simulation phase uses the parameters we give it to generate values that act as a dataset. The recovery phase tests the EZ diffusion model to generate parameters from the simulated data, which we then compare to the parameters that we gave the simulation test in the first place. Then the evaluation phase determines if the model worked well based on bias and squared error. 

With my project I hoped to see my recovered values being very close to the true values for larger sample sizes, which makes sense to me because there should be far less variability in the data. Bias should average close to zero and squared error should decrease as the sample size increases, which again is exemplified by the fact that more data means better parameter recovery. 

My project was able to gather parameters and get a good estimate of the originals, there were many errors along the way but overall I believe that the EZ diffusion model can decently predict parameter estimates based on simulated data. 

