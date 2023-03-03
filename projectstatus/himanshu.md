## Project Abstract 
- Central question: How do current self-supervised learning methods interact with the infant visual curriculum?
- Here we do...
	- Make a pipeline of pretraining models on infant egocentric videos.
	- Report the results of pretrained models on BrainScore benchmarks.
	- Experiment with different self-supervised learning methods (contrastive and generative)
	- Experiment with synaptic density and image resolution as developmentally relevant factors that affect the results.
	
- What we found (or might find)
	- Curriculum training results in learning representations that better match human behavior: more shape bias in the later layers., 
	- Curriculum training results in learning early representations that better match early visual system properties: more orientation selectivity.
	- This is more emphasized in <x> training algorithms and in <x> experimental conditions. 

- Why it matters:
	- For behavior developmental scientists: Shows whether the embodiment condition of infants can help the early develoment of the visual system. 
	- For computational modelers (AI): A proof of concept for an important case of real-life curriculum learning for further theoretical investigations.

## Sprint Goals/Outcomes

#### Sprint 1:  Mar 1 - Mar 15
---
- Goals:
	- Set up your development environment. => DONE
	- Create a pipeline that loads the infant data and computes the VideoMAE loss.
	- Validation: Benchmark the untrained model on CIFAR10. Report linear probing accuracy after 30 epochs.
	- Train the model for 1 epoch on BigRed200 multi-GPU setting.
	- Validation: Benchmark the pretrained model on CIFAR10. Report linear probing accuracy after 30 epochs.
	- 	Report runtime per epoch for training and validation. Report GPU memory usage.
	- Deliver a model checkpoint that can be passed into BrainScore.
	
- Blockers/Progress
	- 

- Outcomes
	- 



Future Goals:
- By Mar 15: Find the right set of hyperparamters for pretraining to: maximize performance, minimize runtime.
- Run the curriculum vs anti-curriculum experiemnts.
- By April 15: Submit to BrainScore (we should be ready to write the paper after this).
- If there's more time, try more algorithms.
