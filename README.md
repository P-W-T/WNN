# WNN (Waterfall Neural Network)
## Intro
This repository deals with an alternative model architecture (currently tested on regression problems). 

This architecture is based on early branching to the output. The model consists of an initial layer, WNN layers and an output layer. The initial layer calculates a single output from the original inputs. Each WNN layer calculates a state and a single output. This state by this layer to create the single scalar output and forms the input for the next layer. The single scalar outputs are used by the output layer to calculate the final output.

The idea is that by connecting the early layers to the output directly, the gradients will flow more reliably and stabilize training. 

## Files
### Python files
- benchmark_functions.py: This file contains functions to test the models. The functions are created to be nonlinear in different ways (part of the functions are inspired by Liu et al. 2020 and Tsang et al. 2017).
- interaction.py: This file contains functions to investigate and visualize the feature interactions for a basic neural net and the WNN neural net.
- model.py: The file containing the class for a basic neural net (BasicNN) and the WNN neural net (WNNBase is the basic WNN neural network).
- training.py: This file contains functions for training and evaluating models, visualizing training curves and hyperparameter tuning.

### Jupyter notebooks
- Training.ipynb: Jupyter notebook showing how to use the WNN model. This file contains a comparisson and an analysis between the WNNBase model and a basic neural net (BasicNN). 

## Evaluation
Preliminary testing has shown that this architecture is much better compared to the standard neural network in terms of both performance and training stability.
More testing is required.

## Disclaimer
The code here is in development. As such the code might contain bugs and will be updated regularly. Use at your own risk (for the moment)

## References:
- Liu, Z., Song, Q., Zhou, K., Wang, T. H., Shan, Y., & Hu, X. (2020, October). Towards interaction detection using topological analysis on neural networks. In Neural Information Processing Systems.
- Tsang, M., Cheng, D., & Liu, Y. (2017). Detecting statistical interactions from neural network weights. arXiv preprint arXiv:1705.04977.

