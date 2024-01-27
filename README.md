# WNN (Waterfall Neural Network)
## Intro
This repository deals with an alternative model architecture (currently tested on regression problems). 

This architecture is based on early branching to the output. The model consists of an initial layer, WNN layers and an output layer. The initial layer calculates a single output from the original inputs. Each WNN layer calculates a state and a single output. This state by this layer to create the single scalar output and forms the input for the next layer. The single scalar outputs are used by the output layer to calculate the final output.

The idea is that by connecting the early layers to the output directly, the gradients will flow more reliably and stabilize training. 

## Files
### Python files
- benchmark_functions.py
