## Neural Network Design
### Cell states
There are 5 distinct cell states in this simulation: open, closed, bot, crew, and teleport. However, it doesn't make sense to assign each category a number and dump it
into the neural network. Instead, I will employ a one-of-k encoding layer for relevent states. This means that, for some node (corresponding to a cell) where this state 
is satisfied, the input value will be 1 and 0 elsewhere. The states that will be encoded into this layer will be the bot position,  crewmmate position, and open/closed 
cells. This will mean the input layer will be 11x11x3 nodes long. I will call each set of encodings an encoding "stack". 

### Network Architecture
Each encoding stack will be fed into its own linear layer before being fed into a larger fully-connected network. The final output will be a layer of 8 nodes, each
corresponding to an action.

#### Structure
Bot Position Encoding Stack (11x11)      Crew Position Encoding Stack (11x11)       Open/Closed Cell Encoding Stack (11x11)
             |                                              |                                          |
    Linear Layer (64)                               Linear Layer (64)                          Linear Layer (64)
                            \                               |                               /
                                                    Linear Layer (128)
                                                            |
                                                    Linear Layer (64)
                                                            |
                                                     Output Layer (9)
                                                            |
                                                   Masked Output Layer (9)




### Action Restriction
Since we know what the correct set of actions are at any given position, we should integrate this into the design of the network instead of just hoping it learns it. The
way I achieve this will be applying a mask over the final output layer, setting all invalid actions to negative infinity. Then, I softmax. 


