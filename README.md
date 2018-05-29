# Instructions

Make sure to install all dependencies.

Open a command prompt in this directory and run main.py, then type one of these commands:

• find [image or directory] -> Searches and shows elephants in each image

• classify [image or directory] -> Gets distance to hyperplane for each image

• test_thresholds -> Runs object detector with various thresholds on all images, returns PR-graph

• test_parameters [img amount] [C parameter] [p parameter] -> Trains both SVM with the given parameters and runs the object detector on all test images, returns F1 measure

• show_graph -> Shows the saved PR graph from running test_thresholds
