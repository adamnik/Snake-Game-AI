Snake Game with Reinforcement Learning (Approximate Q-Learning)

Algorithm for Q-Learning adapted from Dave Musicant Black Jack assignment.

Idea for game states adapted from "Snake Played by a Deep Reinforcement Learning Agent" by
Hennie de Harder:
https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36

To run the code from the command line:
	
	python3 snakeGame.py [Optional: -FPS=] [Optional: -iterations=]
	
You should also pip install pygame if necessary:
	
	pip install pygame
	

You may change the number of iterations the simulation runs by typing:

	-iterations=x 

Where x is the number of interations you would like. 
We typically tested with 200 iterations, this takes about 11 minutes (@25fps).


You can also change the frames per second the simulation runs at. We 
tested with a default of 25fps but if you would like to slow it down
(or speed it up) you can change it with:

	-fps=x 

where x is the fps you would like. 

