'''
Snake Game with Reinforcement Learning (Approximate Q-Learning)

Algorithm for Q-Learning adapted from Dave Musicant Black Jack assignment

Idea for game states adapted from "Snake Played by a Deep Reinforcement Learning Agent" by
Hennie de Harder

https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36
'''
import pygame, random, math
from pygame.locals import *
from typing import Tuple, NamedTuple
from collections import defaultdict, namedtuple
import copy
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# sets parameters for window size/display
windowHeight = 600
windowWidth = 600
cellSize = 60
rowCount = windowHeight / cellSize
columnCount = windowWidth / cellSize
clock = pygame.time.Clock()

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'
directions = [UP, DOWN, LEFT, RIGHT]

Feature = NamedTuple('Feature', [('featureKey', Tuple), ('featureValue', int)])

class Snake:

	def __init__(self, apple, iterations, fig, ax):

		# green snake
		self.color = (0,255,0)
		self.x = 0
		self.y = 0
		# snakeBody dictionary used to store information about each segment of the snake, including its x and y locations
		self.snakeBody = []
		self.reset()
		self.direction = ''
		self.score = 0
		self.iteration = 0
		self.iterationLimit = iterations
		self.model = Model()
		self.apple = apple
		self.reward = 0
		self.results = {'x':[], 'y':[], 'avg':[]}
		self.fig = fig
		self.ax = ax

	#reset the snake after each iteration of the game
	def reset(self):
		# sets initial location of snake randomly
		self.x = random.randint(3, columnCount - 2) * cellSize
		self.y = random.randint(3, rowCount - 2) * cellSize
		self.snakeBody = [{'x': self.x, 'y': self.y}]
		self.reward = 0

	def updateGraph(self, x, y):
		if len(self.results['x']) < self.iterationLimit:
			self.results['x'].append(x)
			self.results['y'].append(y)
			last10 = self.results['y'][-10:]
			self.results['avg'].append(sum(last10)/len(last10))

		def animate():
			self.ax.clear()
			self.ax.scatter(self.results['x'], self.results['y'])
			self.ax.plot(self.results['x'], self.results['avg'], label='Previous 10 Iteration Average')

		animate()
		plt.legend(loc='upper left')
		plt.xlabel('Iteration')
		plt.ylabel('Score')
		plt.draw()

	#reset
	def endGame(self):
		self.updateGraph(self.iteration, self.getScore())
		self.reset()
		self.iteration += 1

	# draws the initial snake head --> the first segment of the snake
	def drawSnake(self, win):

		# draws cube for each segment stored in snakeBody
		for i in range(len(self.snakeBody)):
			cube = pygame.Rect(self.snakeBody[i]['x'], self.snakeBody[i]['y'], cellSize, cellSize)
			pygame.draw.rect(win, self.color, cube)
			#head has eyes
			if i == 0:
				eyeHeight = cellSize / 3
				eyeWidth = eyeHeight / 2
				eye1 = pygame.Rect(self.snakeBody[i]['x'] + (cellSize/2) - eyeWidth - 5, self.snakeBody[i]['y'] + eyeHeight, eyeWidth, eyeHeight)
				eye2 = pygame.Rect(self.snakeBody[i]['x'] + (cellSize/2) + 5, self.snakeBody[i]['y'] + eyeHeight, eyeWidth, eyeHeight)
				pygame.draw.ellipse(win, (0,0,0), eye1)
				pygame.draw.ellipse(win, (0,0,0), eye2)

		pygame.display.update()

	def move(self, win, FPS):

		#randomly assigns a direction to the snake at the beginning of each game
		self.direction = random.choice(directions)
		while True:
			for event in pygame.event.get():
				if event.type == QUIT:
					pygame.quit()
					exit()
			# clock affects how often the screen is updated, a higher FPS will make the snake move faster
			clock.tick(FPS)

			#define state for Q function
			curSnakeHead, curSnakeBody, curApple = copy.deepcopy(self.getHead()), copy.deepcopy(self.snakeBody), copy.deepcopy(self.apple)
			state = State(curSnakeHead, curSnakeBody, curApple)
			self.changeDirection(state)
			attachHead = moveHead(self.getHead(), self.direction)
			apple = False
			collision = False


			# Add new section for head
			self.snakeBody.insert(0, attachHead)
			#new state for incorporate feedback
			newState = State(self.getHead(), self.snakeBody, self.apple)

			# checks with collision with apple
			if self.snakeBody[0]['x'] == self.apple.getX() and self.snakeBody[0]['y'] == self.apple.getY():
				#randomizes the location of the next apple
				self.apple.setRandomX()
				self.apple.setRandomY()
				self.apple.drawApple(win)
				self.drawSnake(win)
				pygame.display.update()
				apple = True

			else:
				# removes last element, if no apple eaten
				self.snakeBody.pop()
				drawWin(win)
				self.apple.drawApple(win)
				self.drawSnake(win)
				pygame.display.update()

			# reset after snake hits border
			if self.snakeBody[0]['x'] <= -1 or self.snakeBody[0]['x'] >= windowWidth or self.snakeBody[0]['y'] <= -1 or self.snakeBody[0]['y'] >= windowHeight:
				collision = True
				reward = self.model.getReward(state, self.direction, apple, collision)
				#incorporate feedback
				self.model.incorporateFeedback(state, self.direction, reward, newState, possibleDirections(self.direction))
				self.endGame()
				if self.iteration >= self.iterationLimit:
					return
				self.move(win, FPS)

			else:
				# reset after snake hits itself
				for seg in self.snakeBody[1:]:
					if seg['x'] == self.snakeBody[0]['x'] and seg['y'] == self.snakeBody[0]['y']:
						collision = True
						reward = self.model.getReward(state, self.direction, apple, collision)
						#incorporate feedback
						newState = State(self.getHead(), self.snakeBody, self.apple)
						self.model.incorporateFeedback(state, self.direction, reward, newState, possibleDirections(self.direction))
						self.endGame()
						if self.iteration >= self.iterationLimit:
							return
						self.move(win, FPS)

			reward = self.model.getReward(state, self.direction, apple, collision)
			self.model.incorporateFeedback(state, self.direction, reward, newState, possibleDirections(self.direction))
			if self.iteration >= self.iterationLimit:
				return


    #picks action based of E-greedy algorithm
	def changeDirection(self, state):
		newDirection = self.model.getAction(state, possibleDirections(self.direction))
		self.direction = newDirection

	# score is the number of added segments
	def getScore(self):
		self.score = len(self.snakeBody) - 1
		return self.score

	def getHead(self):
		return self.snakeBody[0]

#snake can't go in opposite of current direction
def getOppositeDirection(direction):
	if direction == UP:
		return DOWN
	elif direction == DOWN:
		return UP
	elif direction == LEFT:
		return RIGHT
	elif direction == RIGHT:
		return LEFT

# find location of new head
def moveHead(head, direction):
	if direction == UP:
		return {'x': head['x'], 'y': head['y'] - cellSize}

	elif direction == DOWN:
		return {'x': head['x'], 'y': head['y'] + cellSize}

	elif direction == RIGHT:
		return {'x': head['x'] + cellSize, 'y': head['y']}

	elif direction == LEFT:
		return {'x': head['x'] - cellSize, 'y': head['y']}

def possibleDirections(curDirection):
	possibleDirections = []
	for dir in directions:
		if dir != getOppositeDirection(curDirection):
			possibleDirections.append(dir)
	return possibleDirections

class Apple:

	def __init__(self):
		# red apple
		self.color = (255,0,0)
		self.x = 0
		self.y = 0

	def drawApple(self, win):

		# apple is red cube
		self.apple = pygame.Rect(self.x, self.y, cellSize, cellSize)
		pygame.draw.rect(win, self.color, self.apple)

		pygame.display.update()

	# randomly selects x location for apple
	def setRandomX(self):
		self.x = random.randint(2, columnCount - 1) * cellSize

	def getX(self):
		return self.x

	# randomly selects y location for apple
	def setRandomY(self):
		self.y = random.randint(2, rowCount - 1) * cellSize

	def getY(self):
		return self.y

class State:
	def __init__(self, snakeHead, snakeBody, apple):
		self.snakeHead = snakeHead
		self.snakeBody = snakeBody
		self.apple = apple

class Model:
	def __init__(self):
		self.discount = 0.95 #adjust
		self.weights = defaultdict(float)
		self.features = []
		self.iterations = 0

	#Q function score
	def getQ(self, state, action):
		score = 0
		for feature, value in self.getFeatures(state, action):
			score += self.weights[feature] * value
		return score

	#gets action based off E-greedy algorithm
	def getAction(self, state, actions):
		self.iterations += 1
		if self.getExploration() < random.random():
			action = max((self.getQ(state, action), action) for action in actions)[1]
			return action
		else:
			action = random.choice(actions)
			return action

	def getAlpha(self):
		return 1.0 / math.sqrt(self.iterations)

	#dynamic exploration probability
	def getExploration(self):
		rate = 0.5 - (0.01 * self.iterations)
		if rate >= 0:
			return rate
		else:
			return 0.01

	'''
	Features of given state (formed into a tuple):

		1) Apple above snake
		2) Apple below snake
		3) Apple to the left of snake
		4) Apple to the right of snake

		5) Danger above snake
		6) Danger below snake
		7) Danger to the left of snake
		8) Danger to the right of snake

		9) Direction = Up
		10) Direction = Down
		11) Direction = Left
		12) Direction = Right

	'''
	def getFeatures(self, state, action):
		features = []

		appleVert = state.apple.getY() - state.snakeHead['y']
		if appleVert < 0:
			features.append(1)
		else:
			features.append(0)
		if appleVert > 0:
			features.append(1)
		else:
			features.append(0)

		appleHor = state.apple.getX() - state.snakeHead['x']
		if appleHor < 0:
			features.append(1)
		else:
			features.append(0)
		if appleHor > 0:
			features.append(1)
		else:
			features.append(0)

		for key in ['y', 'x']:
			for num in [-cellSize, cellSize]:
				point = copy.deepcopy(state.snakeHead)
				point[key] = point.get(key) + num
				if point[key] < 0 or point[key] >= windowHeight or point in state.snakeBody:
					features.append(1)
				else:
					features.append(0)

		for dir in directions:
			if dir == action:
				features.append(1)
			else:
				features.append(0)

		features = tuple(features)

		return [Feature(featureKey=features, featureValue=1)]


	def getReward(self, state, action, apple, collision):
		'''
		1) +30 for getting apple
		2) -50 for dying
		3) +1 for getting closer to apple, -1 for going further from apple
		'''
		reward = 0
		newHeadLocation = moveHead(state.snakeHead, action)

		#reward of getting apple
		if apple:
			reward += 30

		#dying
		if collision:
			reward -= 50

		#reward getting closer to apple
		Xdist = (abs(state.apple.getX() - state.snakeHead['x']) - abs(state.apple.getX() - newHeadLocation['x'])) / cellSize
		Ydist = (abs(state.apple.getY() - state.snakeHead['y']) - abs(state.apple.getY() - newHeadLocation['y'])) / cellSize

		reward += Xdist + Ydist

		return reward

	def incorporateFeedback(self, state, action, reward, newState, actions):
		features = self.getFeatures(state, action)
		maxQ = max((self.getQ(newState, action), action) for action in actions)[0]
		for featureKey, featureValue in features:
			self.weights[featureKey] += self.getAlpha() * ((reward + (self.discount * maxQ)) - self.getQ(state, action)) * featureValue

# draws window for snake
def drawWin(win):

	# black background
	win.fill((0,0,0))

	x = 0
	y = 0

	# draws the grid using lines across the window
	for line in range(windowWidth // cellSize):
		x = x + cellSize
		pygame.draw.line(win, (255,255,255), (x,0), (x, windowWidth))
		y = y + cellSize
		pygame.draw.line(win, (255,255,255), (0,y), (windowHeight, y))

	pygame.display.update()

def main():

	#user specified FPS and number of game iterations
	parser = argparse.ArgumentParser(description = "Snake Game with Reinforcement Learning")
	parser.add_argument("-FPS",
						help = "Specifies the speed of game. Example: -FPS=15",
						required = False,
						default = 25,
						type = int)
	parser.add_argument("-iterations",
						help = "Specifies the number of iterations of the game. Example: -iterations=150",
						required = False,
						default = 200,
						type = int)
	args = parser.parse_args()

	#initializes the game
	pygame.init()

	# defines the windows
	win = pygame.display.set_mode((windowHeight, windowWidth))

	# draws game window
	drawWin(win)

	plt.ion()
	fig, ax = plt.subplots()

	# initializes apple
	apple = Apple()
	apple.setRandomX()
	apple.setRandomY()
	apple.drawApple(win)

	# initializes snake
	snake = Snake(apple, args.iterations, fig, ax)
	snake.drawSnake(win)

	# moves snake until there is collision with self/border
	snake.move(win, args.FPS)

	#keep windows open until program is exited
	print("Click red 'x' on pygame window to terminate program.")
	while True:
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				exit()

if __name__ == '__main__':
	main()
