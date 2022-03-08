import pygame
import Map
import neat
from neat import nn, population
import sys
import pickle
import math
import Snake
import random

bestFitness = 0
generationNumber = 0
blockSize = 20
width = 30
height = 30
screenSize = (width * blockSize, height*blockSize)
gameSpeed = 250
clock = pygame.time.Clock()
snake = Snake.Snake(7, 2, 10)
foodPos = [random.randint(0, width-1), random.randint(0, height-1)]

# best_fitness = 0

pygame.init()
screen = pygame.display.set_mode((width*blockSize, height*blockSize))

pygame.time.set_timer(pygame.USEREVENT, gameSpeed)
scr = pygame.surfarray.pixels2d(screen)


def drawSnake():
    for x, y in snake.body:
        pygame.draw.rect(screen, (255, 255, 255), [blockSize*x, blockSize*y, blockSize-4, blockSize-4])


def wallCollision():
    x, y = snake.body[0]
    if x < 0 or x > width-1 or y < 0 or y > height-1:
        return True
    return False


def positivity(x):
    if x > 0:
        return x
    return 0


def getGameMat():
    global foodPos, snake

    mat = []

    for i in range(0, height):
        mat.append([0 for j in range(0, width)])

    mat[foodPos[1]][foodPos[0]] = 2
    for x, y in snake.body:
        mat[y][x] = 1

    return mat


# dist to wall, food, and tail - look ahead, left, and right


def getInputs():
    global snake, foodPos

    headX, headY = snake.body[0]
    tailX, tailY = snake.body[-1]
    distWestWall = headX
    distEastWall = width - headX
    distNorthWall = headY
    distSouthWall = height - headY

    distFoodX = foodPos[0] - headX
    distFoodY = foodPos[1] - headY

    distTailX = tailX - headX
    distTailY = tailY - headY

    distTailAhead = 0
    distTailLeft = 0
    distTailRight = 0

    distFoodLeft = 0
    distFoodRight = 0
    distFoodAhead = 0

    # left
    if snake.dx == -1:
        distWallAhead = distWestWall
        distWallLeft = distSouthWall
        distWallRight = distNorthWall

        distFoodAhead = -distFoodX

        if distFoodY > 0:
            distFoodLeft = distFoodY
            distFoodRight = -distFoodY
        elif distFoodY < 0:
            distFoodLeft = -distFoodY
            distFoodRight = distFoodY

        distTailAhead = -distTailX

        if distTailY > 0:
            distTailLeft = distTailY
            distTailRight = -distTailY
        elif distTailY < 0:
            distTailLeft = -distTailY
            distTailRight = distTailY

    # right
    elif snake.dx == 1:
        distWallAhead = distEastWall
        distWallLeft = distNorthWall
        distWallRight = distSouthWall

        distFoodAhead = distFoodX

        if distFoodY > 0:
            distFoodLeft = -distFoodY
            distFoodRight = distFoodY
        elif distFoodY < 0:
            distFoodLeft = distFoodY
            distFoodRight = -distFoodY

        distTailAhead = distTailX

        if distTailY > 0:
            distTailLeft = -distTailY
            distTailRight = distTailY
        elif distTailY < 0:
            distTailLeft = distTailY
            distTailRight = -distTailY

    # up
    elif snake.dy == -1:
        distWallAhead = distNorthWall
        distWallLeft = distWestWall
        distWallRight = distEastWall

        distFoodAhead = -distFoodY

        if distFoodX > 0:
            distFoodRight = distFoodX
            distFoodLeft = -distFoodX
        elif distFoodX < 0:
            distFoodRight = -distFoodX
            distFoodLeft = distFoodX

        distTailAhead = -distTailY

        if distTailX > 0:
            distTailRight = distTailX
            distTailLeft = -distTailX
        elif distTailX < 0:
            distTailRight = -distTailX
            distTailLeft = distTailX

    # down
    else:
        distWallAhead = distSouthWall
        distWallLeft = distEastWall
        distWallRight = distWestWall

        distFoodAhead = distFoodY

        if distFoodX > 0:
            distFoodRight = -distFoodX
            distFoodLeft = distFoodX
        elif distFoodX < 0:
            distFoodRight = distFoodX
            distFoodLeft = -distFoodX

        distTailAhead = distTailY

        if distTailX > 0:
            distTailRight = -distTailX
            distTailLeft = distTailX
        elif distTailX < 0:
            distTailRight = distTailX
            distTailLeft = -distTailX

    # return [distWallAhead, distWallLeft, distWallRight, distFoodX, distFoodY]
    return [distWallAhead, distFoodAhead, distTailAhead,
            distWallLeft, distFoodLeft, distTailLeft,
            distWallRight, distFoodRight, distTailRight]


def turnLeft():
    global snake, direction

    if snake.dx == -1:
        snake.dx = 0
        snake.dy = 1
    elif snake.dx == 1:
        snake.dx = 0
        snake.dy = -1
    elif snake.dy == 1:
        snake.dx = 1
        snake.dy = 0
    elif snake.dy == -1:
        snake.dx = -1
        snake.dy = 0
    direction = (direction-1) % 4


def turnRight():
    global snake, direction

    if snake.dx == -1:
        snake.dx = 0
        snake.dy = -1
    elif snake.dx == 1:
        snake.dx = 0
        snake.dy = 1
    elif snake.dy == 1:
        snake.dx = -1
        snake.dy = 0
    elif snake.dy == -1:
        snake.dx = 1
        snake.dy = 0
    direction = (direction+1) % 4


def reset():
    global snake, foodPos, direction
    snake = Snake.Snake(7, 2, 10)
    foodPos = [random.randint(0, width-1), random.randint(0, height-1)]
    snake.dx = 1
    snake.dy = 0
    direction = 1


def eval_fitness(genomes, config):
    global bestFitness, screen, width, height, blockSize, scr, generationNumber, pop, foodPos

    genomeNumber = 0
    for id, g in genomes:
        speed = 10
        score = 0.0
        hunger = 100
        net = nn.FeedForwardNetwork.create(g, config)
        reset()
        pygame.time.set_timer(pygame.USEREVENT, speed)
        error = 0
        foodDist = -1

        while True:

            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.USEREVENT:
                mat = getGameMat()
                inputs = getInputs()
                outputs = net.activate(inputs)
                moveDirection = outputs.index(max(outputs))
                # 0 is stay straight, 1 is turn left, 2 is turn right
                if moveDirection == 1:
                    turnLeft()
                if moveDirection == 2:
                    turnRight()

                hunger -= 1

                if not snake.move() or wallCollision() or hunger <= 0:
                    break
                else:
                    # reward moving towards the food
                    newDist = math.sqrt((foodPos[0] - snake.body[0][0]) ** 2 + (foodPos[1] - snake.body[0][1]) ** 2)
                    if newDist < foodDist:
                        score += 50
                    else:
                        score -= 75

                    foodDist = newDist

            if foodPos in snake.body:
                print("ate food")
                snake.grow()
                speed -= 5
                foodPos = [random.randint(1, width - 1), random.randint(1, height - 1)]
                hunger = 200

                # reward eating the food
                score += 250

            screen.fill((0, 0, 0))
            pygame.draw.rect(screen, (255, 0, 0), [blockSize * foodPos[0], blockSize * foodPos[1], blockSize - 4, blockSize - 4])
            drawSnake()
            pygame.display.update()

        print("Score:", score)
        score = positivity(score)
        g.fitness = positivity((-1/((math.sqrt(score+1))/10)) + 1)
        print("Error:", error)
        bestFitness = max(bestFitness, g.fitness)
        print("Generation:", generationNumber, "\tGenome:", genomeNumber, "\tFitness", g.fitness, "\tBest Fitness:", bestFitness)
        genomeNumber += 1
    generationNumber += 1


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, "config-feedforward")

pop = neat.Population(config)
pop.run(eval_fitness, 100)
