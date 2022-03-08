import pygame


class Map:
    color = (0, 0, 0)

    def __init__(self, screen, width, height, blockSize, color=12800):
        self.screen = screen
        self.width = width
        self.height = height
        self.blockSize = blockSize
        self.color = ((color & 0xFF0000) / 256 **2, ((color & 0x00FF00) / 256) % 256, color & 0x0000FF)

    def draw(self):
        self.screen.fill(self.color)
