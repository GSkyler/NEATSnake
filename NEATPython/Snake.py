class Snake:

    def __init__(self, x, y, blockSize):
        self.length = 5
        self.x = x
        self.y = y
        self.body = []
        for i in range(0, 5):
            self.body.append([x-i, y])

        self.blockSize = blockSize
        self.dx = 1
        self.dy = 0
        self.growing = False

    def move(self):
        if not self.growing:
            self.body.pop()
        else:
            self.growing = False

        x, y = self.body[0]
        newX = x + self.dx
        newY = y + self.dy

        self.body.insert(0, [newX, newY])

        if self.body[0] in self.body[1:]:
            return False
        return True

    def grow(self):
        self.growing = True
