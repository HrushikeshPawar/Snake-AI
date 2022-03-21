from copy import deepcopy
from Snake_Game import BLOCKSIZE, GRID_H, GRID_W, Snake, Point, Game, Direction
# from numpy import array
from queue import PriorityQueue


class Grid:

    def __init__(
        self,
        snake: Snake,
        food: Point,
        grid_size: tuple[int, int] = (GRID_H, GRID_W),
        walls: list[str] = ['X', 'T'],
    ) -> None:

        self.rows, self.cols = grid_size
        self.walls = walls
        self.generate_grid(snake=snake, food=food)

    def generate_grid(self, snake: Snake, food: Point) -> None:
        """
        Generate the grid
        """
        self.start = (snake.head.x // BLOCKSIZE, snake.head.y // BLOCKSIZE)
        self.snake_tail = [(p.x // BLOCKSIZE, p.y // BLOCKSIZE) for p in snake.tail]
        self.end = (food.x // BLOCKSIZE, food.y // BLOCKSIZE)
        grid = []
        self.cells = []

        for row in range(self.rows):
            grid.append([])
            for col in range(self.cols):

                if (row, col) in self.snake_tail:
                    grid[row].append('X')

                elif (row, col) == self.start:
                    grid[row].append('H')

                elif (row, col) == self.end:
                    grid[row].append('F')
                    self.cells.append((row, col))

                else:
                    grid[row].append(' ')
                    self.cells.append((row, col))

        if len(self.snake_tail) > 0:
            row, col = self.snake_tail[-1]
            grid[row][col] = 'T'
            self.cells.append((row, col))

        # self.grid = array(grid, dtype=object).transpose()
        self.grid = grid
        # return self

    def __str__(self) -> None:

        grid_str = '# ' + '# ' * self.cols + '#\n'
        for row in range(self.rows):
            grid_str += '# '
            for col in range(self.cols):
                grid_str += self.grid[row][col] + ' '
            grid_str += '#\n'
        grid_str += '# ' + '# ' * self.cols + '#\n\n'

        return grid_str

    __repr__ = __str__


class A_Star_Solver:

    def __init__(self, grid: Grid, game: Game) -> None:
        self.game = game
        self.grid = grid
        self._reset_solver()

    def heuristic(self, start: tuple, end: tuple) -> int:
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    def _reset_solver(self) -> None:

        self.open = PriorityQueue()

        # The G Score is the cost of getting from the start node to a particular node.
        self.g_score = {cell: float('inf') for cell in self.grid.cells}
        self.g_score[self.grid.start] = 0

        # The F Score is the estimated total cost of getting from the start node to the goal
        self.f_score = {cell: float('inf') for cell in self.grid.cells}
        self.f_score[self.grid.start] = self.heuristic(self.grid.start, self.grid.end)

    def _get_neighbours(self, point: tuple[int, int], wall: list[str]) -> list[tuple[int, int]]:
        """
        Get the neighbours of a point
        """
        neighbours = []
        for row, col in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            neighbour = (point[0] + row, point[1] + col)

            if neighbour in self.grid.cells and self.grid.grid[neighbour[0]][neighbour[1]] not in wall:
                neighbours.append(neighbour)
        return neighbours

    def _reconstruct_path(self, rpath: dict, dest: tuple[int, int]) -> list:

        current = dest
        x, y = dest
        fwd_path = [Point(x, y)]

        # for key, value in rpath.items():
        #     print(f'{key} -> {value}')
        while current != self.grid.start:
            fwd_path.insert(0, rpath[current])
            current = rpath[current]

        # print()
        # for point in fwd_path:
            # print(f'{point} ', end=' -> ')
        # print()
        return fwd_path

    def search(self, dest: tuple[int, int] = None, wall: list[str] = ['X', 'T'], m: int = 1) -> list:

        # Update the grid
        self.grid.generate_grid(self.game.snake, self.game.food)
        # print('Grid Generated')

        if dest is None:
            dest = self.grid.end

        # Add the start node to the open list
        self._reset_solver()
        self.open.put((m * self.f_score[self.grid.start], m * self.f_score[self.grid.start], self.grid.start))
        rpath = dict()
        print(f'\nDestination: {dest}\nStart: {self.grid.start}',)

        # Loop until the open list is empty or the end node is found
        while not self.open.empty():

            current_cell = self.open.get()[2]

            # Check if the current cell is the end cell
            if current_cell == dest:
                break

            # Get the neighbours of the current cell
            # print('Generating Neighbours')
            for neighbour in self._get_neighbours(current_cell, wall):
                # print(f'Neighbour: {neighbour}', neighbour == dest)

                tmp_g_score = m * self.g_score[current_cell] + 1
                tmp_h_score = m * self.heuristic(neighbour, dest)
                tmp_f_score = tmp_h_score + tmp_g_score
                # print('Calculated Scores')
                # print(tmp_g_score, self.g_score[neighbour], tmp_g_score < self.g_score[neighbour])
                # If the neighbour is not in the open list or the neighbour is in the open list but the new path is better
                if (tmp_g_score < self.g_score[neighbour]):
                    self.g_score[neighbour] = tmp_g_score
                    self.f_score[neighbour] = tmp_f_score
                    self.open.put((tmp_f_score, tmp_h_score, neighbour))
                    cx, cy = current_cell
                    nx, ny = neighbour
                    rpath[Point(nx, ny)] = Point(cx, cy)
        # print(f'Is the path empty? {self.open.empty()} - Length of Path Space : {len(rpath)}')
        # print(f'Is Destination in Path? {dest in rpath}')
        if self.open.empty() and dest not in rpath:
            return []
        return self._reconstruct_path(rpath, dest)

    def _get_directions(self, path: list) -> Direction:

        directions = {
            Point(0, -1): Direction.UP,
            Point(0, 1): Direction.DOWN,
            Point(-1, 0): Direction.LEFT,
            Point(1, 0): Direction.RIGHT
        }

        # Convert the path into directions for our snake to follow
        # print('Generating Directions')
        dirs = []
        for i in range(len(path) - 1):
            dirs.append(directions[path[i + 1] - path[i]])
        # print('Directions Generated')

        return dirs

    def _check_if_path_is_safe(self, path: list) -> bool:

        if len(path) == 0:
            return False

        print('Checking if path is safe.')
        grid_copy = deepcopy(self.grid)
        snake_copy = deepcopy(self.game.snake)
        directions = self._get_directions(path)
        for direction in directions[:-1]:
            self.game.snake.move(direction)
            self.game.snake.tail.pop()
        self.game.snake.move(directions[-1])

        tail_tip = (self.game.snake.tail[-1].x // BLOCKSIZE, self.game.snake.tail[-1].y // BLOCKSIZE)
        print(f'Tail Tip: {tail_tip}')

        path = self.search(tail_tip, ['X'])
        # if len(path) == 0:
        #     print(self._get_neighbours(tail_tip, ['X']))
        for direction in self._get_directions(path):
            self.game.snake.move(direction)
            self.game.snake.tail.pop()

        # self.grid.generate_grid(self.game.snake, self.game.food)
        # print(self.grid)

        self.grid = deepcopy(grid_copy)
        self.game.snake = deepcopy(snake_copy)

        # self.grid.generate_grid(self.game.snake, self.game.food)
        # print(self.grid)

        return len(path) > 0

    def _move_snake_along_path(self, path: list) -> tuple[bool, int]:

        for direction in self._get_directions(path):
            game_over, score = self.game.play_step(direction)

        return game_over, score

    def _move_snake_by_one_block(self) -> list:
        print('Tail too close to Head. Moving one place away from tail.')
        grid_copy = deepcopy(self.grid)
        snake_copy = deepcopy(self.game.snake)

        self.game.snake.move(self.game.snake.direction)
        self.game.snake.tail.pop()

        head_x, head_y = (self.game.snake.head.x // BLOCKSIZE, self.game.snake.head.y // BLOCKSIZE)
        head = Point(head_x, head_y)

        if not self.game._is_collision():
            self.game.snake = snake_copy
            self.grid = grid_copy
            path = [head, head + self.game.snake.direction.value]
            print(f'Moving Snake: {self.game.snake.direction}')
        else:

            self.game.snake = deepcopy(snake_copy)
            self.grid = deepcopy(grid_copy)
            directions = [Direction.DOWN, Direction.UP, Direction.LEFT, Direction.RIGHT]
            for direction in directions:
                print(direction, snake_copy.head, self.game.snake.head)
                self.game.snake.move(direction)
                self.game.snake.tail.pop()
                if not self.game._is_collision():
                    self.game.snake = snake_copy
                    self.grid = grid_copy
                    path = [head, head + direction.value]
                    print(f'Moving Snake: {direction}')
                    break
                else:
                    self.game.snake = deepcopy(snake_copy)
                    self.grid = deepcopy(grid_copy)
        return path

    def _move_towards_tail(self) -> None:

        # Get the path to the tail
        tail_tip = (self.game.snake.tail[-1].x // BLOCKSIZE, self.game.snake.tail[-1].y // BLOCKSIZE)
        print('Taking a longer path towards tail.')
        path = self.search(tail_tip, ['X'], m=-1)

        if len(path) == 0:
            path = self._move_snake_by_one_block()

        # print(f'Path to Tail: {path}')
        return self._move_snake_along_path(path)

    def next_move(self):

        # Search for the path
        # print('Path Seach Started')
        path = self.search()
        # print('Path Seach Completed')
        print(f'Path Length: {len(path)}')

        # Path is safe and Non-empty
        if self._check_if_path_is_safe(path) and len(path) > 0 and self.game.snake.length < GRID_H * GRID_W // 2:
            print('Current path is safe, moving along the path.')
            return self._move_snake_along_path(path)

        #  Path is non-empty but unsafe
        elif len(path) > 0:
            print('Current path is unsafe, trying a longer path.')
            path = self.search(m=-1)

            # If this path is also unsafe, then move towards the tail
            if self._check_if_path_is_safe(path):
                print('The longer path is safe, moving along the longer path.')
                return self._move_snake_along_path(path)
            else:
                print('The longer path is unsafe, moving towards the tail.')
                return self._move_towards_tail()

        # If not path exists, move towards the tail
        else:
            print('No, path exists, hence moving towards tail.')
            return self._move_towards_tail()
        # if path == []:
        #     print('No path found! Hence moving towards tail.')
        #     # game_over, score = self.game.play_step(self.game.snake.direction)
        #     # return game_over, score
        #     # print('Current path is not safe, hence moving towards the tail.')
        #     print(self.grid)
        #     return self._move_towards_tail()

        # # Check if the path is safe
        # if self.game.snake.length <= min(GRID_H, GRID_W):
        #     return self._move_snake_along_path(path)

        # else:
        #     is_path_safe = self._check_if_path_is_safe(path)
        #     print(f'Is current path safe?: {is_path_safe}')
        #     if is_path_safe:
        #         print('Current path is safe, moving along the path.')
        #         return self._move_snake_along_path(path)
        #     else:
        #         print('Current path is not safe, hence moving towards the tail.')
        #         return self._move_towards_tail()
