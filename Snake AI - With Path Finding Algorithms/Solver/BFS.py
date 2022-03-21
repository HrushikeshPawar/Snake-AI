from Snake_Game import BLOCKSIZE, GRID_H, GRID_W, Game, Point, Direction, Snake
# from copy import deepcopy
from numpy import array
from copy import deepcopy


class BSF_Solver:

    def __init__(self, game: Game) -> None:

        self.game   = game
        self._grid  = None
        self.update_grid()
        self._images = []

    def update_grid(self) -> None:

        self._grid = [[0 for _ in range(GRID_H)] for _ in range(GRID_W)]
        # self._grid = [[0 for _ in range(GRID_W)] for _ in range(GRID_H)]
        # self._grid = zeros((GRID_H, GRID_W), dtype=object)

        head = self.game.snake.head
        # print('Head: ', head, head.x // BLOCKSIZE, head.y // BLOCKSIZE)
        self._grid[head.x // BLOCKSIZE][head.y // BLOCKSIZE] = 1

        for p in self.game.snake.tail:
            self._grid[p.x // BLOCKSIZE][p.y // BLOCKSIZE] = 'X'
        if len(self.game.snake.tail) > 0:
            tail_tip = self.game.snake.tail[-1]
            self._grid[tail_tip.x // BLOCKSIZE][tail_tip.y // BLOCKSIZE] = 'T'

        food = self.game.food
        # print('Food: ', food, food.x // BLOCKSIZE, food.y // BLOCKSIZE, '\n')
        self._grid[food.x // BLOCKSIZE][food.y // BLOCKSIZE] = 'F'

    def _get_neighbours(self, point: tuple[int, int], walls: list) -> None:

        x, y = point
        neighbours = []

        if x > 0 and self._grid[x - 1][y] not in walls:
            neighbours.append((x - 1, y))
        if x < GRID_W - 1 and self._grid[x + 1][y] not in walls:
            neighbours.append((x + 1, y))
        if y > 0 and self._grid[x][y - 1] not in walls:
            neighbours.append((x, y - 1))
        if y < GRID_H - 1 and self._grid[x][y + 1] not in walls:
            neighbours.append((x, y + 1))

        return neighbours

    def _make_step(self, walls: list, k: int = 1, dest_letter: str = 'F') -> None:
        flag = 0
        for i in range(GRID_W):
            for j in range(GRID_H):
                if self._grid[i][j] == k:

                    neighbours = self._get_neighbours((i, j), walls)
                    for x, y in neighbours:
                        if self._grid[x][y] == 0 or self._grid[x][y] == dest_letter:
                            flag += 1
                            self._grid[x][y] = k + 1

        if flag == 0:
            # self._print_grid(self._grid)
            pass

        else:
            # print(f'{flag} steps performed!')
            pass

    def _find_shortest_path_BSF(self, dest: tuple[int, int], walls: list = ['X', 'T'], dest_letter: str = 'F') -> list:
        """
        Find shortest path using Breadth First Search Algorithm.
        """

        x, y = dest
        # print(dest, (self.game.food.x // BLOCKSIZE, self.game.food.y // BLOCKSIZE), self._grid[x][y])

        k = 0
        while self._grid[x][y] == dest_letter and k <= GRID_W * GRID_H:
            k += 1
            self._make_step(walls=walls, k=k, dest_letter=dest_letter)
            # print(k, self._grid[x][y])
        if k == 0:
            print('While didnt run')
            pass
        # self._print_grid(self._grid)

        path = [Point(x, y)]
        # path = [Point(y, x)]

        if isinstance(self._grid[x][y], str):
            return []

        while self._grid[x][y] > 1:

            neighbours = self._get_neighbours((x, y), walls)
            for i, j in neighbours:
                if self._grid[i][j] == self._grid[x][y] - 1:
                    path.append(Point(i, j))
                    # path.append(Point(j, i))
                    x, y = i, j
                    break
        return path

    def _mark_the_path(self, path: list) -> list[list]:

        Grid = array(self._grid, dtype=object)  # .transpose()
        for point in path:
            x, y = point.x, point.y
            Grid[x][y] = '+'

        return Grid.transpose()

    def _get_direction(self, p: Point) -> Direction:

        if p == Point(0, -1):
            return Direction.UP
        elif p == Point(0, 1):
            return Direction.DOWN
        elif p == Point(-1, 0):
            return Direction.LEFT
        elif p == Point(1, 0):
            return Direction.RIGHT

    def _print_grid(self, grid: list[list]) -> None:
        for row in grid:
            for elm in row:
                print(f"{elm: >2}", end=' ')
            print()
        print('\n\n')

    def next_move(self) -> None:
        self.update_grid()
        dest = (self.game.food.x // BLOCKSIZE, self.game.food.y // BLOCKSIZE)
        path = self._find_shortest_path_BSF(dest)
        Grid = self._mark_the_path(path)

        if path == []:
            print('No path found!')
            game_over, score = self.game.play_step(self.game.snake.direction)
            return game_over, score

        dirs = []
        for i in range(len(path) - 1):
            dirs.append(self._get_direction(path[i] - path[i + 1]))
        dirs = dirs[::-1]

        self._print_grid(Grid)

        for dir in dirs:
            game_over, score = self.game.play_step(dir)

        return game_over, score


# Doesn't Work AS EXPECTED - Needs to be fixed
class BSF_Modified_Solver(BSF_Solver):

    def __init__(self, game: Game) -> None:
        super().__init__(game)

    def _is_tail_tip_reachable(self, tail_tip: Point) -> bool:

        x, y = tail_tip.x // BLOCKSIZE, tail_tip.y // BLOCKSIZE

        k = 0
        while self._grid[x][y] == 'X' and k <= GRID_W * GRID_H:
            k += 1
            self._make_step(k=1, walls=['X'], dest_letter='T')
        self._print_grid(array(self._grid, dtype=object).transpose())
        print('\n\n')
        return not k == GRID_W * GRID_H + 1

    def _get_directions(self, walls: list = ['X', 'T'], dest_letter='F') -> list[Direction]:
        self.update_grid()
        dest = (self.game.food.x // BLOCKSIZE, self.game.food.y // BLOCKSIZE)
        path = self._find_shortest_path_BSF(dest=dest, walls=walls, dest_letter=dest_letter)

        if path == []:
            print('No path found!')
            return []

        dirs = []
        for i in range(len(path) - 1):
            dirs.append(self._get_direction(path[i] - path[i + 1]))

        return dirs[::-1]

    def _duplicate_grid(self, snake: Snake) -> array:

        Grid = [[0 for _ in range(GRID_H)] for _ in range(GRID_W)]
        # Grid = zeros((GRID_H, GRID_W), dtype=object)

        head = snake.head
        # print('Head: ', head, head.x // BLOCKSIZE, head.y // BLOCKSIZE)
        Grid[head.x // BLOCKSIZE][head.y // BLOCKSIZE] = 1

        for p in snake.tail:
            Grid[p.x // BLOCKSIZE][p.y // BLOCKSIZE] = 'X'

        food = self.game.food
        # print('Food: ', food, food.x // BLOCKSIZE, food.y // BLOCKSIZE, '\n')
        Grid[food.x // BLOCKSIZE][food.y // BLOCKSIZE] = 'F'

        return array(Grid, dtype=object).transpose()

    def _is_next_move_safe(self, dirs: list) -> bool:
        print('Function Called!')
        future_snake = deepcopy(self.game.snake)
        for dir in dirs:
            future_snake.move(dir)
            future_snake.tail.pop()
        Dup_Grid    = self._duplicate_grid(future_snake)
        tail_tip    = future_snake.tail[-1]
        # self._print_grid(Dup_Grid)

        return self._is_tail_tip_reachable(tail_tip, Dup_Grid)

    def _make_safe_move(self, dirs: list) -> None:

        for dir in dirs:
            game_over, score = self.game.play_step(dir)
        return game_over, score

    def _get_nearest_edge(self) -> Direction:

        head = Point(self.game.snake.head.x // BLOCKSIZE, self.game.snake.head.y // BLOCKSIZE)
        dup_snake = deepcopy(self.game.snake)
        edges = {
            f'{head.x}_LEFT'          : (Direction.LEFT, 0, 0),
            f'{head.y}_UP'            : (Direction.UP, 1, 0),
            f'{GRID_W - head.x}_RIGHT': (Direction.RIGHT, 0, (GRID_W - 1) * BLOCKSIZE),
            f'{GRID_H - head.y}_DOWN' : (Direction.DOWN, 1, (GRID_H - 1) * BLOCKSIZE)
        }

        flag = False
        options = list(edges.keys())
        list.sort(options, key=lambda x: int(x.split('_')[0]))
        options.sort()
        while True:  # and options:
            self._print_grid(self._duplicate_grid(dup_snake))
            dup_snake = deepcopy(self.game.snake)
            print(options)
            edge, index, boundary = edges[options.pop(0)]
            if boundary == 0:
                while dup_snake.head[index] > 0:
                    # print(edge, index, boundary, dup_snake.head, dup_snake.tail)
                    dup_snake.move(edge)
                    dup_snake.tail.pop()
                    if dup_snake.head in dup_snake.tail:
                        flag = True
                        break
            else:
                while dup_snake.head[index] < boundary:
                    # print(edge, index, boundary, dup_snake.head, dup_snake.tail)
                    dup_snake.move(edge)
                    dup_snake.tail.pop()
                    if dup_snake.head in dup_snake.tail:
                        flag = True
                        break

            if not flag:
                return edge, index, boundary

    def _move_to_edges(self) -> None:

        direction, index, boundary = self._get_nearest_edge()
        print(direction, index, boundary)
        path = [self.game.snake.head]
        while path[-1][index] != boundary:
            self.game.snake.move(direction)
            path.append(self.game.snake.head)
        self._print_grid(self._grid)
        self.update_grid()

    def _move_to_tail(self) -> None:

        self.update_grid()
        tail_tip = (self.game.snake.tail[-1].x // BLOCKSIZE, self.game.snake.tail[-1].y // BLOCKSIZE)
        path = self._find_shortest_path_BSF(dest=tail_tip, walls=['X'], dest_letter='T')
        x, y = tail_tip

        if path == []:
            print('No path found to tail tip!', self._grid[x][y])
            return []

        dirs = []
        for i in range(len(path) - 1):
            dirs.append(self._get_direction(path[i] - path[i + 1]))

        dirs = dirs[::-1]

        if dirs == []:
            return self.game.play_step(self.game.snake.direction)
        else:
            return self._make_safe_move(dirs)

    def next_move(self) -> None:

        # Get directions for next move
        dirs = self._get_directions()
        if dirs == []:
            return self.game.play_step(self.game.snake.direction)

        if self.game.snake.length < min(GRID_H, GRID_W):
            return self._make_safe_move(dirs)

        else:
            print('Moving to tail!')
            print('Is Tail Tip Reacable: ', self._is_tail_tip_reachable(self.game.snake.tail[-1]))
            self._move_to_tail()
            dirs = self._get_directions()
            if dirs == []:
                return self.game.play_step(self.game.snake.direction)
            return self._make_safe_move(dirs)
