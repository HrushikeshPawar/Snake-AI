from glob import glob
# from PIL import Image
from Solver.BFS import BSF_Solver  # , BSF_Modified_Solver
from Snake_Game import Game  # , BLOCKSIZE, Point
import os
import moviepy.editor as mpy
from Solver.A_Star import A_Star_Solver, Grid
from copy import deepcopy
# from numpy import array


def main():
    # game = Game(ishuman=True)
    game = Game(solver='BSF', save_gif=False)
    bsf = BSF_Solver(game)
    # bsf_mod = BSF_Modified_Solver(game)

    try:
        # old_score = 0
        while True:
            # game_over, score = bsf.game.play_step()

            # if score > old_score:

            #     dest = (bsf.game.food.x // BLOCKSIZE, bsf.game.food.y // BLOCKSIZE)
            #     bsf.update_grid()

            #     path = bsf.find_shortest_path_BSF(dest)
            #     Grid = bsf.mark_the_path(path)
            #     print(f'\n{path}\n')

            #     for row in Grid:
            #         for elm in row:
            #             print(f"{elm: >2}", end=' ')
            #         print()
            #     print('\n\n')
            #     old_score = score
            # action = bsf.next_move()
            # print(action)
            # game_over, score = bsf.game.play_step(action)
            game_over, score = bsf.next_move()
            # game_over, score = bsf_mod.next_move()

            if game_over:
                print(f'Game Over! Your score is {score}.\n')
                break
    except KeyboardInterrupt:
        print('\n\nExiting...')
        print(f'Your score is {score}.\n')
        quit()


def Test_Gif():
    # a, b, c, d = Point(x=9, y=6), Point(x=9, y=5), Point(x=9, y=4), Point(x=9, y=3)
    # print(a - b)
    # print(b - c)
    # print(c - d)

    imgs = glob(r'Pics/*.png')
    list.sort(imgs, key=lambda x: int(x.split('screenshot')[1].split('.png')[0]))
    gif_name = 'test.gif'

    # frames = []
    # for img in imgs:
    #     frames.append(Image.open(img))
    # frames[0].save('test.gif', format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
    # list.sort(imgs, key=lambda x: int(x.split('.png')[0]))
    with open('image_list.txt', 'w') as file:
        for item in imgs:
            file.write("%s\n" % item)
    os.system('convert @image_list.txt test.gif')
    clip = mpy.ImageSequenceClip(imgs, fps=60)
    clip.write_gif(gif_name, fps=60)


def Test_A_Star_Solver():
    SAVE_GIF = True
    game = Game(ishuman=False, solver='A_Star', save_gif=SAVE_GIF)
    grid = Grid(game.snake, game.food)
    a_star = A_Star_Solver(game=game, grid=grid)
    old_score = 0

    try:
        while True:
            # print('\n\n')
            # a_star.grid.generate_grid(a_star.game.snake, a_star.game.food)
            # print(a_star.grid)
            game_over, score = a_star.next_move()
            # print(a_star.grid)

            if score > old_score:
                # a_star.grid.grid = array(a_star.grid.grid).transpose()
                old_score = score

            if game_over:
                print(f'Game Over! Your score is {old_score}.\n')
                break
    except KeyboardInterrupt:
        print('\n\nExiting...')
        print(f'Your score is {old_score}.\n')
        if SAVE_GIF:
            game._save_gif()
        quit()


def Compare_BFS_and_A_Start_Solvers():
    SAVE_GIF = True
    game = Game(ishuman=False, solver='A_Star', save_gif=SAVE_GIF)
    snake, food = deepcopy(game.snake), deepcopy(game.food)
    grid = Grid(game.snake, game.food)
    a_star = A_Star_Solver(game=game, grid=grid)
    score = 0

    try:
        while True:
            game_over, score = a_star.next_move()
            if game_over:
                print(f'Game Over! Your score is {score}.\n')
                break
    except KeyboardInterrupt:
        print('\n\nExiting...')
        print(f'Your score for A* solver is {score}.\n')
        if SAVE_GIF:
            game._save_gif()

    game2 = Game(ishuman=False, solver='BSF', save_gif=SAVE_GIF)
    game2.snake, game2.food = snake, food
    bsf = BSF_Solver(game2)
    score = 0
    try:
        while True:
            game_over, score = bsf.next_move()
            if game_over:
                print(f'Game Over! Your score is {score}.\n')
                break
    except KeyboardInterrupt:
        print('\n\nExiting...')
        print(f'Your score for A* solver is {score}.\n')
        if SAVE_GIF:
            game2._save_gif()


if __name__ == '__main__':
    # main()
    # Test_A_Star_Solver()
    Compare_BFS_and_A_Start_Solvers()
