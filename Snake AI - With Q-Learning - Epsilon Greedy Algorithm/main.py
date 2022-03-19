from snake_game_ai import SnakeGame

if __name__ == '__main__':
    game = SnakeGame()
    cnt = 0
    try:
        while cnt <= 150:
            game_over, score = game.play_step()
            if game_over:
                print(f'Game Over! Your score is {score}.\n')
                cnt += 1
                break
    except KeyboardInterrupt:
        print('\n\nExiting...')
        print(f'Your score is {score}.\n')
        quit()
