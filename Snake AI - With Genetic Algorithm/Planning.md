# The Snake (Individual)

- The Snake will be a subclass of Individual Class

- We will also initialize our Neural Network Structure for every snake separately, while initializing the snake.

- This neural network structure will have an input array made up of vision inputs and the current direction input and optional the tail direction input as well.

- The Snake will be given $8 * 3 + 4$ inputs i.e
    1. Will have vision in 8 directions.

        1. The vision will be able to find the distance between its head and the wall in the given direction.
        2. It will tell if the food is visible in that direction or not.
        3. It will tell if the snake's body is visible in that direction or not.
    2. The current Direction of the snakes head.
        1. It will be **one-hot encoding** as a $4$ vector, $1$ if its moving in that direction or zero otherwise.