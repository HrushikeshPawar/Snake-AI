from helper import VISION16, VISION8, VISION4, Direction, Point


def check_helper_functions():

    print('Vision16:')
    for vis in VISION16:
        print(vis)

    print('\nVision8:')
    for vis in VISION8:
        print(vis)

    print('\nVision4:')
    for vis in VISION4:
        print(vis)

    print('\nDirection Mul:')
    print(Direction.UP * 2)
    print(Direction.UP * -2)

    print('\nDirection Add:')
    print(Point(1, 2) + Point(3, 4))


if __name__ == '__main__':
    check_helper_functions()
