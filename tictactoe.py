board = [['_'] * 3 for _ in range(3)]

def print_board():
    for row in board:
        print(' '.join(row))

def check_winner():
    lines = [
        board[0],
        board[1],
        board[2],
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        [board[0][0], board[1][1], board[2][2]],
        [board[0][2], board[1][1], board[2][0]],
    ]
    for line in lines:
        if line == ['X', 'X', 'X']:
            return 'X'
        elif line == ['O', 'O', 'O']:
            return 'O'
    return None

def place_mark(x, y, char):
    if board[x][y] == '_':
        board[x][y] = char
        return True
    return False

def main():
    current_player = 'X'
    done = False
    print_board()
    while not done:
        try:
            x = int(input(f'Player {current_player}, enter row (0-2): '))
            y = int(input(f'Player {current_player}, enter column (0-2): '))
            if place_mark(x, y, current_player):
                print_board()
                winner = check_winner()
                if winner:
                    print(f'Player {winner} wins!')
                    done = True
                elif all(cell != '_' for row in board for cell in row):
                    print("It's a tie!")
                    done = True
                current_player = 'O' if current_player == 'X' else 'X'
            else:
                print('That spot is taken. Try again.')
        except ValueError:
            print('Invalid input. Please enter numbers.')

if __name__ == '__main__':
    main()
